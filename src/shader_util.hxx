#pragma once

#include "vec3f.hxx"
#include "spectral.hxx"
#include "texture.hxx"
#include "sampler.hxx"
#include "ray.hxx"
#include "scene.hxx"

#include <numeric> // for iota on Windows
#include <boost/container/small_vector.hpp>




struct LambdaSelection
{
  Index3 indices;  // Wavelength bin indices.
  Spectral3 weights; // Monte-carlo weight comprising sensor sensitivity over selection probability.
  Wavelengths3 wavelengths; // The actual wavelengths
};


enum TransportType : char 
{
  RADIANCE,
  IMPORTANCE
};

struct PathContext
{
  PathContext() = default;
  PathContext(const LambdaSelection &_lambdas, TransportType _transport, int pixel_index_)
    : lambda_idx(_lambdas.indices), wavelengths{_lambdas.wavelengths}, transport(_transport), pixel_index{pixel_index_}
  {}
  PathContext(const LambdaSelection &_lambdas, int pixel_index_ = -1)
    : PathContext{_lambdas, RADIANCE, pixel_index_}
  {}
  Index3 lambda_idx {};
  Wavelengths3 wavelengths {};
  TransportType transport { RADIANCE };
  int pixel_index = -1;
};



struct LambdaSelectionStrategy
{
  /* Stratified mode: Divide spectrum in N sections, where N is the number of simultaneously traced wavelengths.
   * Pick one wavelength from each section. In my case I pick the wavelength from the first section at random,
   * and take the others at a fixed offset equal to the stratum size. */
  static constexpr int strata_size = Color::NBINS / Spectral3::RowsAtCompileTime;
  static_assert(Color::NBINS == strata_size * Spectral3::RowsAtCompileTime, "Bin count must be multiple of number of simultaneously traced wavelengths");
  
  SpectralN lambda_weights;
  LambdaSelectionStrategy()
  {
    // One over the probability, that the wavelength is selected.
    lambda_weights.setConstant(strata_size);
  }
  
  static Index3 MakeIndices(int main_idx)
  {
    return Index3{main_idx, main_idx+strata_size, main_idx+2*strata_size};
  }
  
  static int PrimaryIndex(const Index3 &idx)
  {
    return idx[0];
  }
  
  static Wavelengths3 SampleWavelengthStrata(const Index3 &bin_indices, Sampler &sampler)
  {
    Wavelengths3 wl;
    for (int i=0; i<static_size<Spectral3>(); ++i)
    {
      auto bounds  = Color::GetWavelengthBinBounds(bin_indices[i]);
      wl[i] = bounds.first + sampler.Uniform01()*(bounds.second-bounds.first);
    }
    return wl;
  }
  
  LambdaSelection WithWeights(Sampler &sampler) const
  {
    int main_idx = sampler.UniformInt(0, strata_size-1);
    auto idx     = MakeIndices(main_idx);
    auto weights = Take(lambda_weights, idx);
    return LambdaSelection{idx, weights, SampleWavelengthStrata(idx, sampler)};
  }
};


class LambdaSelectionStrategyShuffling
{
  /* Stratified mode: Divide spectrum in N sections, where N is the number of simultaneously traced wavelengths.
   * Pick one wavelength from each section. In my case I pick the wavelength from the first section at random,
   * and take the others at a fixed offset equal to the stratum size. */
  static constexpr int strata_size = Color::NBINS / Spectral3::RowsAtCompileTime;
  static_assert(Color::NBINS == strata_size * Spectral3::RowsAtCompileTime, "Bin count must be multiple of number of simultaneously traced wavelengths");
  std::array<int,strata_size> current_selection_permutation;
  int current_idx = strata_size;
  
  void Shuffle(Sampler &sampler)
  {
    RandomShuffle(
      current_selection_permutation.begin(),
      current_selection_permutation.end(),
      sampler);
  }
  
public:
  LambdaSelectionStrategyShuffling()
  {
    // One over the probability, that the wavelength is selected.
    std::iota(current_selection_permutation.begin(), current_selection_permutation.end(), 0);
  }
  
  static constexpr int NUM_SAMPLES_REQUIRED = strata_size; // On average to make a full sweep across the spectrum, i.e. to have all wavelengths covered.
  
  static Index3 MakeIndices(int main_idx, Sampler &sampler)
  {
    Index3 ret{main_idx, main_idx+strata_size, main_idx+2*strata_size};
    // Permutation does nothing except when doing spectral rendering. 
    // In this case I can simply use the wavelength of the first index because
    // it has equal chance of being in the R, G, or B stratum.
    // To render a prism, for instance, the first wavelength would be taken to
    // determine the index of refraction and the contribution of the other 
    // two wavelengths would be zero'd out.
    RandomShuffle(ret.data(), ret.data()+ret.size(), sampler);
    return ret;
  }
  
  static int PrimaryIndex(const Index3 &idx)
  {
    return idx[0];
  }
  
  LambdaSelection WithWeights(Sampler &sampler)
  {
    if (++current_idx >= current_selection_permutation.size())
    {
      Shuffle(sampler);
      current_idx = 0;
    }
    int lambda_idx = current_selection_permutation[current_idx];
    auto idx     = MakeIndices(lambda_idx, sampler);
    auto weights = Spectral3{strata_size};
    return LambdaSelection{idx, weights, LambdaSelectionStrategy::SampleWavelengthStrata(idx, sampler)};
  }
};


inline LambdaSelection SelectRgbPrimaryWavelengths()
{
  Index3 idx = Color::LambdaIdxClosestToRGBPrimaries();
  return {
    idx,
    Spectral3::Ones(),
    Take(Color::GetWavelengths(),idx)
  };
}



struct VolumePdfCoefficients
{
  double pdf_scatter_fwd{ 1. }; // Moving forward. Pdf for scatter event to occur at the end of the given segment.
  double pdf_scatter_bwd{ 1. }; // Backward. For scatter event at the segment start, moving from back to start.
  double transmittance{ 1. }; // Corresponding transmittances.
};

inline std::tuple<double, double> FwdCoeffs(const VolumePdfCoefficients &c)
{
  return std::make_tuple(c.pdf_scatter_fwd, c.transmittance);
}

inline std::tuple<double, double> BwdCoeffs(const VolumePdfCoefficients &c)
{
  return std::make_tuple(c.pdf_scatter_bwd, c.transmittance);
}


inline void Accumulate(VolumePdfCoefficients &accumulated, const VolumePdfCoefficients &segment_coeff, bool is_first, bool is_last)
{
  accumulated.pdf_scatter_fwd *= is_last ? segment_coeff.pdf_scatter_fwd : segment_coeff.transmittance;
  accumulated.pdf_scatter_bwd *= is_first ? segment_coeff.pdf_scatter_bwd : segment_coeff.transmittance;
  accumulated.transmittance *= segment_coeff.transmittance;
}


class PiecewiseConstantTransmittance
{
  static constexpr int PIECEWISE_STATIC_ALLOC_SIZE = 96;
  using Boundaries = boost::container::small_vector<float, PIECEWISE_STATIC_ALLOC_SIZE>;
  using Weights = boost::container::small_vector<Spectral3, PIECEWISE_STATIC_ALLOC_SIZE>;
  Boundaries boundaries;
  Weights weights;
public:
  void PushBack(float t, const Spectral3 &weight)
  {
    assert(boundaries.empty() || t>=boundaries.back());
    boundaries.push_back(t);
    weights.push_back(weight);
  }
  
  float End() const
  {
    assert(!boundaries.empty());
    return boundaries.back();
  }
  
  void Clear()
  {
    boundaries.clear();
    weights.clear();
  }
  
  // Lookup
  Spectral3 operator()(float t) const
  {
    auto it = std::lower_bound(boundaries.begin(), boundaries.end(), t);
    if (it != boundaries.end())
    {
      assert (t <= boundaries.back());
      return weights[it-boundaries.begin()];
    }
    else // t is larger than last element in the sequence
      return Spectral3::Zero();
  }
};


namespace TrackingDetail
{

template<class WeightMultiplyFunctor>
inline bool RussianRouletteSurvival(double weight, int iteration, Sampler &sampler, WeightMultiplyFunctor multiply_weight_with)
{
  assert (weight > -0.1); // Should be non-negative, really.
  if (weight <= 0.)
    return false;
  if (iteration < 5)
    return true;
  double prob_survival = std::min(weight, 1.);
  if (sampler.Uniform01() < prob_survival)
  {
    multiply_weight_with(1./prob_survival);
    return true;
  }
  else
  {
    multiply_weight_with(0.);
    return false;
  }
}


// Is passing parameters like that as efficient has having the same number of item as normal individual arguments?
// Ref: Kutz et a. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"
inline void ComputeProbabilitiesHistoryScheme(
  const Spectral3 &weights,
  std::initializer_list<std::reference_wrapper<const Spectral3>> sigmas,
  std::initializer_list<std::reference_wrapper<double>> probs)
{
  double normalization = 0.;
  auto it_sigma = sigmas.begin();
  auto it_probs = probs.begin();
  for (; it_sigma != sigmas.end(); ++it_sigma, ++it_probs)
  {
    const Spectral3 &s = it_sigma->get();
    assert (s.minCoeff() >= 0.);
    double p = (s*weights).mean();
    it_probs->get() = p;
    normalization += p;
  }
  if (normalization > 0.)
  {
    double norm_inv = 1./normalization;
    for (it_probs = probs.begin(); it_probs != probs.end(); ++it_probs)
      it_probs->get() *= norm_inv; 
  }
  else // Zeroed weights?
  {
    const double p = 1.0/probs.size();
    for (it_probs = probs.begin(); it_probs != probs.end(); ++it_probs)
      it_probs->get() = p;
  }
}

} // TrackingDetail


inline const Material& GetMaterialOf(const SurfaceInteraction &ia, const Scene &scene)
{
  return scene.GetMaterialOf(ia.hitid);
}

//inline const Medium& GetMediumOf(const SurfaceInteraction &ia, const Scene &scene)
//{
//  return *ASSERT_NOT_NULL(GetMaterialOf(ia, scene).medium);
//}

inline const Shader& GetShaderOf(const SurfaceInteraction &ia, const Scene &scene)
{
  return *ASSERT_NOT_NULL(GetMaterialOf(ia, scene).shader);
}



