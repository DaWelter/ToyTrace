#pragma once

#include "vec3f.hxx"
#include "spectral.hxx"
#include "texture.hxx"
#include "sampler.hxx"
#include "ray.hxx"
#include "scene.hxx"

enum TransportType : char 
{
  RADIANCE,
  IMPORTANCE
};

struct PathContext
{
  PathContext() = default;
  explicit PathContext(const Index3 &_lambda_idx, TransportType _transport = RADIANCE )
    : lambda_idx(_lambda_idx), transport(_transport)
  {}
  Index3 lambda_idx {};
  TransportType transport { RADIANCE };
  int pixel_x = {-1};
  int pixel_y = {-1};
};


struct LambdaSelectionFactory
{
  /* Stratified mode: Divide spectrum in N sections, where N is the number of simultaneously traced wavelengths.
   * Pick one wavelength from each section. In my case I pick the wavelength from the first section at random,
   * and take the others at a fixed offset equal to the stratum size. */
  static constexpr int strata_size = Color::NBINS / Spectral3::RowsAtCompileTime;
  static_assert(Color::NBINS == strata_size * Spectral3::RowsAtCompileTime, "Bin count must be multiple of number of simultaneously traced wavelengths");
  
  SpectralN lambda_weights;
  LambdaSelectionFactory()
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
  
  std::pair<Index3, Spectral3> WithWeights(Sampler &sampler) const
  {
    int main_idx = sampler.UniformInt(0, strata_size-1);
    auto idx     = MakeIndices(main_idx);
    auto weights = Take(lambda_weights, idx);
    return std::make_pair(idx, weights);
  }
};



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
  double norm_inv = 1./normalization;
  it_probs = probs.begin();
  for (; it_probs != probs.end(); ++it_probs)
    it_probs->get() *= norm_inv;
}
}


inline const Material& GetMaterialOf(const SurfaceInteraction &ia, const Scene &scene)
{
  return scene.GetMaterialOf(ia.hitid);
}

inline const Shader& GetShaderOf(const SurfaceInteraction &ia, const Scene &scene)
{
  return *ASSERT_NOT_NULL(GetMaterialOf(ia, scene).shader);
}

inline const Medium& GetMediumOf(const SurfaceInteraction &ia, const Scene &scene)
{
  return *ASSERT_NOT_NULL(GetMaterialOf(ia, scene).medium);
}

inline const RadianceOrImportance::AreaEmitter& GetEmitterOf(const SurfaceInteraction &ia, const Scene &scene)
{
  return *ASSERT_NOT_NULL(GetMaterialOf(ia, scene).emitter);
}