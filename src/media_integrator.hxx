#pragma once

#include "spectral.hxx"
#include "sampler.hxx"
#include "shader.hxx"
#include "scene.hxx"
#include <limits>

class MediumTracker;

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
  if (sampler.GetRandGen().Uniform01() < prob_survival)
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


// Yields x, distributed according to the pdf sigma_ext*Transmittance(x)
inline double HomogeneousTransmittanceDistanceSample(double sigma_ext, double r)
{
  return -std::log(1. - r) / sigma_ext;
}

inline double HomogeneousTransmittance(double sigma_ext, double t)
{
  return std::exp(-sigma_ext * t);
}


Medium::InteractionSample HomogeneousSpectralTracking(const materials::Medium &medium, const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context);
Medium::InteractionSample SpectralTracking(const materials::Medium &medium, const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context);
void HomogeneousConstructShortBeamTransmittance(const Medium& medium, const RaySegment& segment, Sampler& sampler, const PathContext& context, PiecewiseConstantTransmittance& pct);
Spectral3 EstimateTransmission(const Medium &medium, const RaySegment &segment, Sampler &sampler, const PathContext &context);

} // TrackingDetail

// From Miller et al. (2019) "A null-scattering path integral formulation of light transport"
namespace nullpath
{

constexpr inline double SmallishDouble = std::numeric_limits<double>::min()*1024;

// Rows <-> Contribution components, 
// Cols <-> Probabilities, one per wavelength.
using Spectral33 = Eigen::Array<double, Spectral3::RowsAtCompileTime, Spectral3::RowsAtCompileTime>;

void AccumulateContributions(Spectral33 &w, const Spectral33 &x);
Spectral33 FixNan(const Spectral33 &x);

struct ThroughputAndPdfs
{
  Spectral33 weights_nulls{ Eigen::ones };
  Spectral33 weights_track{ Eigen::ones };
};

struct InteractionSample : public ThroughputAndPdfs
{
  materials::MediaCoefficients coeffs;
  double t = 0.;
};


InteractionSample Tracking(const materials::Medium &medium, const RaySegment& segment, Sampler& sampler, const PathContext &context);
ThroughputAndPdfs Transmission(const Medium &medium, const RaySegment &segment, Sampler &sampler, const PathContext &context);

std::tuple<MaybeSomeInteraction, double, ThroughputAndPdfs>
Tracking(
  const Scene &scene,
  const Ray &ray,
  Sampler &sampler,
  MediumTracker &medium_tracker,
  const PathContext &context);

ThroughputAndPdfs Transmission(
  const Scene &scene, 
  RaySegment seg, 
  Sampler &sampler,
  MediumTracker &medium_tracker, 
  const PathContext &context);

} // namespace nullpath
