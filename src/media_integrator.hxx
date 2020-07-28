#pragma once

#include "spectral.hxx"
#include "sampler.hxx"

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