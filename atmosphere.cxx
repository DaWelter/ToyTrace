#include "atmosphere.hxx"
#include "ray.hxx"
#include "sampler.hxx"
#include "phasefunctions.hxx"

namespace Atmosphere
{

Simple::Simple(const Double3& _planet_center, double _planet_radius, int _priority)
  : Medium(_priority),
    geometry{_planet_center, _planet_radius},
    constituents{}
{

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
void ComputeProbabilitiesHistoryScheme(
  const Spectral &weights,
  std::initializer_list<std::reference_wrapper<const Spectral>> sigmas, 
  std::initializer_list<std::reference_wrapper<double>> probs)
{
  double normalization = 0.;
  auto it_sigma = sigmas.begin();
  auto it_probs = probs.begin();
  for (; it_sigma != sigmas.end(); ++it_sigma, ++it_probs)
  {
    const Spectral &s = it_sigma->get();
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


Medium::InteractionSample Simple::SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
{
  // Select a wavelength
  assert(!context.beta.isZero());
  Medium::InteractionSample smpl{
    0.,
    Spectral{1.}
  };
  // The lowest point gives the largest collision coefficients along the path.
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  Spectral sigma_s, sigma_a, sigma_n;
  double prob_t, prob_n;
  double altitude = geometry.ComputeAltitude(lowest_point);
  constituents.ComputeCollisionCoefficients(
    altitude, sigma_s, sigma_a);
  double sigma_t_majorant = (sigma_a + sigma_s).maxCoeff();
  double inv_sigma_t_majorant = 1./sigma_t_majorant;
  // Then start tracking.
  // - No russian roulette because the probability to arrive
  // beyond the end of the segment must be identical to the transmissivity.
  // - Need to limit the number of iterations to protect against evil edge
  // cases where a particles escapes outside of the scene boundaries but
  // due to incorrect collisions the path tracer thinks that we are in a medium.
  constexpr int emergency_abort_max_num_iterations = 100;
  int iteration = 0;
  while (++iteration < emergency_abort_max_num_iterations)
  {
    smpl.t -= std::log(sampler.Uniform01()) * inv_sigma_t_majorant;
    if (smpl.t > segment.length)
    {
      return smpl;
    }
    else
    {
      const Double3 pos = segment.ray.PointAt(smpl.t);
      altitude = geometry.ComputeAltitude(pos);
      constituents.ComputeCollisionCoefficients(
        altitude, sigma_s, sigma_a);
      sigma_n = sigma_t_majorant - sigma_s - sigma_a;
      assert(sigma_n.minCoeff() >= -1.e-3); // By definition of the majorante
      TrackingDetail::ComputeProbabilitiesHistoryScheme(smpl.weight, {sigma_s, sigma_n}, {prob_t, prob_n});
      double r = sampler.Uniform01();
      if (r < prob_t) // Scattering/Absorption
      {
        smpl.weight *= inv_sigma_t_majorant / prob_t * sigma_s;
        return smpl;
      }
      else // Null collision
      {
        smpl.weight *= inv_sigma_t_majorant / prob_n * sigma_n;
      }
    }
  }
  assert (false);
  return smpl;
}


Spectral Simple::EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
{
  Spectral estimate{1.};
  // The lowest point gives the largest collision coefficients along the path.
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  double lowest_altitude = geometry.ComputeAltitude(lowest_point);
  // First compute the majorante
  Spectral sigma_s, sigma_a, sigma_n;
  constituents.ComputeCollisionCoefficients(
        lowest_altitude, sigma_s, sigma_a);
  double sigma_t_majorant = (sigma_a + sigma_s).maxCoeff();
  double inv_sigma_t_majorant = 1./sigma_t_majorant;
  // Then start tracking.
  double t = 0.;
  int iteration = 0;
  bool gogogo = true;
  do
  {
    t -=std::log(sampler.Uniform01()) * inv_sigma_t_majorant;
    if (t > segment.length)
    {
      gogogo = false;
    }
    else
    {
      const Double3 pos = segment.ray.PointAt(t);
      constituents.ComputeCollisionCoefficients(
            geometry.ComputeAltitude(pos), sigma_s, sigma_a);
      sigma_n = sigma_t_majorant - sigma_s - sigma_a;
      assert(sigma_n.minCoeff() >= 0.); // By definition of the majorante
      estimate *= sigma_n * inv_sigma_t_majorant;
      
      gogogo = TrackingDetail::RussianRouletteSurvival(
        estimate.maxCoeff(), iteration++, sampler, [&estimate](double q) { estimate *= q; });
    }
  }
  while(gogogo);
  return estimate;
}


void Simple::ComputeProbabilities(const Double3 &pos, const PathContext &context, Spectral &prob_lambda, Spectral *prob_constituent_given_lambda) const
{
  constexpr int NL = static_size<Spectral>();
  constexpr int NC = SimpleConstituents::NUM_CONSTITUENTS;
  double altitude = geometry.ComputeAltitude(pos);

  double prob_lambda_normalization = 0.;

  constituents.ComputeSigmaS(altitude, prob_constituent_given_lambda);
  for (int lambda = 0; lambda<NL; ++lambda)
  {
    // For a given lambda, the normalization sum goes over the constituents.
    double normalization = 0.;
    for (int c=0; c<NC; ++c)
    {
      // The probability weight to select a constituent for sampling is
      // just the coefficient in front of the respective scattering function.
      normalization += prob_constituent_given_lambda[c][lambda];
    }
    assert(normalization > 0.);
    for (int c=0; c<NC; ++c)
    {
      prob_constituent_given_lambda[c][lambda] /= normalization;
    }
    // The weights of the current path should be reflected in the probability
    // to select some lambda. That is to prevent sampling a lambda which already
    // has a very low weight, or zero as in single wavelength sampling mode.
    // Add epsilon to protect against all zero beta.
    prob_lambda[lambda] = normalization * (context.beta[lambda] + Epsilon);
    prob_lambda_normalization += prob_lambda[lambda];
  }
  assert(prob_lambda_normalization > 0.);
  for (int lambda = 0; lambda<NL; ++lambda)
  {
    prob_lambda[lambda] /= prob_lambda_normalization;
  }
}


Medium::PhaseSample Simple::SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const
{
  constexpr int NL = static_size<Spectral>();
  constexpr int NC = SimpleConstituents::NUM_CONSTITUENTS;

  Spectral prob_lambda;
  Spectral prob_constituent_given_lambda[NC]; // aka sigma_s,i / (sigma_s,0 + sigma_s,1)
  ComputeProbabilities(pos, context, prob_lambda, prob_constituent_given_lambda);

  int lambda = TowerSampling<NL>(prob_lambda.data(), sampler.Uniform01());
  double contiguous_probs[NC] = {
    prob_constituent_given_lambda[0][lambda],
    prob_constituent_given_lambda[1][lambda]
  };
  int constituent = TowerSampling<NC>(contiguous_probs, sampler.Uniform01());
  int no_sampled_constituent = constituent==SimpleConstituents::MOLECULES ? SimpleConstituents::AEROSOLES : SimpleConstituents::MOLECULES;
  Medium::PhaseSample smpl =
      constituent==SimpleConstituents::MOLECULES ?
        constituents.phasefunction_rayleigh.SampleDirection(incident_dir, pos, sampler) :
        constituents.phasefunction_hg.SampleDirection(incident_dir, pos, sampler);
  double pf_pdf[SimpleConstituents::NUM_CONSTITUENTS];
  pf_pdf[constituent] = smpl.pdf;
  Spectral other_pf_value =
    constituent==SimpleConstituents::MOLECULES ?
      constituents.phasefunction_hg.Evaluate(incident_dir, pos, smpl.dir, &pf_pdf[SimpleConstituents::AEROSOLES]) :
      constituents.phasefunction_rayleigh.Evaluate(incident_dir, pos, smpl.dir, &pf_pdf[SimpleConstituents::MOLECULES]);
  smpl.value = prob_constituent_given_lambda[constituent]*smpl.value + prob_constituent_given_lambda[no_sampled_constituent]*other_pf_value;
  smpl.pdf = 0.;
  for (int c = 0; c<NC; ++c)
  {
    for (int lambda = 0; lambda<NL; ++lambda)
    {
      smpl.pdf += pf_pdf[c]*prob_lambda[lambda]*prob_constituent_given_lambda[c][lambda];
    }
  }
  return smpl;
}


Spectral Simple::EvaluatePhaseFunction(const Double3 &incident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const
{
  constexpr int NL = static_size<Spectral>();
  constexpr int NC = SimpleConstituents::NUM_CONSTITUENTS;

  Spectral prob_lambda;
  Spectral prob_constituent_given_lambda[NC];
  ComputeProbabilities(pos, context, prob_lambda, prob_constituent_given_lambda);

  if (pdf)
    *pdf = 0.;
  Spectral result{0.};
  double pf_pdf[NC];
  Spectral pf_val[NC];
  pf_val[SimpleConstituents::AEROSOLES] = constituents.phasefunction_hg.Evaluate(incident_dir, pos, out_direction, &pf_pdf[SimpleConstituents::AEROSOLES]);
  pf_val[SimpleConstituents::MOLECULES] = constituents.phasefunction_rayleigh.Evaluate(incident_dir, pos, out_direction, &pf_pdf[SimpleConstituents::MOLECULES]);
  for (int c = 0; c<NC; ++c)
  {
    if (pdf) for (int lambda = 0; lambda<NL; ++lambda)
    {
      *pdf += pf_pdf[c]*prob_lambda[lambda]*prob_constituent_given_lambda[c][lambda];
    }
    result += prob_constituent_given_lambda[c]*pf_val[c];
  }
  return result;
}



SimpleConstituents::SimpleConstituents()
  : phasefunction_hg(0.76),
    lower_altitude_cutoff(-std::numeric_limits<double>::max())
{
  inv_scale_height[MOLECULES] = 1./8.; // km
  inv_scale_height[AEROSOLES] = 1./1.2;  // km
  at_sealevel[MOLECULES].sigma_a = Spectral{0};
  at_sealevel[MOLECULES].sigma_s = 1.e-3 * Spectral{5.8, 13.5, 33.1};
  at_sealevel[AEROSOLES].sigma_a = 1.e-3 * Spectral{2.22};
  at_sealevel[AEROSOLES].sigma_s = 1.e-3 * Spectral{20.};
  for (auto inv_h : inv_scale_height)
    lower_altitude_cutoff = std::max(-1./inv_h, lower_altitude_cutoff);
}

}
