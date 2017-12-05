#include "atmosphere.hxx"
#include "ray.hxx"
#include "sampler.hxx"
#include "phasefunctions.hxx"
#include "shader_util.hxx"

namespace Atmosphere
{

Simple::Simple(const Double3& _planet_center, double _planet_radius, int _priority)
  : Medium(_priority),
    geometry{_planet_center, _planet_radius},
    constituents{}
{

}


Medium::InteractionSample Simple::SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
{
  assert(!context.beta.isZero());
  Medium::InteractionSample smpl{
    0.,
    Spectral3{1.}
  };
  // The lowest point gives the largest collision coefficients along the path.
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  Spectral3 sigma_s, sigma_a, sigma_n;
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


Spectral3 Simple::EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
{
  Spectral3 estimate{1.};
  // The lowest point gives the largest collision coefficients along the path.
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  double lowest_altitude = geometry.ComputeAltitude(lowest_point);
  // First compute the majorante
  Spectral3 sigma_s, sigma_a, sigma_n;
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


Medium::PhaseSample Simple::SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const
{
  double altitude = geometry.ComputeAltitude(pos);
  Spectral3 sigma_s[SimpleConstituents::NUM_CONSTITUENTS];
  constituents.ComputeSigmaS(altitude, sigma_s);

  return PhaseFunctions::Combined(context.beta, sigma_s[0], constituents.GetPhaseFunction(0), sigma_s[1], constituents.GetPhaseFunction(1))
    .SampleDirection(incident_dir, sampler);
}


Spectral3 Simple::EvaluatePhaseFunction(const Double3 &incident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const
{
  double altitude = geometry.ComputeAltitude(pos);
  Spectral3 sigma_s[SimpleConstituents::NUM_CONSTITUENTS];
  constituents.ComputeSigmaS(altitude, sigma_s);
  
  return PhaseFunctions::Combined(context.beta, sigma_s[0], constituents.GetPhaseFunction(0), sigma_s[1], constituents.GetPhaseFunction(1))
    .Evaluate(incident_dir, out_direction, pdf);
}



SimpleConstituents::SimpleConstituents()
  : phasefunction_hg(0.76),
    lower_altitude_cutoff(-std::numeric_limits<double>::max())
{
  inv_scale_height[MOLECULES] = 1./8.; // km
  inv_scale_height[AEROSOLES] = 1./1.2;  // km
  at_sealevel[MOLECULES].sigma_a = Spectral3{0};
  at_sealevel[MOLECULES].sigma_s = 1.e-3 * Spectral3{5.8, 13.5, 33.1};
  at_sealevel[AEROSOLES].sigma_a = 1.e-3 * Spectral3{2.22};
  at_sealevel[AEROSOLES].sigma_s = 1.e-3 * Spectral3{20.};
  for (auto inv_h : inv_scale_height)
    lower_altitude_cutoff = std::max(-1./inv_h, lower_altitude_cutoff);
}

}
