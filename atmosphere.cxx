#include "atmosphere.hxx"
#include "ray.hxx"
#include "sampler.hxx"
#include "phasefunctions.hxx"
#include "shader_util.hxx"
#include "util.hxx"

namespace Atmosphere
{

template<class ConstituentDistribution_, class Geometry_>
AtmosphereTemplate<ConstituentDistribution_, Geometry_>::AtmosphereTemplate(const Geometry &geometry_, const ConstituentDistribution &constituents_, int _priority)
  : Medium(_priority),
    geometry{geometry_},
    constituents{constituents_},
    phasefunction_hg(0.76)
{

}


template<class ConstituentDistribution_, class Geometry_>
const PhaseFunctions::PhaseFunction& AtmosphereTemplate<ConstituentDistribution_, Geometry_>::GetPhaseFunction(int idx) const
{
  using PF = PhaseFunctions::PhaseFunction;
  return (idx==MOLECULES) ? 
    static_cast<const PF&>(phasefunction_rayleigh) : 
    static_cast<const PF&>(phasefunction_hg);
}


template<class ConstituentDistribution_, class Geometry_>
Medium::InteractionSample AtmosphereTemplate<ConstituentDistribution_, Geometry_>::SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
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
    altitude, sigma_s, sigma_a, context.lambda_idx);
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
        altitude, sigma_s, sigma_a, context.lambda_idx);
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


template<class ConstituentDistribution_, class Geometry_>
Spectral3 AtmosphereTemplate<ConstituentDistribution_, Geometry_>::EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
{
  Spectral3 estimate{1.};
  // The lowest point gives the largest collision coefficients along the path.
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  double lowest_altitude = geometry.ComputeAltitude(lowest_point);
  // First compute the majorante
  Spectral3 sigma_s, sigma_a, sigma_n;
  constituents.ComputeCollisionCoefficients(
        lowest_altitude, sigma_s, sigma_a, context.lambda_idx);
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
            geometry.ComputeAltitude(pos), sigma_s, sigma_a, context.lambda_idx);
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


template<class ConstituentDistribution_, class Geometry_>
Medium::PhaseSample AtmosphereTemplate<ConstituentDistribution_, Geometry_>::SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const
{
  double altitude = geometry.ComputeAltitude(pos);
  Spectral3 sigma_s[NUM_CONSTITUENTS];
  constituents.ComputeSigmaS(altitude, sigma_s, context.lambda_idx);

  return PhaseFunctions::Combined(context.beta, sigma_s[0], GetPhaseFunction(0), sigma_s[1], GetPhaseFunction(1))
    .SampleDirection(incident_dir, sampler);
}


template<class ConstituentDistribution_, class Geometry_>
Spectral3 AtmosphereTemplate<ConstituentDistribution_, Geometry_>::EvaluatePhaseFunction(const Double3 &incident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const
{
  double altitude = geometry.ComputeAltitude(pos);
  Spectral3 sigma_s[NUM_CONSTITUENTS];
  constituents.ComputeSigmaS(altitude, sigma_s, context.lambda_idx);
  
  return PhaseFunctions::Combined(context.beta, sigma_s[0], GetPhaseFunction(0), sigma_s[1], GetPhaseFunction(1))
    .Evaluate(incident_dir, out_direction, pdf);
}





ExponentialConstituentDistribution::ExponentialConstituentDistribution()
  : lower_altitude_cutoff(-std::numeric_limits<double>::max())
{
  inv_scale_height[MOLECULES] = 1./8.; // km
  inv_scale_height[AEROSOLES] = 1./1.2;  // km

  double lambda_reference = Color::GetWavelength(0);
  double Na = 6.022e23; // particles per mol
  double rho_sealevel =1.225; // kg/m3
  double m = 29e-3; // kg/mol
  double N = rho_sealevel * Na / m; // 1./m3
  double ior = 1.00028;
  double ior_term = Sqr(Sqr(ior) - 1.);
  constexpr double m3_to_nm3 = 1.e27;
  constexpr double km_to_nm = 1.e12;
  double sigma_s_ref_prefactor = 8.*Pi*Pi*Pi/3.*ior_term/N*m3_to_nm3*km_to_nm;
  
  at_sealevel[MOLECULES].sigma_a = SpectralN{0};
  for (int lambda_idx = 0; lambda_idx < Color::NBINS; ++lambda_idx)
  {
    double lambda = Color::GetWavelength(lambda_idx);
    at_sealevel[MOLECULES].sigma_s[lambda_idx] = sigma_s_ref_prefactor/Sqr(Sqr(lambda));
    //std::cout << "sigma_s[" << lambda << "] = " << at_sealevel[MOLECULES].sigma_s[lambda_idx] << std::endl;
  }
  at_sealevel[AEROSOLES].sigma_a = 1.e-3 * SpectralN{2.22};
  at_sealevel[AEROSOLES].sigma_s = 1.e-3 * SpectralN{20.};
  for (auto inv_h : inv_scale_height)
    lower_altitude_cutoff = std::max(-1./inv_h, lower_altitude_cutoff);
}


void ExponentialConstituentDistribution::ComputeCollisionCoefficients(double altitude, Spectral3& sigma_s, Spectral3& sigma_a, const Index3 &lambda_idx) const
{
  assert (altitude > lower_altitude_cutoff);
  altitude = (altitude>lower_altitude_cutoff) ? altitude : lower_altitude_cutoff;
  sigma_a = Spectral3{0.};
  sigma_s = Spectral3{0.};
  for (int i=0; i<NUM_CONSTITUENTS; ++i)
  {
    double rho_relative = std::exp(-inv_scale_height[i] * altitude);
    sigma_a += Take(at_sealevel[i].sigma_a, lambda_idx) * rho_relative;
    sigma_s += Take(at_sealevel[i].sigma_s, lambda_idx) * rho_relative;
  }
}


void ExponentialConstituentDistribution::ComputeSigmaS(double altitude, Spectral3* sigma_s_of_constituent, const Index3 &lambda_idx) const
{
  assert (altitude > lower_altitude_cutoff);
  altitude = (altitude>lower_altitude_cutoff) ? altitude : lower_altitude_cutoff;
  for (int i=0; i<NUM_CONSTITUENTS; ++i)
  {
    double rho_relative = std::exp(-inv_scale_height[i] * altitude);
    sigma_s_of_constituent[i]= Take(at_sealevel[i].sigma_s, lambda_idx) * rho_relative;
  }
}


std::unique_ptr<Simple> MakeSimple(const Double3 &planet_center, double radius, int _priority)
{
  return std::make_unique<Simple>(SphereGeometry{planet_center, radius}, ExponentialConstituentDistribution{}, _priority);
}

}
