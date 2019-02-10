#include <fstream>

#ifdef HAVE_JSON
#include <rapidjson/document.h>
#endif

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
    phasefunction_hg(0.7) // or 0.76? Peter Kutz used 0.7
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


// Spectral tracking scheme based on Kutz et al. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes".
template<class ConstituentDistribution_, class Geometry_>
Medium::InteractionSample AtmosphereTemplate<ConstituentDistribution_, Geometry_>::SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
{
  Medium::InteractionSample smpl{
    0.,
    Spectral3{1.},
    Spectral3::Zero()
  };
  Spectral3 sigma_s, sigma_a;
  double prob_t, prob_n;
  // The lowest point gives the largest collision coefficients along the path.
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  double altitude = geometry.ComputeAltitude(lowest_point);
  if (altitude < GetLowerAltitudeCutoff())
  {
    // It can totallly happen that the origin here lies below the surface.
    // It happens when a scattering event is located below the surface due
    // to roundoff errors. The intersection point of a nearby surface will
    // also be computed to lie below the surface.
    smpl.t = 0;
    smpl.weight = Spectral3{0.};
    return smpl;
  }
    
  double sigma_t_majorant = constituents.ComputeSigmaTMajorante(altitude, context.lambda_idx);
  if (sigma_t_majorant <= 0.)
  {
    smpl.t = LargeNumber;
    return smpl;
  }
  double inv_sigma_t_majorant = 1./sigma_t_majorant;
  // Then start tracking.
  // - No russian roulette because the probability to arrive
  // beyond the end of the segment must be identical to the transmissivity.
  // - Need to limit the number of iterations to protect against evil edge
  // cases where a particles escapes outside of the scene boundaries but
  // due to incorrect collisions the path tracer thinks that we are in a medium.
  constexpr int emergency_abort_max_num_iterations = 1000;
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
      Spectral3 sigma_n = sigma_t_majorant - sigma_s - sigma_a;
      assert(sigma_n.minCoeff() >= -1.e-3); // By definition of the majorante
      TrackingDetail::ComputeProbabilitiesHistoryScheme(smpl.weight, {sigma_s, sigma_n}, {prob_t, prob_n});
      double r = sampler.Uniform01();
      if (r < prob_t) // Scattering/Absorption
      {
        smpl.weight *= inv_sigma_t_majorant / prob_t;
        smpl.sigma_s = sigma_s;
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


// Ratio tracking based on Kutz et al. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes". Appendix.
template<class ConstituentDistribution_, class Geometry_>
Spectral3 AtmosphereTemplate<ConstituentDistribution_, Geometry_>::EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const
{
  Spectral3 estimate{1.};
  Spectral3 sigma_s, sigma_a;
  // The lowest point gives the largest collision coefficients along the path.
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  double lowest_altitude = geometry.ComputeAltitude(lowest_point);
  if (lowest_altitude < GetLowerAltitudeCutoff())
  {
    estimate = Spectral3{0.};
    return estimate;
  }
  
  double sigma_t_majorant = constituents.ComputeSigmaTMajorante(lowest_altitude, context.lambda_idx);
  if (sigma_t_majorant <= 0.)
    return estimate;
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
      Spectral3 sigma_n = sigma_t_majorant - sigma_s - sigma_a;
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
VolumePdfCoefficients AtmosphereTemplate<ConstituentDistribution_, Geometry_>::ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const
{
  auto lowest_point = geometry.ComputeLowestPointAlong(segment);
  double lowest_altitude = geometry.ComputeAltitude(lowest_point);
  double sigma_t_majorant = constituents.ComputeSigmaTMajorante(lowest_altitude, context.lambda_idx);
  // Pretend we are in a homogeneous medium with extinction coeff sigma_t_majorant. This is a very bad approximation.
  // But maybe it will do for MIS weighting. I got to try before implementing more elaborate approximations.
  double tr = std::exp(-sigma_t_majorant*segment.length);
  return VolumePdfCoefficients{
    sigma_t_majorant*tr,
    sigma_t_majorant*tr,
    tr,
  };
}



template<class ConstituentDistribution_, class Geometry_>
Medium::PhaseSample AtmosphereTemplate<ConstituentDistribution_, Geometry_>::SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const
{
  double altitude = geometry.ComputeAltitude(pos);
  Spectral3 sigma_s[NUM_CONSTITUENTS];
  constituents.ComputeSigmaS(altitude, sigma_s, context.lambda_idx);

  return PhaseFunctions::SimpleCombined(sigma_s[0], GetPhaseFunction(0), sigma_s[1], GetPhaseFunction(1))
    .SampleDirection(incident_dir, sampler);
}


template<class ConstituentDistribution_, class Geometry_>
Spectral3 AtmosphereTemplate<ConstituentDistribution_, Geometry_>::EvaluatePhaseFunction(const Double3 &incident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const
{
  double altitude = geometry.ComputeAltitude(pos);
  Spectral3 sigma_s[NUM_CONSTITUENTS];
  constituents.ComputeSigmaS(altitude, sigma_s, context.lambda_idx);
  
  return PhaseFunctions::SimpleCombined(sigma_s[0], GetPhaseFunction(0), sigma_s[1], GetPhaseFunction(1))
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


double ExponentialConstituentDistribution::ComputeSigmaTMajorante(double altitude, const Index3 &lambda_idx) const
{
  Spectral3 sigma_s, sigma_a;
  ComputeCollisionCoefficients(altitude, sigma_s, sigma_a, lambda_idx);
  return (sigma_s + sigma_a).maxCoeff();
}

#ifdef HAVE_JSON
std::vector<Color::SpectralN> ReadArrayOfSpectra(const rapidjson::Value &array)
{
  std::vector<Color::SpectralN> ret;
  //for (auto &json_spectrum : array)
  for (int i=0; i<array.Size(); ++i)
  {
    const auto &json_spectrum = array[i];
    if (json_spectrum.Size() != Color::NBINS)
      throw std::invalid_argument("The number of wavelengths in the array does not match the requirement.");
    Color::SpectralN s;
    for (int i=0; i<Color::NBINS; ++i)
    {
      s[i] = json_spectrum[i].GetDouble();
    }
    ret.push_back(s);
  }
  return ret;
}
#endif

TabulatedConstituents::TabulatedConstituents(const std::string& filename)
{
#ifdef HAVE_JSON
  rapidjson::Document d;
  std::ifstream is(filename.c_str(), std::ios::binary | std::ios::ate);
  std::string data;
  { // Following http://en.cppreference.com/w/cpp/io/basic_istream/read
    auto size = is.tellg();
    data.resize(size);
    is.seekg(0);
    is.read(&data[0], size);
  }
  d.Parse(data.c_str());
   
  sigma_t = ReadArrayOfSpectra(d["sigma_t"]);
  sigma_s[MOLECULES] = ReadArrayOfSpectra(d["sigma_s_molecules"]);
  sigma_s[AEROSOLES] = ReadArrayOfSpectra(d["sigma_s_aerosoles"]);
   
  if ((sigma_t.size() != sigma_s[MOLECULES].size()) || (sigma_t.size() != sigma_s[AEROSOLES].size()))
    throw std::invalid_argument("The number of altitude support points is inconsistent among arrays.");
  
  for (auto &s : sigma_t)
  {
    MajoranteType m;
    for (int lambda = 0; lambda < LAMBDA_STRATA_SIZE; ++lambda)
    {
      auto lambda_idx = LambdaSelectionStrategy::MakeIndices(lambda);
      m[lambda] = Take(s, lambda_idx).maxCoeff();
    }
    sigma_t_majorante.push_back(m);
  }
  
  /* It could happen that going from high to low altitudes, the majorante takes a dip. Perhaps
   * due to a dense aerosole layer. Unfortunately, the largest majorante is expected at the lowest point
   * of a ray traversing the atmosphere.
   * In order to ensure that this is true, the following filtering fills the dips with the upper bound of the values 
   * encountered so far going from high to low alt. */
  for (int i=sigma_t_majorante.size()-2; i>=0; --i)
  {
    for (int k=0; k<LAMBDA_STRATA_SIZE; ++k)
      sigma_t_majorante[i][k] = std::max(sigma_t_majorante[i][k], sigma_t_majorante[i+1][k]);
    // HELP: The line below crashes with seg fault! It should be equivalent to the operation in the two lines above! Lib-Eigen shenanigans?
    // When run with valgrind, there is no crash either way. No memory error detected!
    //sigma_t_majorante[i] = sigma_t_majorante[i].max(sigma_t_majorante[i+1]).eval();
  }
  
  if ((sigma_t_majorante[0] < sigma_t_majorante[sigma_t_majorante.size()-1]).any())
    throw std::runtime_error("The extinction majorante at the lowest altitude is lower than at the highest altitude. There is very probably something wrong. Not going to work with that data.");
  
  lower_altitude_cutoff = d["H0"].GetDouble();
  delta_h = d["deltaH"].GetDouble();
  upper_altitude_cutoff = lower_altitude_cutoff + delta_h * (sigma_t_majorante.size()-1);
  inv_delta_h = 1./delta_h;
#else
  throw std::runtime_error("Reading of TabulatedConstituents not implement.");
#endif
}


inline Spectral3 Lerp(const SpectralN &y0, const SpectralN &y1, double f, const Index3& lambda_idx)
{
  auto x0 = Take(y0, lambda_idx);
  auto x1 = Take(y1, lambda_idx);
  return (1.-f)*x0 + f*x1;
}

inline double Lerp(double x0, double x1, double f)
{
  return (1.-f)*x0 + f*x1;
}


void TabulatedConstituents::ComputeCollisionCoefficients(double altitude, Spectral3& sigma_s, Spectral3& sigma_a, const Index3& lambda_idx) const
{
  double real_index = RealTableIndex(altitude);
  if (real_index >= static_cast<double>(AltitudeTableSize()-1)) // Compare in double because overflow
  {
    sigma_a = sigma_s = Spectral3::Zero();
    return;
  }
  int idx = real_index; // Cutting the fractional part amounts to going to the grid site which is lower in altitude.
  if (idx >= 0)
  {
    double f = real_index - idx; // The fractional part.
    sigma_s = Lerp(this->sigma_s[AEROSOLES][idx], this->sigma_s[AEROSOLES][idx+1], f, lambda_idx) + 
              Lerp(this->sigma_s[MOLECULES][idx], this->sigma_s[MOLECULES][idx+1], f, lambda_idx);
    sigma_a = Lerp(this->sigma_t[idx], this->sigma_t[idx+1], f, lambda_idx);
    sigma_a -= sigma_s;
  }
  else
  {
    sigma_s = Take(this->sigma_s[AEROSOLES][0], lambda_idx) +
              Take(this->sigma_s[MOLECULES][0], lambda_idx);
    sigma_a = Take(this->sigma_t[0], lambda_idx);
    sigma_a -= sigma_s;
  }
}


void TabulatedConstituents::ComputeSigmaS(double altitude, Spectral3* sigma_s_of_constituent, const Index3& lambda_idx) const
{
  double real_index = RealTableIndex(altitude);
  int idx = real_index;
  if (idx >= 0 && idx < AltitudeTableSize()-1)
  {
    double f = real_index - idx; // The fractional part.
    for (int constitutent = 0; constitutent<NUM_CONSTITUENTS; ++constitutent)
      sigma_s_of_constituent[constitutent] = Lerp(this->sigma_s[constitutent][idx], this->sigma_s[constitutent][idx+1], f, lambda_idx);
  }
  else if (idx < 0)
  {
    for (int constitutent = 0; constitutent<NUM_CONSTITUENTS; ++constitutent)
      sigma_s_of_constituent[constitutent] = Take(this->sigma_s[constitutent][0], lambda_idx);
  }
  else
  {
    for (int constitutent = 0; constitutent<NUM_CONSTITUENTS; ++constitutent)
      sigma_s_of_constituent[constitutent] = Spectral3::Zero();
  }  
}


double TabulatedConstituents::ComputeSigmaTMajorante(double altitude, const Index3& lambda_idx) const
{
  int lambda_idx_primary = LambdaSelectionStrategy::PrimaryIndex(lambda_idx);
  double real_index = RealTableIndex(altitude);
  int idx = real_index;
  if (idx >= 0 && idx < AltitudeTableSize()-1)
  {
    double f = real_index - idx; // The fractional part.
    return Lerp(sigma_t_majorante[idx][lambda_idx_primary], 
                sigma_t_majorante[idx+1][lambda_idx_primary], f);
  }
  else if (idx < 0)
  {
    return sigma_t_majorante[0][lambda_idx_primary];
  }
  else
  {
    return 0.;
  }  
}



std::unique_ptr<Simple> MakeSimple(const Double3 &planet_center, double radius, int _priority)
{
  return std::make_unique<Simple>(SphereGeometry{planet_center, radius}, ExponentialConstituentDistribution{}, _priority);
}


std::unique_ptr<Tabulated> MakeTabulated(const Double3 &planet_center, double radius, const std::string &datafile, int _priority)
{
  return std::make_unique<Tabulated>(SphereGeometry{planet_center, radius}, TabulatedConstituents{datafile}, _priority);
}


}
