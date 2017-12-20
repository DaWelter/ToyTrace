#pragma once

#include "gtest/gtest_prod.h"

#include "shader.hxx"
#include "ray.hxx"

class SimpleAtmosphereTesting;

namespace Atmosphere
{

// Uses km as length units.
static constexpr int MOLECULES = 0;
static constexpr int AEROSOLES = 1;
static constexpr int NUM_CONSTITUENTS = 2;
  

template<class ConstituentDistribution_, class Geometry_>
class AtmosphereTemplate : public Medium
{
public:
  using ConstituentDistribution = ConstituentDistribution_;
  using Geometry = Geometry_;

private:
  Geometry geometry;
  ConstituentDistribution constituents;
  
  PhaseFunctions::HenleyGreenstein phasefunction_hg;
  PhaseFunctions::Rayleigh phasefunction_rayleigh;
  inline const PhaseFunctions::PhaseFunction& GetPhaseFunction(int idx) const;
  
public:
  AtmosphereTemplate(const Geometry &geometry_, const ConstituentDistribution &constituents_, int _priority);
  InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
};



struct ExponentialConstituentDistribution
{
  ExponentialConstituentDistribution();
  struct SealevelQuantities
  {
    SpectralN sigma_s, sigma_a;
  };
  SealevelQuantities at_sealevel[NUM_CONSTITUENTS];
  double inv_scale_height[NUM_CONSTITUENTS];
  double lower_altitude_cutoff;

  void ComputeCollisionCoefficients(double altitude, Spectral3 &sigma_s, Spectral3 &sigma_a, const Index3 &lambda_idx) const;
  void ComputeSigmaS(double altitude, Spectral3* sigma_s_of_constituent, const Index3 &lambda_idx) const;
  double ComputeSigmaTMajorante(double altitude, const Index3 &lambda_idx) const;
};



struct TabulatedConstituents
{
  static constexpr int LAMBDA_STRATA_SIZE = Color::NBINS / static_size< Spectral3 >();
  using MajoranteType = Eigen::Array<Color::Scalar, LAMBDA_STRATA_SIZE,1>;
  std::vector<MajoranteType> sigma_t_majorante;
  std::vector<Color::SpectralN> sigma_t; // Total.
  std::vector<Color::SpectralN> sigma_s[NUM_CONSTITUENTS];
  double lower_altitude_cutoff;
  double upper_altitude_cutoff;
  double delta_h, inv_delta_h;
  
  inline int AltitudeTableSize() const { return sigma_t_majorante.size(); }
  inline double RealTableIndex(double altitude) const { return (altitude - lower_altitude_cutoff)*inv_delta_h; }
  
  TabulatedConstituents(const TabulatedConstituents &other) = default;
  TabulatedConstituents(const std::string &filename); // Read from JSON.
  void ComputeCollisionCoefficients(double altitude, Spectral3 &sigma_s, Spectral3 &sigma_a, const Index3 &lambda_idx) const;
  void ComputeSigmaS(double altitude, Spectral3* sigma_s_of_constituent, const Index3 &lambda_idx) const;
  double ComputeSigmaTMajorante(double altitude, const Index3 &lambda_idx) const;
};



struct SphereGeometry
{
  Double3 planet_center;
  double planet_radius;

  inline double ComputeAltitude(const Double3 &pos) const;
  inline Double3 ComputeLowestPointAlong(const RaySegment &seg) const;
};


double SphereGeometry::ComputeAltitude(const Double3 &pos) const
{
  double r = Length(pos - planet_center);
  double h = r - planet_radius;
  return h;
}


Double3 SphereGeometry::ComputeLowestPointAlong(const RaySegment &segment) const
{
  Double3 center_to_org = segment.ray.org - planet_center;
  double t_lowest = -Dot(center_to_org, segment.ray.dir); // To planet center
  if (t_lowest > segment.length) // Looking down, intersection with ground is closer
    t_lowest = segment.length;
  else if (t_lowest < 0.) // Looking up. So the origin is the lowest.
    t_lowest = 0.;
  Double3 lowest_point = segment.ray.PointAt(t_lowest);
  return lowest_point;
}


using Simple = AtmosphereTemplate<ExponentialConstituentDistribution, SphereGeometry>;
using Tabulated = AtmosphereTemplate<TabulatedConstituents, SphereGeometry>;

std::unique_ptr<Simple> MakeSimple(const Double3 &planet_center, double radius, int _priority);
std::unique_ptr<Tabulated> MakeTabulated(const Double3 &planet_center, double radius, const std::string &datafile, int _priority);

}
