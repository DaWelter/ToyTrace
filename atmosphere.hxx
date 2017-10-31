#pragma once

#include "gtest/gtest_prod.h"

#include "shader.hxx"
#include "ray.hxx"

class SimpleAtmosphereTesting;

namespace Atmosphere
{


struct SimpleConstituents
{
  SimpleConstituents();

  // Uses km as length units.
  static constexpr int MOLECULES = 0;
  static constexpr int AEROSOLES = 1;
  struct SealevelQuantities
  {
    Spectral sigma_s, sigma_a;
  };
  static constexpr int NUM_CONSTITUENTS = 2;
  SealevelQuantities at_sealevel[NUM_CONSTITUENTS];
  double inv_scale_height[NUM_CONSTITUENTS];

  PhaseFunctions::HenleyGreenstein phasefunction_hg;
  PhaseFunctions::Rayleigh phasefunction_rayleigh;

  inline void ComputeCollisionCoefficients(double altitude, int lambda_idx, double &sigma_s, double &sigma_a) const;
};


struct SphereGeometry
{
  Double3 planet_center;
  double planet_radius;

  inline double ComputeAltitude(const Double3 &pos) const;
  inline Double3 ComputeLowestPointAlong(const RaySegment &seg) const;
};


class Simple : public Medium
{
  SimpleConstituents constituents;
  SphereGeometry geometry;

public:
  Simple(const Double3 &_planet_center, double _planet_radius, int _priority);
  InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  Spectral EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler) const override;
  Spectral EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const override;
};


inline void SimpleConstituents::ComputeCollisionCoefficients(double altitude, int lambda_idx, double &sigma_s, double &sigma_a) const
{
  altitude = (altitude>-inv_scale_height[0]) ? altitude : -inv_scale_height[0];
  sigma_a = 0.;
  sigma_s = 0.;
  for (int i=0; i<NUM_CONSTITUENTS; ++i)
  {
    double rho_relative = std::exp(-inv_scale_height[i] * altitude);
    sigma_a += at_sealevel[i].sigma_a[lambda_idx] * rho_relative;
    sigma_s += at_sealevel[i].sigma_s[lambda_idx] * rho_relative;
  }
}


inline double SphereGeometry::ComputeAltitude(const Double3 &pos) const
{
  double r = Length(pos - planet_center);
  double h = r - planet_radius;
  return h;
}


inline Double3 SphereGeometry::ComputeLowestPointAlong(const RaySegment &segment) const
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


}