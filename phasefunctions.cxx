#include "phasefunctions.hxx"
#include "sampler.hxx"

namespace PhaseFunctions 
{

namespace
{
// Incidient direction is assumed +z. 
Double3 mu_to_direction(double mu, double r2)
{
  const double sn = std::sin(Pi * r2);
  const double cs = std::cos(Pi * r2);
  const double z = mu;
  const double rho = std::sqrt(1. - z*z);
  return Double3{sn*rho, cs*rho, z};
}
}



Sample Uniform::SampleDirection(const Double3& reverse_incident_dir, const Double3 &pos, Sampler& sampler) const
{
  return Sample{
    SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare()),
    Spectral{1./UnitSphereSurfaceArea},
    1./UnitSphereSurfaceArea
  };
}

Spectral Uniform::Evaluate(const Double3& reverse_indcident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
{
  if (pdf)
    *pdf = 1./UnitSphereSurfaceArea;
  return Spectral{1./UnitSphereSurfaceArea};
}



namespace RayleighDetail
{

/* Select the cosine of the angle between the outgoing ray and the incident direction. */
double sample_mu(double r)
{
  double a = std::pow(-2.+4.*r+std::sqrt(5.-16.*r+16.*r*r), 1./3.);
  return a - 1./a;
}
 
double value(double mu)
{
  return (3./(16.*Pi))*(1.+mu*mu);
}

}


Sample Rayleigh::SampleDirection(const Double3& reverse_incident_dir, const Double3 &pos, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(-reverse_incident_dir);
  Double2 r = sampler.UniformUnitSquare();
  auto mu = RayleighDetail::sample_mu(r[0]);
  auto dir = mu_to_direction(mu, r[1]);
  auto value = RayleighDetail::value(mu);
  return Sample{
    m * dir,
    Spectral{value},
    value // is also a valid probability density since energy conservation demands normalization to one over the unit sphere.
  };
}


Spectral Rayleigh::Evaluate(const Double3& reverse_incident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
{
  auto val = RayleighDetail::value(Dot(-reverse_incident_dir, out_direction));
  if (pdf)
    *pdf = val;
  return Spectral{val};
}




namespace HenleyGreensteinDetail
{
  
/* Select the cosine of the angle between the outgoing ray and the incident direction. */
double sample_mu(double r, double g)
{
  double s = 2.0*r - 1.0;
  double g2 = g*g;
  if (std::abs(g) > 1.0e-3)
  {
    double t1 = (1.0 - g2)/(1.0 + g*s);
    double mu = (1.0 + g2 - t1*t1) / (2.0*g);
    return mu;
  }
  else // Taylor Expansion
  {
    double s2 = s*s;
    double mu = s + 1.5 * g*(1.0 - s2) - 2.0*g2*s*(1.0 - s2);
    return mu;
  }
}

double value(double mu, double g)
{
  double g2 = g*g;
  double term = 1.0 + g2 - 2.0*g * mu;
  return 1.0/(4.*Pi)*(1.0 - g2)/(std::sqrt(term)*term);
}

}


Sample HenleyGreenstein::SampleDirection(const Double3& reverse_incident_dir, const Double3 &pos, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(-reverse_incident_dir);
  Double2 r = sampler.UniformUnitSquare();
  auto mu = HenleyGreensteinDetail::sample_mu(r[0], g);
  auto dir = mu_to_direction(mu, r[1]);
  auto value = HenleyGreensteinDetail::value(mu, g);
  return Sample{
    m * dir,
    Spectral{value},
    value
  };
}


Spectral HenleyGreenstein::Evaluate(const Double3& reverse_incident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
{
  auto val = HenleyGreensteinDetail::value(Dot(-reverse_incident_dir, out_direction), g);
  if (pdf)
    *pdf = val;
  return Spectral{val};
}

}