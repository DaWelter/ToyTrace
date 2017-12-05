#include "phasefunctions.hxx"
#include "sampler.hxx"

namespace PhaseFunctions 
{

namespace
{
// Incidient direction is assumed +z. 
Double3 mu_to_direction(double mu, double r2)
{
  const double sn = std::sin(Pi*2.* r2);
  const double cs = std::cos(Pi*2.* r2);
  const double z = mu;
  const double rho = std::sqrt(1. - z*z);
  return Double3{sn*rho, cs*rho, z};
}
}



Sample Uniform::SampleDirection(const Double3& reverse_incident_dir, Sampler& sampler) const
{
  return Sample{
    SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare()),
    Spectral3{1./UnitSphereSurfaceArea},
    1./UnitSphereSurfaceArea
  };
}

Spectral3 Uniform::Evaluate(const Double3& reverse_indcident_dir, const Double3& out_direction, double* pdf) const
{
  if (pdf)
    *pdf = 1./UnitSphereSurfaceArea;
  return Spectral3{1./UnitSphereSurfaceArea};
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


Sample Rayleigh::SampleDirection(const Double3& reverse_incident_dir, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(-reverse_incident_dir);
  Double2 r = sampler.UniformUnitSquare();
  auto mu = RayleighDetail::sample_mu(r[0]);
  auto dir = mu_to_direction(mu, r[1]);
  auto value = RayleighDetail::value(mu);
  return Sample{
    m * dir,
    Spectral3{value},
    value // is also a valid probability density since energy conservation demands normalization to one over the unit sphere.
  };
}


Spectral3 Rayleigh::Evaluate(const Double3& reverse_incident_dir, const Double3& out_direction, double* pdf) const
{
  auto val = RayleighDetail::value(Dot(-reverse_incident_dir, out_direction));
  if (pdf)
    *pdf = val;
  return Spectral3{val};
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


Sample HenleyGreenstein::SampleDirection(const Double3& reverse_incident_dir, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(-reverse_incident_dir);
  Double2 r = sampler.UniformUnitSquare();
  auto mu = HenleyGreensteinDetail::sample_mu(r[0], g);
  auto dir = mu_to_direction(mu, r[1]);
  auto value = HenleyGreensteinDetail::value(mu, g);
  return Sample{
    m * dir,
    Spectral3{value},
    value
  };
}


Spectral3 HenleyGreenstein::Evaluate(const Double3& reverse_incident_dir, const Double3& out_direction, double* pdf) const
{
  auto val = HenleyGreensteinDetail::value(Dot(-reverse_incident_dir, out_direction), g);
  if (pdf)
    *pdf = val;
  return Spectral3{val};
}


PhaseFunctions::Sample Combined::SampleDirection(const Double3& reverse_incident_dir, Sampler& sampler) const
{
  constexpr int NL = static_size<Spectral3>();
  constexpr int NC = NUM_CONSTITUENTS;
  static_assert(NC == 2, "Must be 2 constituents");

  int lambda = TowerSampling<NL, Color::Scalar>(prob_lambda.data(), sampler.Uniform01());
  Color::Scalar contiguous_probs[NC] = {
    prob_constituent_given_lambda[0][lambda],
    prob_constituent_given_lambda[1][lambda]
  };
  Color::Scalar pf_pdf[NUM_CONSTITUENTS];
  
  int constituent = TowerSampling<NC, Color::Scalar>(contiguous_probs, sampler.Uniform01());
  int not_sampled_constituent = constituent==0 ? 1 : 0;
  
  auto smpl = pf[constituent]->SampleDirection(reverse_incident_dir, sampler);
  pf_pdf[constituent] = smpl.pdf;
  
  Spectral3 other_pf_value = pf[not_sampled_constituent]->Evaluate(reverse_incident_dir, smpl.dir, &pf_pdf[not_sampled_constituent]);
  
  smpl.value = prob_constituent_given_lambda[constituent]*smpl.value + prob_constituent_given_lambda[not_sampled_constituent]*other_pf_value;
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


Spectral3 Combined::Evaluate(const Double3& reverse_indcident_dir, const Double3& out_direction, double* pdf) const
{
  constexpr int NL = static_size<Spectral3>();
  constexpr int NC = NUM_CONSTITUENTS;
  static_assert(NC == 2, "Must be 2 constituents");

  if (pdf)
    *pdf = 0.;
  Spectral3 result{0.};
  double pf_pdf[NC];
  Spectral3 pf_val[NC];
  pf_val[0] = pf[0]->Evaluate(reverse_indcident_dir, out_direction, &pf_pdf[0]);
  pf_val[1] = pf[1]->Evaluate(reverse_indcident_dir, out_direction, &pf_pdf[1]);
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



}
