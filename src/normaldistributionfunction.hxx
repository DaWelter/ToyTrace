#pragma once

#include "vec3f.hxx"

/* --- Microfacet normal distributions -------*/
struct BeckmanDistribution
{
  const double alpha; // aka. roughness
  
  // This formula is using the normalized distribution D(cs) 
  // such that Int_omega D(cs) cs dw = 1, using the differential solid angle dw, 
  // integrated over the hemisphere.
  double EvalByHalfVector(double ns_dot_wh) const
  {
    if (ns_dot_wh <= 0.)
      return 0.;
    const double cs = ns_dot_wh;
    const double t1 = (cs*cs-1.)/(cs*cs*alpha*alpha);
    const double t2 = alpha*alpha*cs*cs*cs*cs*Pi;
    return std::exp(t1)/t2;
  }
  
  /* Samples the Beckman microfacet distribution D(m) times |m . n|. 
  * The surface normal n is assumed to be aligned with the z-axis.
  * Returns the half-angle vector m. 
  * Ref: Walter et al. (2007) "Microfacet Models for Refraction through Rough Surfaces" Eq. 28, 29. */
  Double3 SampleHalfVector(Double2 r) const
  {
    const double t1 = -alpha*alpha*std::log(r[0]);
    const double t = 1/(t1+1);
    const double z = std::sqrt(t);
    const double rho = std::sqrt(1-t);
    const double omega = 2*Pi*r[1];
    const double sn = std::sin(omega);
    const double cs = std::cos(omega);
    return Double3{cs*rho, sn*rho, z};
  }
};


/* ------------  Shadowing functions -----------*/
inline double G1Beckmann(double cos_v_m, double cos_v_n, double alpha)
{
  // From Walter et al. 2007 "Microfacet Models for Refraction"
  // Eq. 27 which pertains to the Beckman facet distribution.
  // "... Instead we will use the Smith shadowing-masking approximation [Smi67]."
  if (cos_v_m * cos_v_n < 0.) return 0.;
  double a = cos_v_n/(alpha * std::sqrt(1 - cos_v_n * cos_v_n));
  if (a >= 1.6)
    return 1.;
  else
    return (3.535*a + 2.181*a*a)/(1.+2.276*a+2.577*a*a);
}


inline double G1VCavity(double cos_v_m, double cos_v_n, double cos_n_m)
{
  // From Heitz et. al 2014 "Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals"
  // It is Eq (3).
  // Part of the Cook-Torrance G2(VCavity) Function.
  if (cos_v_m * cos_v_n < 0.) return 0.;
  return std::min(1., 2.*std::abs(cos_n_m*cos_v_n)/(std::abs(cos_v_m)+Epsilon));
}


// double G2VCavity(double wh_dot_in, double wh_dot_out, double ns_dot_in, double ns_dot_out, double ns_dot_wh)
// {
//     // All in one G2 Function for single sided materials. (No transmission)
//     // Cook & Torrance model. Seems to work good enough although Walter et al. has concerns about it's realism.
//     // Ref: Cook and Torrance (1982) "A reflectance model for computer graphics"
//     double t1 = 2.*std::abs(ns_dot_wh*ns_dot_out) / (std::abs(wh_dot_out) + Epsilon);
//     double t2 = 2.*std::abs(ns_dot_wh*ns_dot_in) / (std::abs(wh_dot_in) + Epsilon);
//     return std::min(1., std::min(t1, t2));
// }


inline double G2VCavity(double wh_dot_in, double wh_dot_out, double ns_dot_in, double ns_dot_out, double ns_dot_wh)
{
    double g1_i = G1VCavity(wh_dot_in, ns_dot_in, ns_dot_wh);
    double g1_o = G1VCavity(wh_dot_out, ns_dot_out, ns_dot_wh);
    return std::min(g1_i, g1_o);
}


struct VisibleNdfVCavity
{
// Transform wh so that it samples the VNDF distribution for the VCavity shadowing function.
// wh : Half-vector sampled from NDF*|wh.n|
// wi : Viewing direction
// wh and wi must be defined in a local frame where the normal n is aligned with the z-axis!
static void Sample(Double3 &wh, const Double3 &wi, double r)
{
  Double3 wh_prime{-wh[0], -wh[1], wh[2]};
  if (wi[2] < 0)
  {
    wh[2] = -wh[2];
    wh_prime[2] = -wh_prime[2];
  }  
  double prob = ClampDot(wi, wh_prime)/(ClampDot(wi, wh) + ClampDot(wi, wh_prime));
  if (wi[2] < 0)
    prob = 1.0-prob;
  if (r < prob)
  {
    wh = wh_prime;
  }
  if (wi[2] < 0)
    wh[2] = -wh[2];
}

// Get the PDF corresponding to the sample function.
// wh : Half-vector
// wi : Viewing direction
// wh and wi must be defined in a local frame where the normal n is aligned with the z-axis!
static double Pdf(double ndf_val, const Double3 &wh, const Double3 &wi)
{
  double g1 = G1VCavity(Dot(wi, wh), wi[2], wh[2]);
  return g1*ndf_val*std::abs(Dot(wi,wh))/(std::abs(wi[2]) + Epsilon);
}


};

/*------------ Utils ---------------------*/
inline double HalfVectorPdfToExitantPdf(double pdf_wh, double wh_dot_in)
{
    double out_direction_pdf = pdf_wh*0.25/(wh_dot_in+Epsilon); // From density of h_r to density of out direction.
    return out_direction_pdf;
}


inline Double3 HalfVectorToExitant(const Double3 &h_r, const Double3 &reverse_incident_dir)
{
  double hr_dot_in = Dot(reverse_incident_dir, h_r);
  // See walter 2007, eq. 38. // Should I use the "reflection" formula with the abs in it instead?
  // However, for that case I haven't found the correct pdf transform for the outgoing direction yet ...
//   if (hr_dot_in < 0.)
//   {
//     hr_dot_in = -hr_dot_in;
//     h_r = -h_r;
//   }
  Double3 out_direction = 2.*hr_dot_in*h_r - reverse_incident_dir;
//   if (hr_dot_in < 0.) // Only needed when using the "reflection" variant with std::abs in it.
//     out_direction = Normalized(out_direction);
  return out_direction;
}


