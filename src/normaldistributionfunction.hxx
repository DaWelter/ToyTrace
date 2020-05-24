#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "shader_physics.hxx"

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


inline double G2VCavityTransmissive(double wh_dot_in, double wh_dot_out, double ns_dot_in, double ns_dot_out, double ns_dot_wh)
{
  //double g1_i = std::min(1., 2.*std::abs(ns_dot_wh*ns_dot_in)/(std::abs(wh_dot_in)+Epsilon));
  //double g1_o = std::min(1., 2.*std::abs(ns_dot_wh*ns_dot_out)/(std::abs(wh_dot_out)+Epsilon));
  double g1_i = G1VCavity(wh_dot_in, ns_dot_in, ns_dot_wh);
  double g1_o = G1VCavity(wh_dot_out, ns_dot_out, ns_dot_wh);
  return std::max(g1_i + g1_o - 1., 0.);
}


struct VisibleNdfVCavity
{
  inline static constexpr double WI_BELOW_HORIZON_FIX_FACTOR = 1.e-6;
  // Transform wh so that it samples the VNDF distribution for the VCavity shadowing function.
  // wh : Half-vector sampled from NDF*|wh.n|
  // wi : Viewing direction
  // wh and wi must be defined in a local frame where the normal n is aligned with the z-axis!
  static void Sample(Double3 &wh, const Double3 &wi, double r)
  {
    Double3 wh_prime{-wh[0], -wh[1], wh[2]};
    double prob = ClampDot(wi, wh_prime)/(ClampDot(wi, wh) + ClampDot(wi, wh_prime) + WI_BELOW_HORIZON_FIX_FACTOR);
    if (r < prob)
    {
      wh = wh_prime;
    }
  }

  // Get the PDF corresponding to the sample function.
  // wh : Half-vector
  // wi : Viewing direction
  // wh and wi must be defined in a local frame where the normal n is aligned with the z-axis!
  static double Pdf(double ndf_val, const Double3 &wh, const Double3 &wi)
  {
    const Double3 wh_prime{-wh[0],-wh[1],wh[2]};
    const double wh_prime_dot_in = ClampDot(wi, wh_prime);
    const double wh_dot_in = ClampDot(wh, wi);
    const double prob = (2.*wh_dot_in + WI_BELOW_HORIZON_FIX_FACTOR)/(wh_dot_in + wh_prime_dot_in + WI_BELOW_HORIZON_FIX_FACTOR);
    return ndf_val*std::abs(wh[2])*prob;
    
  }
};



// f(w) = f_t(w) + f_r(w)  = f_t(ht(w))  + f_r(hr(w))
//
// Change of integration variables ->
// 
// I[xi==t] f_t(h) Li(wt(h))   + I[xi==t] f_r(h) Li(wr(h))      xi ~ p(xi | h) = { p_t = 1.-fr,  p_r = fr }
// -------------------------        ---------------------
//      p_t   p(h)                     p_r   p(h)


// p(w) for MIS ???
//
// p(w) = p(hr(w)) dhr/dw p(xi=r | hr(w))    +      p(ht(w)) dht/dw * p(xi=t | ht(w)))
// 
// Int p(w) dw = 1 ?? 
// Change of variables
// Int p(w) dw = Int dh p(h) p(xi=r | h) + p(h) p(xi=t | h) dh = 1!??


struct TransmissiveMicrofacetDensity
{
  const Double3 wi;
  const double eta_i_over_t;
  using NDF = BeckmanDistribution;
  const NDF &ndf;
  
  double Pdf(const Double3 &wo) const
  {
    double pdf_wot = 0.;
    boost::optional<Double3> wht = HalfVectorRefracted(wi, wo, eta_i_over_t);
    if (wht)
    {
      const double fr_wht = FresnelReflectivity(AbsDot(*wht,wi), eta_i_over_t);
      const double ndf_transm = ndf.EvalByHalfVector(std::abs((*wht)[2]))*std::abs((*wht)[2]);
      pdf_wot = HalfVectorPdfToTransmittedPdf(ndf_transm, eta_i_over_t, Dot(*wht, wi), Dot(*wht, wo));
      pdf_wot *= 1.0-fr_wht;
      assert(std::isfinite(pdf_wot));
    }
    
    const Double3 whr = HalfVector(wi, wo);
    const double fr_whr = FresnelReflectivity(AbsDot(whr,wi), eta_i_over_t);  
    const double ndf_reflect = ndf.EvalByHalfVector(std::abs(whr[2]))*std::abs(whr[2]);
    double pdf_wor = HalfVectorPdfToReflectedPdf(ndf_reflect, Dot(whr, wi));
    pdf_wor *= fr_whr;
    
    assert(std::isfinite(pdf_wor));
    return pdf_wor + pdf_wot;
  }
  
  Double3 Sample(const Double2 &r1, double r2)
  {
    Double3 wh = ndf.SampleHalfVector(r1);

    double fr = FresnelReflectivity(AbsDot(wh,wi), eta_i_over_t);
    boost::optional<Double3> wt = Refracted(wi, wh, eta_i_over_t);
    if (!wt)
      fr = 1.0;
    if (r2 < fr)
    {
      return Reflected(wi, wh);
    }
    else
      return *wt;
  }
};


