#pragma once

#include "vec3f.hxx"
#include "util.hxx"

/* ------------  Reflection & Refraction ---------- */
/* ------------------------------------------------ */

template<class Derived>
inline auto Reflected(const Eigen::MatrixBase<Derived>& reverse_incident_dir, const Eigen::MatrixBase<Derived>& normal)
{
  return (2.*reverse_incident_dir.dot(normal)*normal - reverse_incident_dir).eval();
}

// Return n if the vector component of dir in the direction of n is positive, else -n.
template<class Derived>
inline auto AlignedNormal(const Eigen::MatrixBase<Derived>&n, const Eigen::MatrixBase<Derived>&dir)
{
  using Scalar = typename Derived::Scalar;
  return Dot(n, dir)>0 ? Scalar{1}*n : Scalar{-1}*n;
}


// Adapted from pbrt. eta is the ratio of refractive indices eta_i / eta_t
inline boost::optional<Double3> Refracted(const Double3 &wi, const Double3 &m, double eta_i_over_t) 
{
    const double eta = eta_i_over_t;
    const double c = Dot(m, wi);
    const double t1 = 1. - c*c;
    const double t2 = 1.0 - eta * eta * t1;

    // Handle total internal reflection for transmission
    if (t2 < 0.) 
      return boost::none;
    const double s = Dot(wi, m)>=0. ? 1.0 : -1.0;
    const double t3 = std::sqrt(t2);
    return Double3{(eta*c - s*t3)*m - eta*wi};
}


inline Double3 HalfVector(const Double3 &wi, const Double3 &wo)
{
  Double3 ret = wo + wi;
  double norm = Length(ret);
  if (norm > 0)
    return ret/norm;
  else // Added by me.
    // Pick a direction perpendicular to wi and wo.
    return OrthogonalSystemZAligned(wi).col(0);
}


inline Double3 HalfVectorRefracted(const Double3 &wi, const Double3 &wo, double eta_i_over_t)
{
  // From Walter et al. (2007) "Microfacet Models for Refraction through Rough Surfaces" Eq. 16
  Double3 ret = -(eta_i_over_t*wi + wo);
  double norm = Length(ret);
  if (norm > 0)
    return ret/norm;
  else // Added by me.
    // Pick a direction perpendicular to wi and wo. 
    return OrthogonalSystemZAligned(wi).col(0);
}


inline double HalfVectorPdfToReflectedPdf(double pdf_wh, double wh_dot_in)
{
    //assert(wh_dot_in >= 0); 
    // Because half-vector for reflection. Forming the half-vector from wi+wo, the condition wh_dot_in>=0 should always be true.
    double out_direction_pdf = pdf_wh*0.25/(std::abs(wh_dot_in)+Epsilon); // From density of h_r to density of out direction.
    return out_direction_pdf;
}


inline double HalfVectorPdfToTransmittedPdf(double pdf_wh, double eta_i_over_t, double dot_wi_wh, double dot_wo_wh)
{
  double denom = eta_i_over_t*dot_wi_wh + dot_wo_wh;
  denom *= denom;
  return pdf_wh*std::abs(dot_wo_wh)/denom;
}



/* --- Fresnel terms ------- */
/* ------------------------- */

template<class T>
inline T SchlicksApproximation(const T &kspecular, double n_dot_dir)
{
  // Ref: Siggraph 2012 Course. "Background: Physics and Math of Shading (Naty Hoffman)"
  //      http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
  return kspecular + (1.-kspecular)*std::pow(1-n_dot_dir, 5.);
}


template<class T>
inline T AverageOfProjectedSchlicksApproximationOverHemisphere(const T &kspecular)
{
  // Computes I= 1/Pi * Int_HalfSphere F_schlick(w)*cos(theta) dw
  // The average albedo of ideal specular reflection.
  return kspecular + (1.-kspecular)*2./42.;
  // The 42 here is no joke. It comes out of Wolfram Alpha when ordered to compute:
  // integrate (1-cos(x))^5*cos(x)*sin(x) from 0 to pi/2
  // The factor two comes from the integration over the azimuthal angle.
}


inline double FresnelReflectivity(
  double cs_i,  // cos theta of incident direction
  double cs_t,  // cos theta of refracted(!) direction. Must be >0.
  double eta_i_over_t
)
{
  // https://en.wikipedia.org/wiki/Fresnel_equations
  // I divide both nominator and denominator by eta_2
  assert (cs_i >= 0.); 
  assert (cs_t >= 0.); // Take std::abs(Dot(n,r), or flip normal.
  assert (eta_i_over_t > 0.);
  double rs_nom = eta_i_over_t * cs_i - cs_t;
  double rs_den = eta_i_over_t * cs_i + cs_t;
  double rp_nom = eta_i_over_t * cs_t - cs_i;
  double rp_den = eta_i_over_t * cs_t + cs_i;
  return 0.5*(Sqr(rs_nom/rs_den) + Sqr(rp_nom/rp_den));
}


inline double FresnelReflectivity(double cos_n_wi, double eta_i_over_t)
{
  // From Walter et al. (2007). Equivalent to the other formula but does not require the refracted direction.
  double c = std::abs(cos_n_wi);
  double tmp = Sqr(1.0/eta_i_over_t) - 1.0 + c*c;
  if (tmp < 0)
      return 1.; // Total reflection
  double g = std::sqrt(tmp);
  double nom = c*(g+c)-1.;
  double denom = c*(g-c)+1.;
  return 0.5*Sqr((g-c)/(g+c))*(1. + Sqr(nom/denom));
}
