#include "shader.hxx"
#include "ray.hxx"
#include "scene.hxx"
#include "shader_util.hxx"
#include "normaldistributionfunction.hxx"
#include "shader_physics.hxx"

namespace ShadingInternal
{

/* --- Textureing -------*/
inline Spectral3 MaybeMultiplyTextureLookup(const Spectral3 &color, const Texture *tex, const SurfaceInteraction &surface_hit, const Index3 &lambda_idx)
{
  Spectral3 ret{color};
  if (tex)
  {
    RGB col = tex->GetPixel(UvToPixel(*tex, surface_hit.tex_coord));
    ret *= Color::RGBToSpectralSelection(col, lambda_idx); // TODO: optimize, I don't have to compute the full spectrum.
  }
  return ret;
}


inline double MaybeMultiplyTextureLookup(double _value, const Texture *tex, const SurfaceInteraction &surface_hit)
{
  if (tex)
  {
    RGB col = tex->GetPixel(UvToPixel(*tex, surface_hit.tex_coord));
    _value *= (value(col[0])+value(col[1])+value(col[2]))/3.;
  }
  return _value;
}
  
}

using namespace ShadingInternal;


double Shader::Pdf(const Double3& incident_dir, const SurfaceInteraction& surface_hit, const Double3& out_direction, const PathContext& context) const
{
  SurfaceInteraction intersect{surface_hit};
  // Puke ... TODO: Abolish requirement of normal alignment with incident dir.
  if (Dot(incident_dir, intersect.normal) < 0.)
  {
    intersect.normal = -intersect.normal;
    intersect.shading_normal = -intersect.shading_normal;
  }
  // TODO: This should be implemented in each shader so that only the pdf is computed (?)!
  double pdf;
  this->EvaluateBSDF(incident_dir, intersect, out_direction, context, &pdf);
  return pdf;
}


DiffuseShader::DiffuseShader(const SpectralN &_reflectance, std::shared_ptr<Texture> _diffuse_texture)
  : Shader(),
    kr_d(_reflectance),
    diffuse_texture(std::move(_diffuse_texture))
{
  // Wtf? kr_d is the (constant) Lambertian BRDF. Energy conservation
  // demands Int|_Omega kr_d cos(theta) dw <= 1. Working out the math
  // I obtain kr_d <= 1/Pi. 
  // But well, reflectance, also named Bihemispherical reflectance
  // [TotalCompendium.pdf,pg.31] goes up to one. Therefore I divide by Pi. 
  kr_d *= 1./Pi;
}


Spectral3 DiffuseShader::EvaluateBSDF(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const
{
  Spectral3 ret{0.};
  assert (Dot(surface_hit.normal, incident_dir)>=0); // Because normal is aligned such that this conditions should be true.
  double n_dot_out = Dot(surface_hit.normal, out_direction);
  double nsh_dot_out = Dot(surface_hit.shading_normal, out_direction);
  if (n_dot_out > 0.) // In/Out on same side of geometric surface?
  {
    Spectral3 kr_d_taken = Take(kr_d, context.lambda_idx);
    ret = MaybeMultiplyTextureLookup(kr_d_taken, diffuse_texture.get(), surface_hit, context.lambda_idx);
  }
  if (pdf)
  {
    *pdf = std::max(0., nsh_dot_out)/Pi;
  }
  return ret;
}


ScatterSample DiffuseShader::SampleBSDF(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, Sampler& sampler, const PathContext &context) const
{
  auto m = OrthogonalSystemZAligned(surface_hit.shading_normal);
  Double3 v = SampleTrafo::ToCosHemisphere(sampler.UniformUnitSquare());
  double pdf = v[2]/Pi;
  Double3 out_direction = m * v;
  if (Dot(surface_hit.normal, out_direction) > 0)
  {
    Spectral3 value = Take(kr_d, context.lambda_idx);
    value = MaybeMultiplyTextureLookup(value, diffuse_texture.get(), surface_hit, context.lambda_idx);
    return ScatterSample{out_direction, value, pdf};
  }
  else
  {
    return ScatterSample{out_direction, Spectral3::Zero(), pdf};
  }
}



SpecularReflectiveShader::SpecularReflectiveShader(const SpectralN& reflectance)
  : Shader(),
    kr_s(reflectance)
{
}


Spectral3 SpecularReflectiveShader::EvaluateBSDF(const Double3& incident_dir, const SurfaceInteraction& surface_hit, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  if (pdf)
    *pdf = 0.;
  return Spectral3{0.};
}


ScatterSample SpecularReflectiveShader::SampleBSDF(const Double3& incident_dir, const SurfaceInteraction& surface_hit, Sampler& sampler, const PathContext &context) const
{
  Double3 r = Reflected(incident_dir, surface_hit.shading_normal);
  double cos_rn = Dot(surface_hit.normal, r);
  
  auto NominalSample = [&]() -> ScatterSample
  {
    double cos_rsdn = Dot(surface_hit.shading_normal, r);
    auto kr_s_taken = Take(kr_s, context.lambda_idx);
    return ScatterSample{r, kr_s_taken/cos_rsdn, 1.};
  };
  
  ScatterSample smpl = (cos_rn < 0.) ?
    ScatterSample{r, Spectral3{0.}, 1.} :
    NominalSample();
  SetPmfFlag(smpl);
  return smpl;
}


inline bool OnSameSide(const Double3 &reverse_incident_dir, const SurfaceInteraction &surface_hit, const Double3 &other_dir)
{
  assert(Dot(reverse_incident_dir, surface_hit.normal) >= 0.);
  return Dot(other_dir, surface_hit.normal) >= 0.;
}



SpecularTransmissiveDielectricShader::SpecularTransmissiveDielectricShader(double _ior_ratio, double ior_lambda_coeff_) 
  : Shader{}, ior_ratio{_ior_ratio}, ior_lambda_coeff{ior_lambda_coeff_}
{
  if (ior_lambda_coeff != 0)
    require_monochromatic = true;
}


ScatterSample SpecularTransmissiveDielectricShader::SampleBSDF(const Double3 &reverse_incident_dir, const SurfaceInteraction &surface_hit, Sampler& sampler, const PathContext &context) const
{
  double abs_shn_dot_i = std::abs(Dot(surface_hit.shading_normal, reverse_incident_dir));

  // Continue with shader sampling ...
  bool entering = Dot(surface_hit.geometry_normal, reverse_incident_dir) > 0.;
  double eta_i_over_t = [this,entering,&context]() {
    if (ior_lambda_coeff == 0)
      return entering ? 1./ior_ratio  : ior_ratio; // eta_i refers to ior on the side of the incomming random walk!
    else
    {
      double ior = ior_ratio + ior_lambda_coeff*context.wavelengths[0];
      return entering ? 1./ior  : ior;
    }
  }();
  
  /* Citing Veach (pg. 147): "Specular BSDF’s contain Dirac distribu-
  tions, which means that the only allowable operation is sampling: there must be an explicit
  procedure that generates a sample direction and a weight. When the specular BSDF is not
  symmetric, the direction and/or weight computation for the adjoint is different, and thus
  there must be two different sampling procedures, or an explicit flag that specifies whether
  the direct or adjoint BSDF is being sampled."! */

  double radiance_weight = (context.transport==RADIANCE) ? Sqr(eta_i_over_t) : 1.;
  
  double fresnel_reflectivity = 1.;
  boost::optional<Double3> wt = Refracted(reverse_incident_dir, surface_hit.shading_normal, eta_i_over_t);
  if (wt)
  {
    double abs_shn_dot_r = std::abs(Dot(*wt, surface_hit.shading_normal));
    fresnel_reflectivity = FresnelReflectivity(abs_shn_dot_i, abs_shn_dot_r, eta_i_over_t);
  }
  assert (fresnel_reflectivity >= -0.00001 && fresnel_reflectivity <= 1.000001);
  
  bool do_sample_reflection = sampler.Uniform01() < fresnel_reflectivity;
  assert (do_sample_reflection || (bool)(wt));
  
  // First determine PDF and randomwalk direction.
  ScatterSample smpl;
  if (do_sample_reflection)
  {
    smpl.coordinates = Reflected(reverse_incident_dir, surface_hit.shading_normal);
    smpl.pdf_or_pmf = Pdf::MakeFromDelta(fresnel_reflectivity);
  }
  else
  {
    smpl.coordinates = *wt;
    smpl.pdf_or_pmf = Pdf::MakeFromDelta(1.-fresnel_reflectivity);
  }
  
  // Veach style handling of shading normals. See  Veach Figure 5.8.
  // In this case, the BRDF and the BTDF are almost equal.
  smpl.value = (double)smpl.pdf_or_pmf / std::abs(Dot(smpl.coordinates, surface_hit.shading_normal));
  // Must use the fresnel_reflectivity term like in the pdf to make it cancel.
  // Then I must use the dot product with the shading normal to make it cancel with the 
  // corresponding term in the reflection integration (outside of BSDF code).
  if (Dot(smpl.coordinates, surface_hit.normal) < 0) 
  {
    // Evaluate BTDF
    smpl.value *= radiance_weight;
  }
  
  assert((smpl.value*std::abs(Dot(surface_hit.shading_normal, smpl.coordinates)) / smpl.pdf_or_pmf).maxCoeff() < 2.0);
  
  if (ior_lambda_coeff != 0)
  {
    smpl.value[1] = smpl.value[2] = 0;
  }
  return smpl;
}


Spectral3 SpecularTransmissiveDielectricShader::EvaluateBSDF(const Double3 &reverse_incident_dir, const SurfaceInteraction& surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const
{
  if (pdf)
    *pdf = 0.;
  return Spectral3{0.};
}



SpecularPureRefractiveShader::SpecularPureRefractiveShader(double _ior_ratio) : Shader{}, ior_ratio{_ior_ratio}
{

}


ScatterSample SpecularPureRefractiveShader::SampleBSDF(const Double3& reverse_incident_dir, const SurfaceInteraction& surface_hit, Sampler& sampler, const PathContext& context) const
{
  ScatterSample smpl;
  double shn_dot_i = std::abs(Dot(surface_hit.shading_normal, reverse_incident_dir));
  bool entering = Dot(surface_hit.geometry_normal, reverse_incident_dir) > 0.;
  double eta_i_over_t = entering ? 1./ior_ratio  : ior_ratio; // eta_i refers to ior on the side of the incomming random walk! 
    
  double radiance_weight = (context.transport==RADIANCE) ? Sqr(eta_i_over_t) : 1.;
  
  boost::optional<Double3> wt = Refracted(reverse_incident_dir, surface_hit.shading_normal, eta_i_over_t);
  
  if (!wt) // Total reflection. Neglected!
  {
    smpl.value = {0.};
    smpl.pdf_or_pmf = Pdf::MakeFromDelta(1.);
    smpl.coordinates = reverse_incident_dir;
    return smpl;
  }
  smpl.coordinates = *wt;
  smpl.pdf_or_pmf = Pdf::MakeFromDelta(1.);
  if (OnSameSide(reverse_incident_dir, surface_hit, *wt))
    smpl.value = Spectral3{0.}; // Should be on other side of geometric surface, but we are not!
  else
  {
    smpl.value = Spectral3{-1./Dot(*wt, surface_hit.shading_normal)*radiance_weight};
  }
  return smpl;
}


Spectral3 SpecularPureRefractiveShader::EvaluateBSDF(const Double3& reverse_incident_dir, const SurfaceInteraction& surface_hit, const Double3& out_direction, const PathContext& context, double* pdf) const
{
  if (pdf)
    *pdf = 0.;
  return Spectral3{0.};
}




namespace {
struct LocalFrame // Helper
{
  Eigen::Matrix3d m_local;
  Eigen::Matrix3d m_local_inv;
  Double3 ng; // In local frame

  LocalFrame(const SurfaceInteraction &surface_hit);
};


LocalFrame::LocalFrame(const SurfaceInteraction& surface_hit)
{
  m_local = OrthogonalSystemZAligned(surface_hit.shading_normal);
  m_local_inv = m_local.transpose();
  ng = m_local_inv * surface_hit.normal;
}


#define SAMPLE_VNDF
//--gtest_break_on_failure

struct MicrofacetShaderWrapper
{
  const PathContext &context;
  const BeckmanDistribution &ndf;
  const LocalFrame &frame;
  const Spectral3 &color;
  
  Spectral3 Evaluate(const Double3 &wi, const Double3 &wh, const Double3& wo, double *pdf) const
  {
    const double n_dot_out = Dot(frame.ng, wo);
    const double ns_dot_out = wo[2];
    const double ns_dot_in  = wi[2];

    const double ns_dot_wh = wh[2];
    const double wh_dot_out = Dot(wo, wh);
    const double wh_dot_in  = Dot(wi, wh);
    
    double microfacet_distribution_val = ndf.EvalByHalfVector(std::abs(ns_dot_wh));
    const double geometry_term = G2VCavity(wh_dot_in, wh_dot_out, ns_dot_in, ns_dot_out, ns_dot_wh);
    
    if (pdf)
    {    
  #ifdef SAMPLE_VNDF
      Double3 wh_flip = wh[2]<0 ? -wh : wh;
      double sample_pdf = VisibleNdfVCavity::Pdf(microfacet_distribution_val, wh_flip, wi);
  #else
      double sample_pdf = microfacet_distribution_val * std::abs(ns_dot_wh);
  #endif
      sample_pdf = HalfVectorPdfToReflectedPdf(sample_pdf, Dot(wh_flip, wi));
      *pdf = sample_pdf;
    }
  
    if (n_dot_out <= 0.) // Not on same side of geometric surface?
    {
      return Spectral3{0.};
    }

    //microfacet_distribution_val *= Heaviside(ns_dot_wh);
    
    Spectral3 fresnel_term = SchlicksApproximation(color, std::abs(wh_dot_in));
    assert (fresnel_term.allFinite());
    double monochromatic_terms = geometry_term*microfacet_distribution_val*0.25/(std::abs(ns_dot_in*ns_dot_out)+Epsilon);
    assert(std::isfinite(monochromatic_terms));
    return monochromatic_terms*fresnel_term;
  }
  
  
  std::pair<Double3, Double3> Sample(const Double3 &wi, Sampler& sampler) const
  {
    Double3 wh = ndf.SampleHalfVector(sampler.UniformUnitSquare());
  #ifdef SAMPLE_VNDF
    VisibleNdfVCavity::Sample(wh, wi, sampler.Uniform01());
  #endif
    const Double3 out_direction = Reflected(wi, wh);
    return std::make_pair(wh, out_direction);
  }
};

}


MicrofacetShader::MicrofacetShader(
  const SpectralN &_glossy_reflectance,
  double _glossy_exponent,
  std::shared_ptr<Texture> _glossy_exponent_texture)
  : Shader(),
    kr_s(_glossy_reflectance), 
    alpha_max(_glossy_exponent),
    glossy_exponent_texture(std::move(_glossy_exponent_texture))
{
}


Spectral3 MicrofacetShader::EvaluateBSDF(const Double3 &reverse_incident_dir, const SurfaceInteraction &surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const
{
  LocalFrame frame{surface_hit};
  double alpha = MaybeMultiplyTextureLookup(alpha_max, glossy_exponent_texture.get(), surface_hit);
  BeckmanDistribution ndf{alpha};
  Spectral3 kr_s_taken = Take(kr_s, context.lambda_idx);
  const Double3 wi = frame.m_local_inv * reverse_incident_dir;
  const Double3 wo = frame.m_local_inv * out_direction;
  const Double3 wh = Normalized(wi+wo);
  return MicrofacetShaderWrapper{context, ndf, frame, kr_s_taken}.Evaluate(wi, wh, wo, pdf);
}


ScatterSample MicrofacetShader::SampleBSDF(const Double3 &reverse_incident_dir, const SurfaceInteraction &surface_hit, Sampler& sampler, const PathContext &context) const
{
  LocalFrame frame{surface_hit};
  double alpha = MaybeMultiplyTextureLookup(alpha_max, glossy_exponent_texture.get(), surface_hit);
  BeckmanDistribution ndf{alpha};
  Spectral3 kr_s_taken = Take(kr_s, context.lambda_idx);
  MicrofacetShaderWrapper brdf{context, ndf, frame, kr_s_taken};
  const Double3 wi = frame.m_local_inv * reverse_incident_dir;
  auto [wh, wo] = brdf.Sample(wi, sampler);
  double pdf = NaN;
  Spectral3 color = brdf.Evaluate(wi, wh, wo, &pdf);
  return ScatterSample {
    frame.m_local * wo,
    color,
    pdf
  };
}

  
  

struct GlossyTransmissiveDielectricWrapper
{
  const Double3 wi;
  const PathContext &context;
  const BeckmanDistribution &ndf;
  const LocalFrame &frame; 
  const double eta_i_over_t; // eta_i refers to ior on the side of the incomming random walk!
  
  double Evaluate(const Double3& wo, double *pdf) const
  {
    assert(Dot(frame.ng, wi)>=0.);
    const double n_dot_out = Dot(frame.ng, wo);
    const double ns_dot_out = wo[2];
    const double ns_dot_in  = wi[2];  
    
    if (pdf)
    {    
      *pdf = TransmissiveMicrofacetDensity{wi, eta_i_over_t, ndf}.Pdf(wo);
    }
    
    if (n_dot_out >= 0) // Evaluate BRDF
    {
      const Double3 whr = HalfVector(wi, wo);
      const double ns_dot_wh = whr[2];
      const double wh_dot_out = Dot(wo, whr);
      const double wh_dot_in  = Dot(wi, whr);
      
      const double fr_whr = FresnelReflectivity(std::abs(wh_dot_in), eta_i_over_t);  
      const double ndf_reflect = ndf.EvalByHalfVector(std::abs(ns_dot_wh));
      const double geometry_term = G2VCavity(wh_dot_in, wh_dot_out, ns_dot_in, ns_dot_out, ns_dot_wh);

      double result = fr_whr*geometry_term*ndf_reflect*0.25/(std::abs(ns_dot_in*ns_dot_out)+Epsilon);
      return result;
    }
    else // Evaluate BTDF
    {
      boost::optional<Double3> wht_ = HalfVectorRefracted(wi, wo, eta_i_over_t);
      if (!wht_)
        return 0.;
      const Double3 wht = (*wht_)[2]<0. ? -(*wht_) : *wht_ ;
      const double ns_dot_wh = wht[2];
      const double wh_dot_out = Dot(wo, wht);
      const double wh_dot_in  = Dot(wi, wht);
      
      const double fr_wht = FresnelReflectivity(std::abs(wh_dot_in), eta_i_over_t);
      const double ndf_transm = ndf.EvalByHalfVector(std::abs(ns_dot_wh));
      const double J_wh_to_wo = HalfVectorPdfToTransmittedPdf(1.0, eta_i_over_t, wh_dot_in, wh_dot_out);
      const double geometry_term = G2VCavityTransmissive(wh_dot_in, wh_dot_out, ns_dot_in, ns_dot_out, ns_dot_wh);
      
      const double result = (1.0-fr_wht)*std::abs(wh_dot_in)*geometry_term*ndf_transm*J_wh_to_wo/(std::abs(ns_dot_in*ns_dot_out)+Epsilon);

      return result;
    }
  }
  
  
  Double3 Sample(Sampler& sampler) const
  {
    return TransmissiveMicrofacetDensity{wi, eta_i_over_t, ndf}.
      Sample(sampler.UniformUnitSquare(), sampler.Uniform01());
  }
};



GlossyTransmissiveDielectricShader::GlossyTransmissiveDielectricShader::GlossyTransmissiveDielectricShader(double _ior_ratio, double alpha_, double alpha_min_, std::shared_ptr<Texture> glossy_exponent_texture_)
  : ior_ratio{_ior_ratio}, alpha_max{alpha_}, alpha_min{alpha_min_}, glossy_exponent_texture{glossy_exponent_texture_}
{
}



Spectral3 GlossyTransmissiveDielectricShader::EvaluateBSDF(const Double3 &reverse_incident_dir, const SurfaceInteraction &surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const
{
  LocalFrame frame{surface_hit};
  const Double3 wi = frame.m_local_inv * reverse_incident_dir;
  const Double3 wo = frame.m_local_inv * out_direction;
  const double eta_i_over_t = (Dot(surface_hit.geometry_normal, reverse_incident_dir)<0.) ? ior_ratio : 1.0/ior_ratio;
  
  const double alpha = alpha_min + MaybeMultiplyTextureLookup(alpha_max-alpha_min, glossy_exponent_texture.get(), surface_hit);
  BeckmanDistribution ndf{alpha};
  
  GlossyTransmissiveDielectricWrapper shd{ wi, context, ndf, frame, eta_i_over_t };
  const double result = shd.Evaluate(wo, pdf);
  return Spectral3{result};
}


ScatterSample GlossyTransmissiveDielectricShader::SampleBSDF(const Double3 &reverse_incident_dir, const SurfaceInteraction &surface_hit, Sampler& sampler, const PathContext &context) const
{
  LocalFrame frame{surface_hit};
  const Double3 wi = frame.m_local_inv * reverse_incident_dir;
  const double eta_i_over_t = (Dot(surface_hit.geometry_normal, reverse_incident_dir)<0.) ? ior_ratio : 1.0/ior_ratio;
  
  const double alpha = alpha_min + MaybeMultiplyTextureLookup(alpha_max-alpha_min, glossy_exponent_texture.get(), surface_hit);
  BeckmanDistribution ndf{alpha};
  
  GlossyTransmissiveDielectricWrapper bsdf{wi, context, ndf, frame, eta_i_over_t};
  
  Double3 wo = bsdf.Sample(sampler);
  double pdf = NaN;
  double value = bsdf.Evaluate(wo, &pdf);
  return ScatterSample {
    frame.m_local * wo,
    Spectral3{value},
    pdf
  };
}




Spectral3 InvisibleShader::EvaluateBSDF(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const
{
  if (pdf)
    *pdf = 0.;
  return Spectral3::Zero();
}


ScatterSample InvisibleShader::SampleBSDF(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, Sampler& sampler, const PathContext &context) const
{
  return ScatterSample{
    -incident_dir, 
    Spectral3{-1. / std::abs(Dot(incident_dir, surface_hit.shading_normal))},
    Pdf::MakeFromDelta(1.)};
}






SpecularDenseDielectricShader::SpecularDenseDielectricShader(const double _specular_reflectivity, const SpectralN& _diffuse_reflectivity, std::shared_ptr<Texture> _diffuse_texture): 
  Shader(), diffuse_part{_diffuse_reflectivity, std::move(_diffuse_texture)}, specular_reflectivity{_specular_reflectivity}
{
  
}


namespace SmoothAndDenseDielectricDetail
{
// Symmetry demands that f(w1,w2)=f(w2,w1). Therefore, following, Klemen & Kalos, I use the factors 
// (1-R(w1))*(1-R(w2)) where R(w) is the reflective albedo of the specular part given and w the incidence direction. 
// Ref: Klemen & Kalos (2001) "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling", 
  
double DiffuseAttenuationFactor(double albedo1, double albedo2, double average_albedo) 
{
  assert(0 <= average_albedo && average_albedo <= 1.);
  assert(0 <= albedo1 && albedo1 <= 1.);
  assert(0 <= albedo2 && albedo2 <= 1.);
  double normalization = 1./(1.-average_albedo);
  // Another factor 1/Pi comes from the normalization built into the Diffuse shader class.
  return (1.-albedo1)*(1.-albedo2)*normalization;
}
}


Spectral3 SpecularDenseDielectricShader::EvaluateBSDF(const Double3& reverse_incident_dir, const SurfaceInteraction& surface_hit, const Double3& out_direction, const PathContext& context, double* pdf) const
{ 
  double cos_out_n = Dot(surface_hit.normal, out_direction);
  if (cos_out_n > 0.)
  {
    double cos_shn_exitant = std::max(0., Dot(surface_hit.shading_normal, out_direction));
    double cos_shn_incident = std::max(0., Dot(surface_hit.shading_normal, reverse_incident_dir));
    
    double reflected_fraction = SchlicksApproximation(specular_reflectivity, cos_shn_incident);
    double other_reflection_term = SchlicksApproximation(specular_reflectivity, cos_shn_exitant);  
    double average_albedo = AverageOfProjectedSchlicksApproximationOverHemisphere<double>(specular_reflectivity);
   
    Spectral3 brdf_value = diffuse_part.EvaluateBSDF(reverse_incident_dir, surface_hit, out_direction, context, pdf); 
    brdf_value *= SmoothAndDenseDielectricDetail::DiffuseAttenuationFactor(
      reflected_fraction, other_reflection_term, average_albedo);
    
    if (pdf)
      *pdf *= (1.-reflected_fraction);
    return brdf_value;
  }
  else
  {
    if (pdf)
      *pdf = 0.;
    return Spectral3{0.};
  }
}



ScatterSample SpecularDenseDielectricShader::SampleBSDF(const Double3& reverse_incident_dir, const SurfaceInteraction& surface_hit, Sampler& sampler, const PathContext& context) const
{
  double cos_shn_incident = std::max(0., Dot(surface_hit.shading_normal, reverse_incident_dir));
  double reflected_fraction = SchlicksApproximation(specular_reflectivity, cos_shn_incident);
  assert(reflected_fraction >= 0. && reflected_fraction <= 1.);
  double decision_var = sampler.Uniform01();
  ScatterSample smpl;
  if (decision_var < reflected_fraction)
  {
    Double3 refl_dir = Reflected(reverse_incident_dir, surface_hit.shading_normal);
    double cos_rn = Dot(surface_hit.normal, refl_dir);
    if (cos_rn >= 0.)
    {
      smpl = ScatterSample{refl_dir, Spectral3{reflected_fraction/(cos_shn_incident+Epsilon)}, reflected_fraction};
    }
    else
    {
      smpl = ScatterSample{refl_dir, Spectral3{0.}, reflected_fraction};
    }
    SetPmfFlag(smpl);
  }
  else
  {
    smpl = diffuse_part.SampleBSDF(reverse_incident_dir, surface_hit, sampler, context);
    double cos_n_exitant = std::max(0., Dot(surface_hit.shading_normal, smpl.coordinates));
    double other_reflection_term = SchlicksApproximation(specular_reflectivity, cos_n_exitant);
    double average_albedo = AverageOfProjectedSchlicksApproximationOverHemisphere<double>(specular_reflectivity);
    smpl.value *= SmoothAndDenseDielectricDetail::DiffuseAttenuationFactor(
      reflected_fraction, other_reflection_term, average_albedo);
    smpl.pdf_or_pmf *= (1.-reflected_fraction);
  }
  return smpl;
}



/*************************************
 * Media
 ***********************************/

// Just to make the compiler happy. We don't really want an implementation for this.
// TODO: Use interface class. Multi inherit from regular medium and, say, IEmissiveMedium, if a medium is emissive.
// Then dynamic_cast to determine if we have to consider emission. I'm not sure. Usually this is considered bad design.
// But so is a default implementation with arbitrary return values.
// I could decompose the medium class into an emissive component and a scattering/absorbing part, but it would make the implementation more messy.
Medium::VolumeSample Medium::SampleEmissionPosition(Sampler &sampler, const PathContext &context) const
{
  return VolumeSample{ /* pos = */ Double3::Zero() };
}

Spectral3 Medium::EvaluateEmission(const Double3 &pos, const PathContext &context, double *pos_pdf) const
{
  if (pos_pdf) *pos_pdf = 0;
  return Spectral3::Zero();
}


/*****************************************
* Derived media classes 
****************************************/

EmissiveDemoMedium::EmissiveDemoMedium(double _sigma_s, double _sigma_a, double extra_emission_multiplier_, double temperature, const Double3 &pos_, double radius_, int priority)
  : Medium(priority, true), sigma_s{_sigma_s}, sigma_a{_sigma_a}, sigma_ext{_sigma_s + _sigma_a}, spectrum{Color::MaxwellBoltzmanDistribution(temperature)}, pos{pos_}, radius{radius_}
{
  one_over_its_volume = 1./(UnitSphereVolume*std::pow(radius, 3));
  spectrum *= extra_emission_multiplier_;
}


Spectral3 EmissiveDemoMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  return phasefunction.Evaluate(indcident_dir, out_direction, pdf);
}


Medium::PhaseSample EmissiveDemoMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler, const PathContext &context) const
{
  return phasefunction.SampleDirection(incident_dir, sampler);
}


Medium::InteractionSample EmissiveDemoMedium::SampleInteractionPoint(const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context) const
{
  auto [ok, tnear, tfar] = ClipRayToSphereInterior(segment.ray.org, segment.ray.dir, 0, segment.length, this->pos, this->radius);
  Medium::InteractionSample smpl;
  if (ok)
  {
    smpl.t = - std::log(1-sampler.Uniform01()) / sigma_ext;
    smpl.t += tnear;
    if (smpl.t < tfar)
    {
      smpl.weight = 1.0 / sigma_ext;
      smpl.sigma_s = sigma_s;
      return smpl;
    } // else there is no interaction within the sphere.
  } // else didn't hit the sphere.
  
  // Must return something larger than the segment length, so the rendering algo knows that there is no interaction with the medium.
  smpl.t = LargeNumber;
  smpl.weight = 1.0;
  smpl.sigma_s = Spectral3::Zero();
  return smpl;
}


VolumePdfCoefficients EmissiveDemoMedium::ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const
{
  auto [ok, tnear, tfar] = ClipRayToSphereInterior(segment.ray.org, segment.ray.dir, 0, segment.length, this->pos, this->radius);
  const bool end_is_in_emissive_volume = LengthSqr(segment.EndPoint() - this->pos) < radius*radius;
  const bool start_is_in_emissive_volume = LengthSqr(segment.ray.org - this->pos) < radius*radius;
  double tr = ok ? std::exp(-sigma_ext*(tfar - tnear)) : 1;
  double end_sigma_ext = end_is_in_emissive_volume ? sigma_ext : 0;
  double start_sigma_ext = start_is_in_emissive_volume ? sigma_ext : 0;
  return VolumePdfCoefficients{
    end_sigma_ext*tr,
    start_sigma_ext*tr,
    tr,
  };
}


Spectral3 EmissiveDemoMedium::EvaluateTransmission(const RaySegment& segment, Sampler &sampler, const PathContext &context) const
{
  auto [ok, tnear, tfar] = ClipRayToSphereInterior(segment.ray.org, segment.ray.dir, 0, segment.length, this->pos, this->radius);
  double tr = ok ? std::exp(-sigma_ext*(tfar - tnear)) : 1;
  return Spectral3{tr};
}


void EmissiveDemoMedium::ConstructShortBeamTransmittance(const RaySegment& segment, Sampler& sampler, const PathContext& context, PiecewiseConstantTransmittance& pct) const
{
  throw std::runtime_error("not implemented");
}



Medium::VolumeSample EmissiveDemoMedium::SampleEmissionPosition(Sampler &sampler, const PathContext &context) const
{
  Double3 r { sampler.Uniform01(), sampler.Uniform01(), sampler.Uniform01() };
  Double3 pos = SampleTrafo::ToUniformSphere3d(r)*radius + this->pos;
  return { pos };
}


Spectral3 EmissiveDemoMedium::EvaluateEmission(const Double3 &pos, const PathContext &context, double *pos_pdf) const
{
  const bool in_emissive_volume = LengthSqr(pos - this->pos) < radius*radius;
  if (pos_pdf)
    *pos_pdf = in_emissive_volume ? one_over_its_volume : 0.;
  return in_emissive_volume ? (sigma_a*Take(spectrum, context.lambda_idx)).eval() : Spectral3::Zero();
}


Medium::MaterialCoefficients EmissiveDemoMedium::EvaluateCoeffs(const Double3& pos, const PathContext& context) const
{
  throw std::runtime_error("not implemented");
}


////////////////////////////////////////////////////////////
Spectral3 VacuumMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  if (pdf)
    *pdf = 1.;
  return Spectral3{0.}; // Because it is a delta function.
}


Medium::InteractionSample VacuumMedium::SampleInteractionPoint(const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context) const
{
  return Medium::InteractionSample{
      LargeNumber,
      Spectral3{1.},
      Spectral3::Zero(),
    };
}


ScatterSample VacuumMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler, const PathContext &context) const
{
  return ScatterSample{
    -incident_dir,
    Spectral3{1.},
    1.
  };
}


Spectral3 VacuumMedium::EvaluateTransmission(const RaySegment& segment, Sampler &sampler, const PathContext &context) const
{
  return Spectral3{1.};
}


void VacuumMedium::ConstructShortBeamTransmittance(const RaySegment& segment, Sampler& sampler, const PathContext& context, PiecewiseConstantTransmittance& pct) const
{
  pct.PushBack(InfinityFloat, Spectral3::Ones());
}


VolumePdfCoefficients VacuumMedium::ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const
{
  return VolumePdfCoefficients{
    0.,
    0.,
    1.
  };
}

Medium::MaterialCoefficients VacuumMedium::EvaluateCoeffs(const Double3& pos, const PathContext& context) const
{
  return { 
    Spectral3::Zero(),
    Spectral3::Zero()
  };
}




HomogeneousMedium::HomogeneousMedium(const SpectralN& _sigma_s, const SpectralN& _sigma_a, int priority)
  : Medium(priority), sigma_s{_sigma_s}, sigma_a{_sigma_a}, sigma_ext{sigma_s + sigma_a},
    phasefunction{new PhaseFunctions::Uniform()}
{
}


Spectral3 HomogeneousMedium::EvaluatePhaseFunction(const Double3& incident_dir, const Double3& pos, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  return phasefunction->Evaluate(incident_dir, out_direction, pdf);
}


ScatterSample HomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler, const PathContext &context) const
{
  return phasefunction->SampleDirection(incident_dir, sampler);
}


Medium::InteractionSample HomogeneousMedium::SampleInteractionPoint(const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context) const
{
  // Ref: Kutz et a. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"
  // Much simplified with constant coefficients.
  // Also, very importantly, sigma_s is not multiplied to the final weight! Compare with Algorithm 4, Line 10.
  Medium::InteractionSample smpl{
    0.,
    Spectral3::Ones(),
    Spectral3::Zero()
  };
  // Shadow the member var by the new var taking only the current lambdas.
  const Spectral3 sigma_ext = Take(this->sigma_ext, context.lambda_idx);
  const Spectral3 sigma_s   = Take(this->sigma_s,   context.lambda_idx);
  double sigma_t_majorant = sigma_ext.maxCoeff();
  const Spectral3 sigma_n = sigma_t_majorant - sigma_ext;
  double inv_sigma_t_majorant = 1./sigma_t_majorant;
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
      assert(sigma_n.minCoeff() >= -1.e-3); // By definition of the majorante
      double prob_t, prob_n;
      TrackingDetail::ComputeProbabilitiesHistoryScheme(smpl.weight*initial_weights, {sigma_s, sigma_n}, {prob_t, prob_n});
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


inline Spectral3 HomogeneousMedium::EvaluateTransmissionHomogeneous(double x, const Spectral3& sigma_ext) const
{
  return (-sigma_ext * x).exp();
}


Spectral3 HomogeneousMedium::EvaluateTransmission(const RaySegment& segment, Sampler &sampler, const PathContext &context) const
{
  const Spectral3 sigma_ext = Take(this->sigma_ext, context.lambda_idx);
  return EvaluateTransmissionHomogeneous(segment.length, sigma_ext);
}


void HomogeneousMedium::ConstructShortBeamTransmittance(const RaySegment& segment, Sampler& sampler, const PathContext& context, PiecewiseConstantTransmittance& pct) const
{
  const Spectral3 sigma_ext = Take(this->sigma_ext, context.lambda_idx);
  constexpr auto N = static_size<Spectral3>();
  std::pair<double,int> items[N]; 
  for (int i=0; i<N; ++i)
  {
    items[i].first = - std::log(1-sampler.Uniform01()) / sigma_ext[i];
    items[i].second = i;
    for (int j=i; j>0 && items[j-1].first>items[j].first; --j)
    {
      std::swap(items[j-1], items[j]);
    }
  }
  // Zero out the spectral channels one after another.
  Spectral3 w = Spectral3::Ones();
  for (int i=0; i<N; ++i)
  {
    pct.PushBack(items[i].first, w);
    w[items[i].second] = 0;
  }
}



VolumePdfCoefficients HomogeneousMedium::ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const
{
  // We take the mean over the densities that would be appropriate for single-lambda sampling. So this is only approximate.
  // With Peter Kutz's spectral tracking method the actual pdf is not accessible in closed form.
   Spectral3 sigma_ext = Take(this->sigma_ext, context.lambda_idx);
   double tr = EvaluateTransmissionHomogeneous(segment.length, sigma_ext).mean();
   double e  = sigma_ext.mean();
   return VolumePdfCoefficients{
     e*tr,
     e*tr,
     tr,
   }; // Forward and backward is the same in homogeneous media.
}


Medium::MaterialCoefficients HomogeneousMedium::EvaluateCoeffs(const Double3& pos, const PathContext& context) const
{
  return {
    Take(this->sigma_s, context.lambda_idx),
    Take(this->sigma_ext, context.lambda_idx)
  };
}






MonochromaticHomogeneousMedium::MonochromaticHomogeneousMedium(double _sigma_s, double _sigma_a, int priority)
  : Medium(priority), sigma_s{_sigma_s}, sigma_ext{_sigma_s + _sigma_a},
    phasefunction{new PhaseFunctions::Uniform()}
{
}


Spectral3 MonochromaticHomogeneousMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  return phasefunction->Evaluate(indcident_dir, out_direction, pdf);
}


Medium::PhaseSample MonochromaticHomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler, const PathContext &context) const
{
  return phasefunction->SampleDirection(incident_dir, sampler);
}


Medium::InteractionSample MonochromaticHomogeneousMedium::SampleInteractionPoint(const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context) const
{
  Medium::InteractionSample smpl;
  smpl.t = - std::log(1-sampler.Uniform01()) / sigma_ext;
  smpl.t = smpl.t < LargeNumber ? smpl.t : LargeNumber;
  smpl.weight = (smpl.t >= segment.length) ? 
    1.0  // This is transmittance divided by probability to pass through the medium undisturbed which happens to be also the transmittance. Thus this simplifies to one.
    :
    (1.0 / sigma_ext); // Transmittance divided by interaction pdf.
  smpl.sigma_s = sigma_s;
  return smpl;
}


VolumePdfCoefficients MonochromaticHomogeneousMedium::ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const
{
  double tr = std::exp(-sigma_ext*segment.length);
  return VolumePdfCoefficients{
    sigma_ext*tr,
    sigma_ext*tr,
    tr,
  }; // Forward and backward is the same in homogeneous media.
}

Spectral3 MonochromaticHomogeneousMedium::EvaluateTransmission(const RaySegment& segment, Sampler &sampler, const PathContext &context) const
{
  return Spectral3{std::exp(-sigma_ext * segment.length)};
}


void MonochromaticHomogeneousMedium::ConstructShortBeamTransmittance(const RaySegment& segment, Sampler& sampler, const PathContext& context, PiecewiseConstantTransmittance& pct) const
{
  double t = - std::log(1-sampler.Uniform01()) / sigma_ext;
  pct.PushBack(t, Spectral3::Ones());
}


Medium::MaterialCoefficients MonochromaticHomogeneousMedium::EvaluateCoeffs(const Double3& pos, const PathContext& context) const
{
  return {
    Spectral3{sigma_s},
    Spectral3{sigma_ext}
  };
}

