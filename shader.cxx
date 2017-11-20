#include "shader.hxx"
#include "ray.hxx"
#include "scene.hxx"

namespace
{
inline Spectral MaybeMultiplyTextureLookup(const Spectral &color, const Texture *tex, const RaySurfaceIntersection &surface_hit)
{
  Spectral ret{color};
  if (tex)
  {
    Double3 uv = surface_hit.primitive().GetUV(surface_hit.hitid);
    ret *= tex->GetTexel(uv[0], uv[1]).array();
  }
  return ret;
}
}



DiffuseShader::DiffuseShader(const Spectral &_reflectance, std::unique_ptr<Texture> _diffuse_texture)
  : Shader(IS_REFLECTIVE),
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


Spectral DiffuseShader::EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3& out_direction, double *pdf) const
{
  double n_dot_out = Dot(surface_hit.shading_normal, out_direction);
  if (pdf)
    *pdf = n_dot_out>0. ? n_dot_out/Pi : 0.;
  if (n_dot_out > 0.)
  {
    return MaybeMultiplyTextureLookup(kr_d, diffuse_texture.get(), surface_hit);
  }
  else
    return Spectral{0.};
}


BSDFSample DiffuseShader::SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(surface_hit.shading_normal);
  Double3 v = SampleTrafo::ToCosHemisphere(sampler.UniformUnitSquare());
  double pdf = v[2]/Pi;
  Double3 out_direction = m * v;
  Spectral value = Dot(out_direction, surface_hit.normal)>0 ? kr_d : Spectral{0.};
  return BSDFSample{out_direction, value, pdf};
}



SpecularReflectiveShader::SpecularReflectiveShader(const Spectral& reflectance)
  : Shader(REFLECTION_IS_SPECULAR|IS_REFLECTIVE),
    kr_s(reflectance)
{
}


Spectral SpecularReflectiveShader::EvaluateBSDF(const Double3& incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, double* pdf) const
{
  if (pdf)
    *pdf = 0.;
  return Spectral{0.};
}


BSDFSample SpecularReflectiveShader::SampleBSDF(const Double3& incident_dir, const RaySurfaceIntersection& surface_hit, Sampler& sampler) const
{
  Double3 r = Reflected(incident_dir, surface_hit.shading_normal);
  double cos_rn = Dot(surface_hit.normal, r);
  if (cos_rn < 0.)
    return BSDFSample{r, Spectral{0.}, 1.};
  else
  {
    double cos_rsdn = Dot(surface_hit.shading_normal, r);
    return BSDFSample{r, kr_s/cos_rsdn, 1.};
  }
}



namespace ModifiedPhongDetail
{
inline std::pair<double,double> PdfAndFactor(double cos_phi, double alpha)
{
  double t =  0.5/Pi*std::pow(cos_phi, alpha);
  return std::make_pair(t*(alpha+1.), t*(alpha+2.));
}

inline double CosPhi(const Double3 &reverse_incident_dir, const Double3 &normal, const Double3& out_direction, double alpha)
{
  // cos_phi is w.r.t the angle between outgoing in reflected incident directions.
  double cos_phi = std::max(0., Dot(Reflected(reverse_incident_dir, normal), out_direction));
  return cos_phi;
}
}


ModifiedPhongShader::ModifiedPhongShader(
  const Spectral &_diffuse_reflectance, std::unique_ptr<Texture> _diffuse_texture,
  const Spectral &_glossy_reflectance, std::unique_ptr<Texture> _glossy_texture,
  double _glossy_exponent)
  : Shader(0),
    kr_d(_diffuse_reflectance), 
    kr_s(_glossy_reflectance), 
    alpha(_glossy_exponent),
    diffuse_texture(std::move(_diffuse_texture)),
    glossy_texture(std::move(_glossy_texture))
{
}


Spectral ModifiedPhongShader::EvaluateBSDF(const Double3 &reverse_incident_dir, const RaySurfaceIntersection &surface_hit, const Double3& out_direction, double *pdf) const
{
  double n_dot_out = Dot(surface_hit.normal, out_direction);
  double cos_phi = ModifiedPhongDetail::CosPhi(reverse_incident_dir, surface_hit.shading_normal, out_direction, alpha);
  auto pdf_and_factor = ModifiedPhongDetail::PdfAndFactor(cos_phi, alpha);

  if (pdf)
  {
    *pdf = pdf_and_factor.first;
  }
  
  if (n_dot_out <= 0.)
    return Spectral{0.};

  auto kr_s_local = MaybeMultiplyTextureLookup(kr_s, glossy_texture.get(), surface_hit);  

  return pdf_and_factor.second * kr_s_local;
}


BSDFSample ModifiedPhongShader::SampleBSDF(const Double3 &reverse_incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const
{
  Double3 reflected = Reflected(reverse_incident_dir, surface_hit.shading_normal);
  auto m = OrthogonalSystemZAligned(reflected);
  Double3 w = SampleTrafo::ToPhongHemisphere(sampler.UniformUnitSquare(), alpha);
  double cos_phi = w[2];
  Double3 out_direction = m * w;
  auto pdf_and_factor = ModifiedPhongDetail::PdfAndFactor(cos_phi, alpha);
  if (Dot(out_direction, surface_hit.normal) < 0.)
  {
    // Reject samples that go under the surface. This is done simply by setting the contribution to zero! It is basic rejection sampling. Or isn't it?
    return BSDFSample{out_direction, Spectral{0.}, pdf_and_factor.first};
  }
  else
  {
    auto kr_s_local = MaybeMultiplyTextureLookup(kr_s, glossy_texture.get(), surface_hit);  
    kr_s_local *= pdf_and_factor.second;
    //double factor = ModifiedPhongDetail::Pdf(reverse_incident_dir, surface_hit.shading_normal, out_direction, alpha);
    //assert(factor <= 0. || pdf > 0.);
    return BSDFSample{out_direction, kr_s_local, pdf_and_factor.second};
  }
}




Spectral InvisibleShader::EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3& out_direction, double *pdf) const
{
  constexpr double tol = Epsilon;
  double u = LengthSqr(incident_dir + out_direction);
  u = u<tol ? 1. : 0.;
  if (pdf)
    *pdf = u;
  return Spectral{u};
}


BSDFSample InvisibleShader::SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const
{
  return BSDFSample{-incident_dir, Spectral{1.}, 1.};
}





Spectral VacuumMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  if (pdf)
    *pdf = 1.;
  return Spectral{0.}; // Because it is a delta function.
}


Medium::InteractionSample VacuumMedium::SampleInteractionPoint(const RaySegment& segment, Sampler& sampler, const PathContext &context) const
{
  return Medium::InteractionSample{
      LargeNumber,
      Spectral{1.}
    };
}


PhaseFunctions::Sample VacuumMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler, const PathContext &context) const
{
  return PhaseFunctions::Sample{
    -incident_dir,
    Spectral{1.},
    1.
  };
}


Spectral VacuumMedium::EvaluateTransmission(const RaySegment& segment, Sampler &sampler, const PathContext &context) const
{
  return Spectral{1.};
}




HomogeneousMedium::HomogeneousMedium(const Spectral& _sigma_s, const Spectral& _sigma_a, int priority)
  : Medium(priority), sigma_s{_sigma_s}, sigma_a{_sigma_a}, sigma_ext{_sigma_s + _sigma_a},
    phasefunction{new PhaseFunctions::Uniform()}
{
}


Spectral HomogeneousMedium::EvaluatePhaseFunction(const Double3& incident_dir, const Double3& pos, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  return phasefunction->Evaluate(incident_dir, pos, out_direction, pdf);
}


PhaseFunctions::Sample HomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler, const PathContext &context) const
{
  return phasefunction->SampleDirection(incident_dir, pos, sampler);
}


Medium::InteractionSample HomogeneousMedium::SampleInteractionPoint(const RaySegment& segment, Sampler& sampler, const PathContext &context) const
{
#if 0
  Medium::InteractionSample smpl;
  /* Collision coefficients that vary with wavelength are handled by a probability
   * density for t that is a sum, i.e. p(t) = 1/n sum_i=1..n sigma_t,i exp(-sigma_t,i t).
   * See PBRT pg 893.
   * This is equivalent to using the balance heuristic for strategies
   * which mean taking a sample implied by the transmittance of some selected channel.
   * The balance heuristic would compute weights which would result in identical
   * smpl.weight coefficients as computed here.
   * Veach shows (p.g. 275) that in the case of a "one-sample" model which I
   * have here this is the best MIS scheme one can use.
  */
  Spectral weights{1./static_size< Spectral >()};
  double r = sampler.Uniform01();
  int component = r<weights[0] ? 0 : (r<(weights[1]+weights[0]) ? 1 : 2);
  smpl.t = - std::log(1-sampler.Uniform01()) / sigma_ext[component];
  smpl.t = smpl.t < LargeNumber ? smpl.t : LargeNumber;
  double t = std::min(smpl.t, segment.length);
  Spectral transmittance = (-sigma_ext * t).exp();
  if (smpl.t < segment.length)
  {
    
    double pdf = (weights * sigma_ext * transmittance).sum();
    smpl.weight = sigma_s * transmittance / pdf;
  }
  else
  {
    double p_surf = (weights * transmittance).sum();
    smpl.weight = transmittance / p_surf;
  }
  return smpl;
#else
  /* Split the integral into one summand per wavelength.
   * However, evaluate only one of the summands probabilistically
   * and ommit the other ones, akin to russian roulette termination.
   * Sample the volume marching integral based on the selected wavelength.
   * Regardless, I transport the full spectrum with each term.
   * Thus I can form a weighted sum over the summands, with coefficients
   * depending on the summand and lambda. For much different collision
   * coefficients (sigma_ext) the coefficients should be identical to the
   * Kroneker Delta, so that one term transports exactly one wavelength.
   * However if all sigma_ext_lamba are equal then the method should
   * transport the full spectrum with the sample currently taken. Thus,
   * recovering the sampling method for monochromatic media.
   * TODO: Generalize for more then 3 wavelengths.
   */
  assert(!context.beta.isZero());
  Spectral lambda_selection_prob = context.beta.abs();
  lambda_selection_prob /= lambda_selection_prob.sum();

#if 0
  // Strict single wavelength sampling!
  Spectral combine_weights[] = {
    Spectral{1, 0, 0},
    Spectral{0, 1, 0},
    Spectral{0, 0, 1}
  };
#else
  Spectral lambda_filter_base = context.beta * sigma_ext;
  double lambda_filter_min = lambda_filter_base.minCoeff();
  double lambda_filter_max = lambda_filter_base.maxCoeff();
  double f = (lambda_filter_max - lambda_filter_min)/(std::abs(lambda_filter_max) + std::abs(lambda_filter_min));
  f = f<0.33 ? 0. : 1.;
  double f_eq = (1.-f) / static_size< Spectral >();
  Spectral combine_weights[] = {
    Spectral{f + f_eq, f_eq, f_eq },
    Spectral{f_eq, f + f_eq, f_eq },
    Spectral{f_eq, f_eq, f + f_eq }
  };
#endif

  Medium::InteractionSample smpl;
  int component = TowerSampling<static_size<Spectral>()>(
        lambda_selection_prob.data(),
        sampler.Uniform01());
  smpl.t = - std::log(sampler.Uniform01()) / sigma_ext[component];
  smpl.t = smpl.t < LargeNumber ? smpl.t : LargeNumber;
  double t = std::min(smpl.t, segment.length);
  Spectral transmittance = (-sigma_ext * t).exp();
  if (smpl.t < segment.length)
  {
    smpl.weight = combine_weights[component] * sigma_s * transmittance / (sigma_ext[component] * transmittance[component] * lambda_selection_prob[component]);
  }
  else
  {
    double p_surf = transmittance[component] * lambda_selection_prob[component];
    smpl.weight = combine_weights[component] * transmittance / p_surf;
  }
  return smpl;
#endif
}


Spectral HomogeneousMedium::EvaluateTransmission(const RaySegment& segment, Sampler &sampler, const PathContext &context) const
{
  return (-sigma_ext * segment.length).exp();
}





MonochromaticHomogeneousMedium::MonochromaticHomogeneousMedium(double _sigma_s, double _sigma_a, int priority)
  : Medium(priority), sigma_s{_sigma_s}, sigma_a{_sigma_a}, sigma_ext{_sigma_s + _sigma_a},
    phasefunction{new PhaseFunctions::Uniform()}
{
}


Spectral MonochromaticHomogeneousMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, const PathContext &context, double* pdf) const
{
  return phasefunction->Evaluate(indcident_dir, pos, out_direction, pdf);
}


Medium::PhaseSample MonochromaticHomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler, const PathContext &context) const
{
  return phasefunction->SampleDirection(incident_dir, pos, sampler);
}


Medium::InteractionSample MonochromaticHomogeneousMedium::SampleInteractionPoint(const RaySegment& segment, Sampler& sampler, const PathContext &context) const
{
  Medium::InteractionSample smpl;
  smpl.t = - std::log(1-sampler.Uniform01()) / sigma_ext;
  smpl.t = smpl.t < LargeNumber ? smpl.t : LargeNumber;
  smpl.weight = (smpl.t >= segment.length) ? 
    1.0
    :
    (sigma_s / sigma_ext);
  return smpl;
}


Spectral MonochromaticHomogeneousMedium::EvaluateTransmission(const RaySegment& segment, Sampler &sampler, const PathContext &context) const
{
  return Spectral{std::exp(-sigma_ext * segment.length)};
}
