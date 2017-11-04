#include "shader.hxx"

#include "ray.hxx"
#include "scene.hxx"


DiffuseShader::DiffuseShader(const Spectral &_reflectance)
  : Shader(0),
    kr_d(_reflectance)
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
  double n_dot_out = Dot(surface_hit.normal, out_direction);
  if (pdf)
    *pdf = n_dot_out>0. ? n_dot_out/Pi : 0.;
  return n_dot_out > 0. ? kr_d : Spectral{0.};
}


BSDFSample DiffuseShader::SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(surface_hit.normal);
  Double3 v = SampleTrafo::ToCosHemisphere(sampler.UniformUnitSquare());
  double pdf = v[2]/Pi;
  v = m * v;
  return BSDFSample{v, kr_d, pdf};
}




TexturedDiffuseShader::TexturedDiffuseShader(const Spectral &_reflectance, std::unique_ptr<Texture> _texture)
  : Shader(0),
    kr_d(_reflectance),
    texture(std::move(_texture))
{
  // See Diffuse Shader
  kr_d *= 1./Pi;
}


Spectral TexturedDiffuseShader::EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3& out_direction, double *pdf) const
{
  double n_dot_out = Dot(surface_hit.normal, out_direction);
  if (pdf)
    *pdf = n_dot_out>0. ? n_dot_out/Pi : 0.;
  Double3 uv = surface_hit.primitive().GetUV(surface_hit.hitid);
  Spectral texture_color = texture->GetTexel(uv[0], uv[1]).array();
  Spectral out_color = texture_color * kr_d;
  return n_dot_out > 0. ? out_color : Spectral{0.};
}


BSDFSample TexturedDiffuseShader::SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(surface_hit.normal);
  Double3 v = SampleTrafo::ToCosHemisphere(sampler.UniformUnitSquare());
  double pdf = v[2]/Pi;
  v = m * v;
  return BSDFSample{v, kr_d, pdf};
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





Spectral VacuumMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
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


PhaseFunctions::Sample VacuumMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler) const
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


Spectral HomogeneousMedium::EvaluatePhaseFunction(const Double3& incident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
{
  return phasefunction->Evaluate(incident_dir, pos, out_direction, pdf);
}


PhaseFunctions::Sample HomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler) const
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


Spectral MonochromaticHomogeneousMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
{
  return phasefunction->Evaluate(indcident_dir, pos, out_direction, pdf);
}


Medium::PhaseSample MonochromaticHomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler) const
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