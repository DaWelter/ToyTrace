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


inline double CosWeightedHemispherePdf(const RaySurfaceIntersection &surface_hit, const Double3& out_direction)
{
  double theta = Dot(surface_hit.normal, out_direction);
  return theta/Pi;
}


Spectral DiffuseShader::EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3& out_direction, double *pdf) const
{
  if (pdf)
    *pdf = CosWeightedHemispherePdf(surface_hit, out_direction);
  return kr_d;
}


BSDFSample DiffuseShader::SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(surface_hit.normal);
  Double3 v = SampleTrafo::ToCosHemisphere(sampler.UniformUnitSquare());
  v = m * v;
  double pdf = CosWeightedHemispherePdf(surface_hit, v);
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


Medium::InteractionSample VacuumMedium::SampleInteractionPoint(const RaySegment& segment, Sampler& sampler) const
{
  return Medium::InteractionSample{
      LargeNumber,
      Spectral{1.}
    };
}


Medium::PhaseSample VacuumMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler) const
{
  return Medium::PhaseSample{
    -incident_dir,
    Spectral{1.},
    1.
  };
}


Spectral VacuumMedium::EvaluateTransmission(const RaySegment& segment) const
{
  return Spectral{1.};
}




IsotropicHomogeneousMedium::IsotropicHomogeneousMedium(const Spectral& _sigma_s, const Spectral& _sigma_a, int priority)
  : Medium(priority), sigma_s{_sigma_s}, sigma_a{_sigma_a}, sigma_ext{_sigma_s + _sigma_a}
{
}


Spectral IsotropicHomogeneousMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
{
  if (pdf)
    *pdf = 1./UnitSphereSurfaceArea;
  return Spectral{1./UnitSphereSurfaceArea};
}


Medium::PhaseSample IsotropicHomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler) const
{
  return Medium::PhaseSample{
    SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare()),
    Spectral{1./UnitSphereSurfaceArea},
    1./UnitSphereSurfaceArea
  };
}


Medium::InteractionSample IsotropicHomogeneousMedium::SampleInteractionPoint(const RaySegment& segment, Sampler& sampler) const
{
  Medium::InteractionSample smpl;
  // This is equivalent to using the balance heuristic for strategies
  // which mean taking a sample implied by the transmittance of some selected channel.
  // The balance heuristic would compute weights which would result in identical
  // smpl.weight coefficients as computed here.
  // Veach shows (p.g. 275) that in the case of a "one-sample" model which I
  // have here this is the best MIS scheme one can use.
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
}


Medium::InteractionSample IsotropicHomogeneousMedium::SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const Spectral &beta) const
{
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
  Spectral lambda_selection_prob = beta.abs();
  lambda_selection_prob /= lambda_selection_prob.sum();

#if 0
  // Strict single wavelength sampling!
  Spectral combine_weights[] = {
    Spectral{1, 0, 0},
    Spectral{0, 1, 0},
    Spectral{0, 0, 1}
  };
#else
  Spectral lambda_filter_base = beta * sigma_ext;
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
  double r = sampler.Uniform01();
  int component = r<lambda_selection_prob[0] ? 0 : (r<(lambda_selection_prob[1]+lambda_selection_prob[0]) ? 1 : 2);
  smpl.t = - std::log(1-sampler.Uniform01()) / sigma_ext[component];
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
}


Spectral IsotropicHomogeneousMedium::EvaluateTransmission(const RaySegment& segment) const
{
  return (-sigma_ext * segment.length).exp();
}





MonochromaticIsotropicHomogeneousMedium::MonochromaticIsotropicHomogeneousMedium(double _sigma_s, double _sigma_a, int priority)
  : Medium(priority), sigma_s{_sigma_s}, sigma_a{_sigma_a}, sigma_ext{_sigma_s + _sigma_a}
{
}


Spectral MonochromaticIsotropicHomogeneousMedium::EvaluatePhaseFunction(const Double3& indcident_dir, const Double3& pos, const Double3& out_direction, double* pdf) const
{
  if (pdf)
    *pdf = 1./UnitSphereSurfaceArea;
  return Spectral{1./UnitSphereSurfaceArea};
}


Medium::PhaseSample MonochromaticIsotropicHomogeneousMedium::SamplePhaseFunction(const Double3& incident_dir, const Double3& pos, Sampler& sampler) const
{
  return Medium::PhaseSample{
    SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare()),
    Spectral{1./UnitSphereSurfaceArea},
    1./UnitSphereSurfaceArea
  };
}


Medium::InteractionSample MonochromaticIsotropicHomogeneousMedium::SampleInteractionPoint(const RaySegment& segment, Sampler& sampler) const
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


Spectral MonochromaticIsotropicHomogeneousMedium::EvaluateTransmission(const RaySegment& segment) const
{
  return Spectral{std::exp(-sigma_ext * segment.length)};
}




/*
Double3 EyeLightShader::Shade(Ray &ray,Scene *scene) 
{
	Double3 normal = ray.hit->GetNormal(ray);
	double dot = -Dot(normal,ray.dir);
	//Assert(dot>0.);
	dot=fabs(dot);
	return col*(dot);
}


Double3 TexturedEyeLightShader::Shade(Ray &ray,Scene *scene) 
{
	Double3 col,texcol;
	col = EyeLightShader::Shade(ray,scene);

	Double3 uv = ray.hit->GetUV(ray);
	texcol = tex.GetTexel(uv[0],uv[1]);
	
	return Product(texcol,col);
}


Double3 ReflectiveEyeLightShader::Shade(Ray &ray,Scene *scene)
{
	Double3 normal = ray.hit->GetNormal(ray);
	double dot = -Dot(normal,ray.dir);
	
	Double3 result = fabs(dot)*col;

	if(ray.level>=MAX_RAY_DEPTH) return result;

	Ray reflRay;
	reflRay.level = ray.level+1;
	reflRay.org = ray.org+ray.t*ray.dir;
	reflRay.t = 1000.;
	reflRay.dir = ray.dir+2.*dot*normal;

	Double3 reflcol = scene->RayTrace(reflRay);

	result = result + reflcol*reflectivity;
	return result;
}



Double3 PhongShader::Shade(Ray &ray,Scene *scene)
{
	Double3 normal = ray.hit->GetNormal(ray);
	Double3 refldir = ray.dir-2.*Dot(normal,ray.dir)*normal;
	Double3 pos = ray.org + ray.t * ray.dir;
	Double3 cold(0),cols(0);
	Double3 res(0);

	Double3 La(1,1,1); // ambient radiation 
	res += ka*Product(La,ca);

	for(int i=0; i<scene->lights.size(); i++)
	{
		Double3 intensity;
		Ray lightray;
		lightray.org = pos;
		if(!scene->lights[i]->Illuminate(lightray,intensity)) continue;
		if(scene->Occluded(lightray)) continue;

		double ddot = Dot(lightray.dir,normal);
		if(ddot>0)
			cold += intensity*Dot(lightray.dir,normal);

		double sdot = Dot(lightray.dir,refldir);
		if(sdot>0) 
			cols += intensity*pow(sdot,ke);
	}
	res += kd*Product(cold,cd) + ks*Product(cols,cs);

	if(ray.level>=MAX_RAY_DEPTH || kr<Epsilon) return res;
	Ray reflRay;
	reflRay.level = ray.level+1;
	reflRay.org = pos;
	reflRay.t = 1000.;
	reflRay.dir = refldir;

	Double3 colr = scene->RayTrace(reflRay);

	res += colr*kr;
	return res;
}*/