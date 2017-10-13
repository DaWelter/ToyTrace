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
      Spectral{1.},
      Spectral{0.}, Spectral{0.},
      1.
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
  smpl.sigma_a = sigma_a;
  smpl.sigma_s = sigma_s;
  // TODO: Tower sampling based on sigma_ext
  int component = sampler.UniformInt(0, 2);
  smpl.t = - std::log(1-sampler.Uniform01()) / sigma_ext[component];
  smpl.t = smpl.t > LargeNumber ? LargeNumber : smpl.t;
  Spectral weights{1./static_size<Spectral>()};
  smpl.pdf = (weights * sigma_ext * (-sigma_ext * smpl.t).exp()).sum();
  smpl.transmission = (smpl.t >= segment.length) ? 
                      Spectral{0.} :
                      (-sigma_ext * smpl.t).exp();
  return smpl;
}


Spectral IsotropicHomogeneousMedium::EvaluateTransmission(const RaySegment& segment) const
{
  return (-sigma_ext * segment.length).exp();
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