#include "shader.hxx"

#include "ray.hxx"
#include "scene.hxx"


DiffuseShader::DiffuseShader(const Spectral &_reflectance)
  : kr_d(_reflectance)
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


Spectral DiffuseShader::EvaluateBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3& out_direction, double *pdf) const
{
  if (pdf)
    *pdf = CosWeightedHemispherePdf(surface_hit, out_direction);
  return kr_d;
}


BRDFSample DiffuseShader::SampleBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const
{
  auto m = OrthogonalSystemZAligned(surface_hit.normal);
  Double3 v = SampleTrafo::ToCosHemisphere(sampler.UniformUnitSquare());
  v = m * v;
  double pdf = CosWeightedHemispherePdf(surface_hit, v);
  return BRDFSample{v, kr_d, pdf};
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