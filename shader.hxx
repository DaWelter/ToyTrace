#ifndef SHADER_HXX
#define SHADER_HXX

#include"vec3f.hxx"
#include"texture.hxx"



class Ray;
class Scene;
class RaySurfaceIntersection;


struct BRDFSample
{
  Double3 dir;
  Spectral scatter_function;
  double pdf;
};


class Shader
{
public:
  virtual ~Shader() {}
  virtual BRDFSample SampleBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit) const = 0;
  virtual Spectral EvaluateBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3 &out_direction, double *pdf) const = 0;
};


class DiffuseShader : public Shader
{
  Spectral albedo; // between zero and one.
public:
  DiffuseShader(const Spectral &albedo);
  BRDFSample SampleBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit) const override;
  Spectral EvaluateBRDF(const Double3 &incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, double *pdf) const override;
};


// class FlatShader : public Shader
// {
// 	Double3 col;
// public:
// 	FlatShader(const Double3 &col) : col(col) {}
// 	virtual Double3 Shade(Ray &ray,Scene *scene) { return col; }
// };
// 
// 
// class PhongShader : public Shader
// {
// 	Double3 ca,cd,cs;
// 	double kr,ka,kd,ks,ke;
// public:
// 	PhongShader(const Double3 &ambient,
// 				const Double3 &diffuse,
// 				const Double3 &specular,
// 				double ka,double kd,double ks,double ke,double kr) 
// 				:	ca(ambient),
// 					cd(diffuse),
// 					cs(specular),
// 					ka(ka),kd(kd),ks(ks),ke(ke),kr(kr)
// 	{}
// 	virtual Double3 Shade(Ray &ray,Scene *scene);
// };

#endif
