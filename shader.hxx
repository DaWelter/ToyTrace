#ifndef SHADER_HXX
#define SHADER_HXX

#include"vec3f.hxx"
#include"texture.hxx"



class Ray;
class Scene;
class SurfaceHit;

// class Shader
// {
// public:
// 	virtual Double3 Shade(Ray &ray,Scene *scene) = 0;
// };


struct DirectionSample
{
  Double3 v;
  double scatter_function;
  double pdf;
};


class Shader
{
public:
  virtual ~Shader() {}
  virtual DirectionSample SampleBRDF(const SurfaceHit &inbound_surface_hit) const = 0;
  virtual Double3 EvaluateBRDF(const SurfaceHit &inbound_surface_hit, const Double3 &out_direction) const = 0;
  virtual double EvaluatePDF(const SurfaceHit &inbound_surface_hit, const Double3 &out_direction) const = 0;
};


class DiffuseShader : public Shader
{
  Double3 albedo; // between zero and one.
public:
  DiffuseShader(const Double3 &albedo);
  DirectionSample SampleBRDF(const SurfaceHit &inbound_surface_hit) const override;
  Double3 EvaluateBRDF(const SurfaceHit& inbound_surface_hit, const Double3& out_direction) const override;
  double EvaluatePDF(const SurfaceHit &inbound_surface_hit, const Double3 &out_direction) const override;
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
