#ifndef SHADER_HXX
#define SHADER_HXX

#include"vec3f.hxx"
#include"texture.hxx"


class Sampler;
class Ray;
class RaySegment;
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
  virtual BRDFSample SampleBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const = 0;
  virtual Spectral EvaluateBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3 &out_direction, double *pdf) const = 0;
};


class DiffuseShader : public Shader
{
  Spectral kr_d; // between zero and 1/Pi.
public:
  DiffuseShader(const Spectral &reflectance);
  BRDFSample SampleBRDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const override;
  Spectral EvaluateBRDF(const Double3 &incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, double *pdf) const override;
};


class Medium
{
public:
  struct InteractionSample
  {
    double t;
    Spectral transmission;
    Spectral sigma_a, sigma_s;
    double pdf;
  };
  struct PhaseSample
  {
    Double3 dir;
    Spectral phase_function;
    double pdf;
  };
  
  virtual ~Medium() {}
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler) const = 0;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler) const = 0;
  virtual Spectral EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const = 0;
};


class VacuumMedium : public Medium
{
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler) const override;
  virtual Spectral EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const override;
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
