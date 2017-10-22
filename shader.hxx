#ifndef SHADER_HXX
#define SHADER_HXX

#include"vec3f.hxx"
#include"texture.hxx"


class Sampler;
class Ray;
class RaySegment;
class Scene;
class RaySurfaceIntersection;


struct BSDFSample
{
  Double3 dir;
  Spectral scatter_function;
  double pdf;
};


enum ShaderFlags : int
{
  REFLECTION_IS_SPECULAR = 1,
  IS_PASSTHROUGH = 3
};


class Shader
{
  int flags;
public:
  Shader(int _flags = 0) : flags(_flags) {}
  virtual ~Shader() {}
  virtual BSDFSample SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const = 0;
  virtual Spectral EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3 &out_direction, double *pdf) const = 0;
  bool IsReflectionSpecular() const { return flags & REFLECTION_IS_SPECULAR; }
  bool IsPassthrough() const { return flags & IS_PASSTHROUGH; }
};


class DiffuseShader : public Shader
{
  Spectral kr_d; // between zero and 1/Pi.
public:
  DiffuseShader(const Spectral &reflectance);
  BSDFSample SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const override;
  Spectral EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, double *pdf) const override;
};


class InvisibleShader : public Shader
{
public:
  InvisibleShader() : Shader(IS_PASSTHROUGH) {}
  BSDFSample SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler) const override;
  Spectral EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, double *pdf) const override;
};



class Medium
{
public:
  struct InteractionSample
  {
    double t;
    // Following PBRT pg 893, the returned weight is either
    // beta_surf = T(t_intersect)/p_surf if the sampled t lies beyond the end of the ray, i.e. t > t_intersect, or
    // beta_med = sigma_s(t) T(t) / p(t) 
    Spectral weight;
  };
  struct PhaseSample
  {
    Double3 dir;
    Spectral phase_function;
    double pdf;
  };
  const int priority;
  Medium(int _priority) : priority(_priority) {}
  virtual ~Medium() {}
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler) const = 0;
  virtual Spectral EvaluateTransmission(const RaySegment &segment) const = 0;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler) const = 0;
  virtual Spectral EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const = 0;
};



class VacuumMedium : public Medium
{
public:
  VacuumMedium() : Medium(-1) {}
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler) const override;
  virtual Spectral EvaluateTransmission(const RaySegment &segment) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler) const override;
  virtual Spectral EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const override;
};


class IsotropicHomogeneousMedium : public Medium
{
  Spectral sigma_s, sigma_a, sigma_ext;
public:
  IsotropicHomogeneousMedium(const Spectral &_sigma_s, const Spectral &_sigma_a, int priority); 
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler) const override;
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const Spectral &beta) const;
  virtual Spectral EvaluateTransmission(const RaySegment &segment) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler) const override;
  virtual Spectral EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const override;
};


class MonochromaticIsotropicHomogeneousMedium : public Medium
{
  double sigma_s, sigma_a, sigma_ext;
public:
  MonochromaticIsotropicHomogeneousMedium(double _sigma_s, double _sigma_a, int priority); 
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler) const override;
  virtual Spectral EvaluateTransmission(const RaySegment &segment) const override;
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
