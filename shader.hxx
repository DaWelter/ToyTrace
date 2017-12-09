#ifndef SHADER_HXX
#define SHADER_HXX

#include <memory>

#include"vec3f.hxx"
#include"spectral.hxx"
#include"texture.hxx"
#include"phasefunctions.hxx"

class Sampler;
struct Ray;
struct RaySegment;
class Scene;
struct RaySurfaceIntersection;


struct PathContext
{
  explicit PathContext(const Index3 &_lambda_idx) :
    beta{1.},
    lambda_idx(_lambda_idx)
  {}
  Spectral3 beta;
  Index3 lambda_idx;
};


struct BSDFSample
{
  Double3 dir;
  Spectral3 scatter_function;
  double pdf;
};


enum ShaderFlags : int
{
  REFLECTION_IS_SPECULAR = 1,
  TRANSMISSION_IS_SPECULAR = 2,
  IS_PASSTHROUGH = 4,
  IS_TRANSMISSIVE = 8,
  IS_REFLECTIVE = 16,
};


class Shader
{
  int flags;
public:
  Shader(int _flags = 0) : flags(_flags) {}
  virtual ~Shader() {}
  virtual BSDFSample SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler, const PathContext &context) const = 0;
  virtual Spectral3 EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, const Double3 &out_direction, const PathContext &context, double *pdf) const = 0;
  bool IsReflectionSpecular() const { return flags & REFLECTION_IS_SPECULAR; }
  bool IsPassthrough() const { return flags & IS_PASSTHROUGH; }
};


class DiffuseShader : public Shader
{
  SpectralN kr_d; // between zero and 1/Pi.
  std::unique_ptr<Texture> diffuse_texture; // TODO: Share textures among shaders?
public:
  DiffuseShader(const RGB &reflectance, std::unique_ptr<Texture> _diffuse_texture);
  BSDFSample SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler, const PathContext &context) const override;
  Spectral3 EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const override;
};



class SpecularReflectiveShader : public Shader
{
  SpectralN kr_s;
public:
  SpecularReflectiveShader(const RGB &reflectance);
  BSDFSample SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler, const PathContext &context) const override;
  Spectral3 EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const override;
};


class MicrofacetShader : public Shader
{
  SpectralN kr_s;
  double alpha;
  std::unique_ptr<Texture> glossy_texture;
public:
  MicrofacetShader(
    const RGB &_glossy_reflectance, std::unique_ptr<Texture> _glossy_texture,
    double _glossy_exponent
  );
  BSDFSample SampleBSDF(const Double3 &reverse_incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler, const PathContext &context) const override;
  Spectral3 EvaluateBSDF(const Double3 &reverse_incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const override;
};



class InvisibleShader : public Shader
{
public:
  InvisibleShader() : Shader(IS_PASSTHROUGH|REFLECTION_IS_SPECULAR|TRANSMISSION_IS_SPECULAR) {}
  BSDFSample SampleBSDF(const Double3 &incident_dir, const RaySurfaceIntersection &surface_hit, Sampler& sampler, const PathContext &context) const override;
  Spectral3 EvaluateBSDF(const Double3 &incident_dir, const RaySurfaceIntersection& surface_hit, const Double3& out_direction, const PathContext &context, double *pdf) const override;
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
    Spectral3 weight;
  };
  using PhaseSample = PhaseFunctions::Sample;
  
  const int priority;
  Medium(int _priority) : priority(_priority) {}
  virtual ~Medium() {}
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const = 0;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const = 0;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const = 0;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const = 0;
};



class VacuumMedium : public Medium
{
public:
  VacuumMedium() : Medium(-1) {}
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
};


class HomogeneousMedium : public Medium
{
  SpectralN sigma_s, sigma_a, sigma_ext;
public:
  std::unique_ptr<PhaseFunctions::PhaseFunction> phasefunction; // filled by parser
public:
  HomogeneousMedium(const RGB &_sigma_s, const RGB &_sigma_a, int _priority); 
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
};


class MonochromaticHomogeneousMedium : public Medium
{
  double sigma_s, sigma_a, sigma_ext;
public:
  std::unique_ptr<PhaseFunctions::PhaseFunction> phasefunction;
public:
  MonochromaticHomogeneousMedium(double _sigma_s, double _sigma_a, int _priority); 
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
};


#endif
