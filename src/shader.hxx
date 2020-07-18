#ifndef SHADER_HXX
#define SHADER_HXX

//#define PRODUCT_DISTRIBUTION_SAMPLING

#include <memory>
#include <type_traits>
//#include <boost/container/static_vector.hpp>

#include "sampler.hxx"
#include "spectral.hxx"
#include "memory_arena.hxx"

#ifdef PRODUCT_DISTRIBUTION_SAMPLING
#include "distribution_mixture_models.hxx"
#endif

struct TagScatterSample {};
using ScatterSample = Sample<Double3, Spectral3, TagScatterSample>;

class Texture;
struct SurfaceInteraction;
struct PathContext;
struct RaySegment;
class PiecewiseConstantTransmittance;

// Included here because it uses ScatterSample.
#include"phasefunctions.hxx"


class Shader
{
public:
  bool require_monochromatic = false;
  bool prefer_path_tracing_over_photonmap = false;
  bool is_pure_specular = false;
  bool is_pure_diffuse = false;
  bool supports_lobes = false;
  Shader() {}
  virtual ~Shader() {}
  virtual ScatterSample SampleBSDF(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, Sampler& sampler, const PathContext &context) const = 0;
  virtual Spectral3 EvaluateBSDF(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, const Double3 &out_direction, const PathContext &context, double *pdf) const = 0;
  virtual double Pdf(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, const Double3 &out_direction, const PathContext &context) const;
  
  virtual double GuidingProbMixShaderAmount(const SurfaceInteraction &surface_hit) const;
#ifdef PRODUCT_DISTRIBUTION_SAMPLING
  virtual vmf_fitting::VonMisesFischerMixture<2> ComputeLobes(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, const PathContext &context) const;
  virtual void IntializeLobes();
#endif

  //virtual materials::LobeOwner PickTheLobe(const Double3 &incident_dir, const SurfaceInteraction &surface_hit, Sampler &sampler, const PathContext &context) const = 0;
};


std::unique_ptr<Shader> MakeGlossyTransmissiveDielectricShader(
  double _ior_ratio, 
  double alpha_, 
  double alpha_min_, 
  std::shared_ptr<Texture> glossy_exponent_texture_);

std::unique_ptr<Shader> MakeDiffuseShader(
  const SpectralN &reflectance, 
  std::shared_ptr<Texture> _diffuse_texture);

std::unique_ptr<Shader> MakeMicrofacetShader(
  const SpectralN &_glossy_reflectance,
  double _glossy_exponent,
  std::shared_ptr<Texture> _glossy_exponent_texture);

std::unique_ptr<Shader> MakeSpecularTransmissiveDielectricShader(
  double _ior_ratio, 
  double ior_lambda_coeff_ = 0.);

std::unique_ptr<Shader> MakeSpecularDenseDielectricShader(
  const double _specular_reflectivity,
  const SpectralN &_diffuse_reflectivity,
  std::shared_ptr<Texture> _diffuse_texture);

std::unique_ptr<Shader> MakeSpecularReflectiveShader(const SpectralN &reflectance);

std::unique_ptr<Shader> MakeInvisibleShader();










///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Media
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace materials
{

enum MediumFlags : int
{
  IS_SCATTERING = 1,
  IS_EMISSIVE = 2,
  IS_MONOCHROMATIC = 4,
  IS_HOMOGENEOUS = 8
};

inline constexpr MediumFlags operator | (MediumFlags lhs, MediumFlags rhs) noexcept
{
    // Good tip from https://softwareengineering.stackexchange.com/questions/194412/using-scoped-enums-for-bit-flags-in-c
    using T = std::underlying_type_t <MediumFlags>;
    return static_cast<MediumFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

struct MediaCoefficients
{
  Spectral3 sigma_s;
  Spectral3 sigma_t;
};


struct VolumePdfCoefficients
{
  double pdf_scatter_fwd{ 1. }; // Moving forward. Pdf for scatter event to occur at the end of the given segment.
  double pdf_scatter_bwd{ 1. }; // Backward. For scatter event at the segment start, moving from back to start.
  double transmittance{ 1. }; // Corresponding transmittances.
};


inline std::tuple<double, double> FwdCoeffs(const VolumePdfCoefficients &c)
{
  return std::make_tuple(c.pdf_scatter_fwd, c.transmittance);
}

inline std::tuple<double, double> BwdCoeffs(const VolumePdfCoefficients &c)
{
  return std::make_tuple(c.pdf_scatter_bwd, c.transmittance);
}


inline void Accumulate(VolumePdfCoefficients &accumulated, const VolumePdfCoefficients &segment_coeff, bool is_first, bool is_last)
{
  accumulated.pdf_scatter_fwd *= is_last ? segment_coeff.pdf_scatter_fwd : segment_coeff.transmittance;
  accumulated.pdf_scatter_bwd *= is_first ? segment_coeff.pdf_scatter_bwd : segment_coeff.transmittance;
  accumulated.transmittance *= segment_coeff.transmittance;
}



using PhaseFunctionPtr = util::MemoryArena::unique_ptr<PhaseFunctions::PhaseFunction>;

class OnTheLine
{
  public:

    virtual void AdvancePointer(double dt) = 0;
    virtual MediaCoefficients GetCoefficients() const = 0;
    // Needed here because the phase function in the atmospheric model depends on the local composition!
    virtual PhaseFunctionPtr AllocatePhaseFunction(util::MemoryArena &arena) = 0;
};

using OnTheLinePtr = util::MemoryArena::unique_ptr<OnTheLine>;
using MemoryArena = util::MemoryArena;


/* Good reference for path tracing with emissive volumes:
    Raab et. al (2008) "Unbiased global illumination with participating media" */
class Medium
{
public:
  // Represents both, scattering and emission/absorption events.
  struct InteractionSample
  {
    double t;
    // Following PBRT pg 893, the returned weight is either
    // weight_surf = T(t_intersect)/p_surf if t > t_intersect, or
    // weight_med =  T(t) / p(t)
    // where t_intersect refers to the end of the supplied segment.
    Spectral3 weight;
    Spectral3 sigma_s;
  };
  using PhaseSample = ScatterSample;
  using MaterialCoefficients = materials::MediaCoefficients;

  struct VolumeSample
  {
    Double3 pos;
  };
 
  const bool is_emissive;
  const bool is_scattering;
  const bool is_monochromatic;
  const bool is_homogeneous;
  const int priority;
  Medium(int _priority, MediumFlags flags) : 
    is_emissive(flags & IS_EMISSIVE),
    is_scattering(flags & IS_SCATTERING),
    is_monochromatic(flags & IS_MONOCHROMATIC),
    is_homogeneous(flags & IS_HOMOGENEOUS), priority(_priority) {}
  virtual ~Medium() {}

  virtual OnTheLinePtr GetOnTheLine(const RaySegment &segment, const PathContext &context, MemoryArena &arena) const;
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, const Spectral3 &initial_weights, Sampler &sampler, const PathContext &context) const = 0;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const = 0;
  virtual void ConstructShortBeamTransmittance(const RaySegment &segment, Sampler &sampler, const PathContext &context, PiecewiseConstantTransmittance &pct) const = 0;
  virtual VolumePdfCoefficients ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const = 0; // Can be approximate. Deterministic.
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const = 0;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const = 0;  
  virtual VolumeSample SampleEmissionPosition(Sampler &sampler, const PathContext &context) const;
  virtual Spectral3 EvaluateEmission(const Double3 &pos, const PathContext &context, double *pos_pdf) const;
  virtual MaterialCoefficients EvaluateCoeffs(const Double3 &pos, const PathContext &context) const = 0;
};

#if 0
/* An emissive ball. The geometry of the ball is specified here because I don't have a general
 * boundary representation which would support generating photons within it's volume. So this medium
 * class needs all the details. Using this info, it does the geometric calculations to sample the photon positions.
 */
class EmissiveDemoMedium : public Medium
{
  double sigma_s, sigma_a, sigma_ext;
  SpectralN spectrum;
  PhaseFunctions::Uniform phasefunction;
  Double3 pos;
  double radius;
  double one_over_its_volume;
public:
  EmissiveDemoMedium(double sigma_s_, double sigma_a, double extra_emission_multiplier_, double temperature, const Double3 &pos_, double radius_, int priority);
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, const Spectral3 &initial_weights, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual void ConstructShortBeamTransmittance(const RaySegment &segment, Sampler &sampler, const PathContext &context, PiecewiseConstantTransmittance &pct) const override;
  virtual VolumePdfCoefficients ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
  virtual VolumeSample SampleEmissionPosition(Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluateEmission(const Double3 &pos, const PathContext &context, double *pos_pdf) const override;
  virtual MaterialCoefficients EvaluateCoeffs(const Double3 &pos, const PathContext &context) const override;
};
#endif

class VacuumMedium : public Medium
{
public:
  VacuumMedium(int priority = -1) : Medium(priority, IS_MONOCHROMATIC|IS_HOMOGENEOUS) {}
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, const Spectral3 &initial_weights, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual void ConstructShortBeamTransmittance(const RaySegment &segment, Sampler &sampler, const PathContext &context, PiecewiseConstantTransmittance &pct) const override;
  virtual VolumePdfCoefficients ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
  virtual MaterialCoefficients EvaluateCoeffs(const Double3 &pos, const PathContext &context) const override;
};


class HomogeneousMedium : public Medium
{
  SpectralN sigma_s, sigma_a, sigma_ext;
  bool is_scattering;
  Spectral3 EvaluateTransmissionHomogeneous(double x, const Spectral3 &sigma_ext) const;
public:
  std::unique_ptr<PhaseFunctions::PhaseFunction> phasefunction; // filled by parser
public:
  HomogeneousMedium(const SpectralN &_sigma_s, const SpectralN &_sigma_a, int _priority); 
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, const Spectral3 &initial_weights, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual void ConstructShortBeamTransmittance(const RaySegment &segment, Sampler &sampler, const PathContext &context, PiecewiseConstantTransmittance &pct) const override;
  virtual VolumePdfCoefficients ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
  virtual MaterialCoefficients EvaluateCoeffs(const Double3 &pos, const PathContext &context) const override;
};


class MonochromaticHomogeneousMedium : public Medium
{
  double sigma_s, sigma_ext;
public:
  std::unique_ptr<PhaseFunctions::PhaseFunction> phasefunction;
public:
  MonochromaticHomogeneousMedium(double _sigma_s, double _sigma_a, int _priority); 
  virtual InteractionSample SampleInteractionPoint(const RaySegment &segment, const Spectral3 &initial_weights, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluateTransmission(const RaySegment &segment, Sampler &sampler, const PathContext &context) const override;
  virtual void ConstructShortBeamTransmittance(const RaySegment &segment, Sampler &sampler, const PathContext &context, PiecewiseConstantTransmittance &pct) const override;
  virtual VolumePdfCoefficients ComputeVolumePdfCoefficients(const RaySegment &segment, const PathContext &context) const override;
  virtual PhaseSample SamplePhaseFunction(const Double3 &incident_dir, const Double3 &pos, Sampler &sampler, const PathContext &context) const override;
  virtual Spectral3 EvaluatePhaseFunction(const Double3 &indcident_dir, const Double3 &pos, const Double3 &out_direction, const PathContext &context, double *pdf) const override;
  virtual MaterialCoefficients EvaluateCoeffs(const Double3 &pos, const PathContext &context) const override;
};


}  // materials


using materials::Medium;
using materials::VolumePdfCoefficients;
using materials::FwdCoeffs;
using materials::BwdCoeffs;
using materials::Accumulate;

#endif
