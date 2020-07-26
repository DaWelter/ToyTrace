#pragma once

#include "vec3f.hxx"
#include "ray.hxx"
#include "sampler.hxx"
#include "radianceorimportance.hxx"
#include "primitive.hxx"

namespace RadianceOrImportance
{
  
  
class PointLight : public PointEmitter
{
  SpectralN col; // Total power distributed uniformely over the unit sphere.
  Double3 pos;
public:
  PointLight(const SpectralN &_col,const Double3 &pos)
    : col{_col},pos(pos)
  {
    col *= 1./UnitSphereSurfaceArea;
  }

  Double3 Position() const override
  {
    return pos;
  }
    
  DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const PathContext &context) const override
  {
    DirectionalSample s{
      SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare()),
      Take(col, context.lambda_idx),
      1./(UnitSphereSurfaceArea)
    };
    return s;
  }
  
  Spectral3 Evaluate(const Double3 &pos, const Double3 &dir_out, const PathContext &context, double *pdf_direction) const override
  {
    assert (Length(pos - this->pos) <= Epsilon);
    if (pdf_direction) 
      *pdf_direction = 1./(UnitSphereSurfaceArea);
    return Take(col, context.lambda_idx);
  }
};


template<class Base>
class AreaUniformMixin : public Base
{
public:
  using Base::Base;
  
  AreaSample TakeAreaSample(const PrimRef &prim_ref, Sampler &sampler, const PathContext &context) const override
  {
    AreaSample smpl;
    smpl.coordinates = prim_ref.geom->SampleUniformPosition(prim_ref.index, sampler);
    smpl.pdf_or_pmf = 1./prim_ref.geom->Area(prim_ref.index);
    smpl.value = Nothing{};
    return smpl;
  }
  
  double EvaluatePdf(const HitId &prim_ref, const PathContext &context) const override
  {
    assert (prim_ref.geom && prim_ref.geom->Area(prim_ref.index) > 0.);
    return 1./prim_ref.geom->Area(prim_ref.index);
  }
};


class UniformAreaLightDirectionalPart : public AreaEmitter
{
  SpectralN spectrum;
  
public:
  UniformAreaLightDirectionalPart(const SpectralN &_spectrum) : spectrum(_spectrum) 
  {
    spectrum *= (1./UnitHalfSphereSurfaceArea); // Convert from area power density to exitant radiance.
  }
  
  DirectionalSample TakeDirectionSampleFrom(const PosSampleCoordinates &area, Sampler &sampler, const PathContext &context) const override
  {
    // TODO: Sample cos term?
    Double3 w_hemi = SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    SurfaceInteraction interaction{area};
    auto m = OrthogonalSystemZAligned(interaction.geometry_normal);
    DirectionalSample s{
      m*w_hemi,
      Take(spectrum, context.lambda_idx),
      1./(UnitHalfSphereSurfaceArea)
    };
    return s;
  }
  
  inline Spectral3 Evaluate(const PosSampleCoordinates &area, const Double3 &dir_out, const PathContext &context, double *pdf_dir) const override
  {
    const SurfaceInteraction interaction{area};
    double visibility = Dot(interaction.geometry_normal, dir_out) > 0. ? 1. : 0.;
    if (pdf_dir)
      *pdf_dir = visibility/UnitHalfSphereSurfaceArea;
    // Cos of angle between exitant dir and normal is dealt with elsewhere.
    return visibility*Take(spectrum, context.lambda_idx);
  }
};


using UniformAreaLight = AreaUniformMixin<UniformAreaLightDirectionalPart>;



class ParallelAreaLightDirectionalPart : public AreaEmitter
{
  SpectralN irradiance;  // (?) The power density w.r.t. area.
 
  
public:
  ParallelAreaLightDirectionalPart(const SpectralN &_irradiance) : irradiance{_irradiance}
  {
  }
  
  DirectionalSample TakeDirectionSampleFrom(const PosSampleCoordinates &area, Sampler &sampler, const PathContext &context) const override
  {
    SurfaceInteraction interaction{area};
    DirectionalSample s{
      interaction.geometry_normal,
      Take(irradiance, context.lambda_idx),
      Pdf::MakeFromDelta(1.)
    };
    return s;
  }
  
  inline Spectral3 Evaluate(const PosSampleCoordinates &area, const Double3 &dir_out, const PathContext &context, double *pdf_dir) const override
  {
    if (pdf_dir)
      *pdf_dir = 0.;
    return Spectral3{0.};
  }
};

using ParallelAreaLight = AreaUniformMixin<ParallelAreaLightDirectionalPart>;



///////////////////////////////////////////////////////////////////
/// Env maps here
///////////////////////////////////////////////////////////////////
class DistantDirectionalLight : public EnvironmentalRadianceField
{
  
  SpectralN col;
  Double3 dir_out;
public:
  DistantDirectionalLight(const SpectralN &_col,const Double3 &_dir_out)
    : col{_col},dir_out(_dir_out)
    {}
  
  DirectionalSample TakeDirectionSample(Sampler &sampler, const PathContext &context) const override
  {
    DirectionalSample s {
      dir_out,
      Take(col, context.lambda_idx),
      Pdf::MakeFromDelta(1.)
    };
    return s;
  }
  
  Spectral3 Evaluate(const Double3 &dir_out, const PathContext &context) const override
  {
    return Spectral3{0.};
  }
  
  double EvaluatePdf(const Double3 &dir_out, const PathContext &context) const override
  {
    return 0;
  }
};


class DistantDomeLight : public EnvironmentalRadianceField
{
  
  SpectralN col;
  Double3 down_dir;
  Eigen::Matrix3d frame;
public:
  DistantDomeLight(const SpectralN &_col, const Double3 &_up_dir)
    : col{_col}, down_dir(-_up_dir)
  {
    frame = OrthogonalSystemZAligned(down_dir);
  }
  
  DirectionalSample TakeDirectionSample(Sampler &sampler, const PathContext &context) const override
  {
    // Generate directions pointing away from the light by
    // sampling the opposite hemisphere!
    auto dir_out = frame * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    double pdf = 1./UnitHalfSphereSurfaceArea;
    DirectionalSample s {
      dir_out,
      Take(col, context.lambda_idx),
      pdf };
    return s;
  }
  
  Spectral3 Evaluate(const Double3 &dir_out, const PathContext &context) const override
  {
    // Similar rationale as above: light comes from the top hemisphere if
    // the direction vector points down.
    auto above = Dot(dir_out, down_dir);
    return above > 0. ? Take(col, context.lambda_idx) : Spectral3{0.};
  }
  
  double EvaluatePdf(const Double3 &dir_out, const PathContext &context) const override
  {
    auto above = Dot(dir_out, down_dir);
    return above > 0. ? 1./UnitHalfSphereSurfaceArea : 0.;
  }
};


class EnvMapLight : public EnvironmentalRadianceField
{
  Eigen::Matrix3f frame;
  const Texture* texture;
  Eigen::VectorXf cmf_rows; // Cummulative density over rows. Marginalized over columns.
  Eigen::Matrix<float,Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cmf_cols; // Conditional cummulative density, given that a particular row was selected.
  std::pair<int,int> MapToImage(const Double3 &dir_out) const;
public:
  EnvMapLight(const Texture* _texture, const Double3 &_up_dir);
  DirectionalSample TakeDirectionSample(Sampler &sampler, const PathContext &context) const override;
  Spectral3 Evaluate(const Double3 &dir_out, const PathContext &context) const override;
  double EvaluatePdf(const Double3 &dir_out, const PathContext &context) const override;
};


// It is super large and super far away so it is best modeled as angular distribution of radiance.
class Sun : public EnvironmentalRadianceField
{
  SpectralN emission_spectrum;
  Eigen::Matrix3d frame;
  // Direction that recieve light from the distant source form a cone in solid angle space.
  // The a positive opening angle is the result of the finite extend of the source.
  double cos_opening_angle;
  double pdf_val;
  
  bool IsInCone(const Double3 &dir_out) const
  {
    Double3 dir_out_center = frame.col(2);
    return Dot(dir_out, dir_out_center) > cos_opening_angle;
  }
  
public:
  Sun(double _total_power, const Double3 &_dir_out, double opening_angle)
  {
    cos_opening_angle = std::cos(Pi/180.*opening_angle);
    pdf_val = 1./(2.*Pi*(1.-cos_opening_angle));
    frame = OrthogonalSystemZAligned(_dir_out);
    // Values should be normalized so that the sum over all bins equals 1.
    emission_spectrum << 0.0182455226337, 0.0163991962465, 0.0190541811123, 0.027969972257, 0.0285963511606, 0.0275706142789, 0.0277031812426, 0.0319734945613, 0.0336952080025, 0.0334350453362, 0.0333240205041, 0.0319734945613, 0.0323827950617, 0.0315857361924, 0.0303628059521, 0.0309195871997, 0.031393514095, 0.0308284474122, 0.0306859379262, 0.0305169150474, 0.0304423461304, 0.029999903889, 0.0296204309553, 0.02907359223, 0.0281870506601, 0.0279368305161, 0.0273452504405, 0.0265896187474, 0.0254197152926, 0.0257411901796, 0.0250319569237, 0.0243724362792, 0.0237725707684, 0.0231097359498, 0.0225562688763, 0.0221850813779;
    emission_spectrum *= _total_power;
    emission_spectrum *= pdf_val; // I need it per solid angle.
  }
  
  DirectionalSample TakeDirectionSample(Sampler &sampler, const PathContext &context) const override
  {
    auto dir_out = SampleTrafo::ToUniformSphereSection(cos_opening_angle, sampler.UniformUnitSquare());
    DirectionalSample s {
      frame * dir_out,
      Take(emission_spectrum, context.lambda_idx),
      pdf_val };
    return s;
  }

  Spectral3 Evaluate(const Double3 &dir_out, const PathContext &context) const override
  {
    ASSERT_NORMALIZED(dir_out);
    bool is_in_cone = IsInCone(dir_out);
    return is_in_cone ? Take(emission_spectrum, context.lambda_idx) : Spectral3{0.};
  }
  
  double EvaluatePdf(const Double3 &dir_out, const PathContext &context) const override
  {
    ASSERT_NORMALIZED(dir_out);
    bool is_in_cone = IsInCone(dir_out);
    return is_in_cone ? pdf_val : 0.;
  }
};




class TotalEnvironmentalRadianceField : public EnvironmentalRadianceField
{
  using IndividualLights = ToyVector<std::unique_ptr<EnvironmentalRadianceField>>;
  const IndividualLights &envlights;
  const EnvironmentalRadianceField& get(int i) const;
  int size() const;
public:
  TotalEnvironmentalRadianceField(const IndividualLights &envlights_);
  DirectionalSample TakeDirectionSample(Sampler &sampler, const PathContext &context) const override;
  Spectral3 Evaluate(const Double3 &emission_dir, const PathContext &context) const override;
  double EvaluatePdf(const Double3 &dir_out, const PathContext &context) const override;
};


} // namespace


namespace Lights
{

namespace Detail
{

// TODO: At least some of the data could be precomputed.
// TODO: For improved precision it would make sense to move the scene center to the origin.
struct EnvLightPointSamplingBeyondScene
{
  double diameter;
  double sufficiently_long_distance_to_go_outside_the_scene_bounds;
  Double3 box_center;

  EnvLightPointSamplingBeyondScene(const Scene &scene);

  Double3 Sample(const Double3 &exitant_direction, Sampler &sampler) const;

  double Pdf(const Double3 &) const
  {
    return 1. / (Pi*0.25*Sqr(diameter));
  }
};


RaySegment SegmentToEnv(const SurfaceInteraction &surface, const Scene &scene, const Double3 &dir);
RaySegment SegmentToEnv(const VolumeInteraction &volume, const Scene &scene, const Double3 &dir);
RaySegment SegmentToPoint(const SurfaceInteraction &surface, const Double3 &pos);
RaySegment SegmentToPoint(const VolumeInteraction &volume, const Double3 &pos);


} // namespace Detail


using EmissionSample = std::tuple<Ray, std::pair<Pdf,Pdf>, Spectral3>;
using LightConnectionSample = std::tuple<RaySegment, Pdf, Spectral3>;

inline static constexpr int IDX_PROB_ENV = 0;
inline static constexpr int IDX_PROB_AREA = 1;
inline static constexpr int IDX_PROB_POINT = 2;
inline static constexpr int IDX_PROB_VOLUME = 3;
inline static constexpr int NUM_LIGHT_TYPES = 4;

struct LightRef
{
  std::uint32_t type : 2;
  std::uint32_t idx : 30;
};

static_assert(sizeof(LightRef) == 4);

inline LightRef MakeLightRef(const Scene &scene, const PrimRef &hitid)
{
  return { (uint32_t)IDX_PROB_AREA , (uint32_t)scene.GetAreaLightIndex(hitid) };
}

inline LightRef MakeLightRef(const Scene &scene, const RadianceOrImportance::TotalEnvironmentalRadianceField &envlight)
{
  return { (uint32_t)IDX_PROB_ENV ,(uint32_t)0 };
}

inline LightRef MakeLightRef(const Scene &scene, const RadianceOrImportance::PointEmitter &pointlight)
{
  return { (uint32_t)IDX_PROB_POINT, (uint32_t)pointlight.scene_index };
}


class Base
{
};


class Env : public Base
{
  const RadianceOrImportance::EnvironmentalRadianceField *env;
public:
  Env(const RadianceOrImportance::EnvironmentalRadianceField &env_) : env{ &env_ } {}

  static constexpr bool IsAngularDistribution() { return true;  }

  const auto & Get() const { return *env; }

  Double3 SurfaceNormal() const { return Double3::Zero(); }

  EmissionSample SampleExitantRay(const Scene &scene, Sampler &sampler, const PathContext &context)
  {
    auto smpl = env->TakeDirectionSample(sampler, context);
    Detail::EnvLightPointSamplingBeyondScene gen{ scene };
    double pdf = gen.Pdf(smpl.coordinates);
    Double3 org = gen.Sample(smpl.coordinates, sampler);
    return { { org, smpl.coordinates }, { smpl.pdf_or_pmf, pdf}, smpl.value };
  }

  LightConnectionSample SampleConnection(const SomeInteraction &from, const Scene &scene, Sampler &sampler, const PathContext &context)
  {
    const auto smpl = env->TakeDirectionSample(sampler, context);
    const auto seg = mpark::visit([&scene, &smpl](auto && ia) -> RaySegment { return Detail::SegmentToEnv(ia, scene, smpl.coordinates); }, from);
    return { seg, smpl.pdf_or_pmf, smpl.value };
  }
};


class Point : public Base
{
  const RadianceOrImportance::PointEmitter *light;
public:
  Point(const RadianceOrImportance::PointEmitter &light_) : light{ &light_ } {}

  static constexpr bool IsAngularDistribution() { return false; }

  const auto & Get() const { return *light; }

  Double3 SurfaceNormal() const { return Double3::Zero(); }

  EmissionSample SampleExitantRay(const Scene &scene, Sampler &sampler, const PathContext &context)
  {
    const Double3 org = light->Position();
    auto smpl = light->TakeDirectionSampleFrom(org, sampler, context);
    return { {org, smpl.coordinates }, { Pdf::MakeFromDelta(1.), smpl.pdf_or_pmf}, smpl.value };
  }

  //LightConnectionSample SampleConnection(const SurfaceInteraction &surfaceFrom, const Scene &scene, Sampler &sampler, const PathContext &context)
  //{
  //  const auto seg = Detail::SegmentToPoint(surfaceFrom, light->Position());
  //  const auto val = light->Evaluate(light->Position(), -seg.ray.dir, context, nullptr);
  //  return { seg, Pdf::MakeFromDelta(1.), val };
  //}

  LightConnectionSample SampleConnection(const SomeInteraction &from, const Scene &scene, Sampler &sampler, const PathContext &context)
  {
    const auto seg = mpark::visit([this](auto && ia) -> RaySegment { return Detail::SegmentToPoint(ia, light->Position()); }, from);
    const auto val = light->Evaluate(light->Position(), -seg.ray.dir, context, nullptr);
    return { seg, Pdf::MakeFromDelta(1.), val };
  }
};


class Area : public Base
{
  PrimRef prim_ref;
  const RadianceOrImportance::AreaEmitter *emitter;
  SurfaceInteraction light_surf;
public:
  Area(const PrimRef &prim_ref_, const Scene &scene)
    : prim_ref{prim_ref_}
  {
    const auto &mat = scene.GetMaterialOf(prim_ref);
    assert(mat.emitter); // Otherwise would would not have been selected.
    emitter = mat.emitter;
  }

  static constexpr bool IsAngularDistribution() { return false; }

  auto Get() const { return prim_ref; }

  Double3 SurfaceNormal() const { return light_surf.geometry_normal; }

  EmissionSample SampleExitantRay(const Scene &scene, Sampler &sampler, const PathContext &context)
  {
    auto area_smpl = emitter->TakeAreaSample(prim_ref, sampler, context);
    auto dir_smpl = emitter->TakeDirectionSampleFrom(area_smpl.coordinates, sampler, context);
    this->light_surf = SurfaceInteraction{ area_smpl.coordinates };
    Ray out_ray{ this->light_surf.pos, dir_smpl.coordinates };
    out_ray.org += AntiSelfIntersectionOffset( this->light_surf, out_ray.dir);
    Spectral3 val = dir_smpl.value * std::abs(Dot(light_surf.smooth_normal, out_ray.dir));
    
    return { out_ray, { area_smpl.pdf_or_pmf, dir_smpl.pdf_or_pmf }, val };
  }

  //LightConnectionSample SampleConnection(const SurfaceInteraction &surfaceFrom, const Scene &scene, Sampler &sampler, const PathContext &context)
  //{
  //  auto area_smpl = emitter->TakeAreaSample(prim_ref, sampler, context);
  //  this->light_surf = SurfaceInteraction{ area_smpl.coordinates };
  //  const auto seg = Detail::SegmentToPoint(surfaceFrom, light_surf.pos);
  //  const auto val = emitter->Evaluate(area_smpl.coordinates, -seg.ray.dir, context, nullptr);
  //  // TODO: multiply val by cos of emitting surface?
  //  return { seg, area_smpl.pdf_or_pmf, val };
  //}

  LightConnectionSample SampleConnection(const SomeInteraction &somewhere, const Scene &scene, Sampler &sampler, const PathContext &context);
};


class Medium : public Base
{
  const ::Medium *medium;
public:
  Medium(const ::Medium &m) :
    medium{ &m } {}

  const auto& Get() const { return *medium; }
};


} // namespace Lights