#pragma once

#include "vec3f.hxx"
#pragma once

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
  ToyVector<double> cmf; // Cumulative (probability) mass function. 1 value per pixel. Row major order.
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
  using IndividualLights = std::vector<std::unique_ptr<EnvironmentalRadianceField>>;
  const IndividualLights &envlights;
  const EnvironmentalRadianceField& get(int i) const;
  int size() const;
public:
  TotalEnvironmentalRadianceField(const IndividualLights &envlights_) : envlights{envlights_} {}
  DirectionalSample TakeDirectionSample(Sampler &sampler, const PathContext &context) const override;
  Spectral3 Evaluate(const Double3 &emission_dir, const PathContext &context) const override;
  double EvaluatePdf(const Double3 &dir_out, const PathContext &context) const override;
};



} // namespace
