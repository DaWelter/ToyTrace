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

  PositionSample TakePositionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    PositionSample s {
      pos,
      Take(col, context.lambda_idx),
      1.
    };
    SetPmfFlag(s);
    return s;
  }
  
  DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const override
  {
    DirectionalSample s{
      SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare()),
      Spectral3{1._sp},
      1./(UnitSphereSurfaceArea)
    };
    return s;
  }
  
  Spectral3 EvaluatePositionComponent(const Double3 &pos, const LightPathContext &context, double *pdf) const override
  {
    assert (Length(pos - this->pos) <= Epsilon);
    if (pdf) *pdf = 0.;
    return Spectral3{0.};
  }
  
  Spectral3 EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    assert (Length(pos - this->pos) <= Epsilon);
    if (pdf) *pdf = 1./(UnitSphereSurfaceArea);
    return Spectral3{1.};
  }
};


class UniformAreaLight : public AreaEmitter
{
  SpectralN spectrum;
public:
  UniformAreaLight(const SpectralN &_spectrum) : spectrum(_spectrum) 
  {
    spectrum *= (1./UnitHalfSphereSurfaceArea); // Convert from area power density to exitant radiance.
  }
  
  virtual AreaSample TakeAreaSample(const Primitive& primitive, Sampler &sampler, const LightPathContext &context) const override
  {
    AreaSample smpl;
    smpl.coordinates = primitive.SampleUniformPosition(sampler);
    smpl.pdf_or_pmf = 1./primitive.Area();
    smpl.value = Nothing{};
    return smpl;
  }
  
  inline Spectral3 Evaluate(const PosSampleCoordinates &area, const Double3 &dir_out, const LightPathContext &context, double *pdf_pos, double *pdf_dir) const override
  {
    const auto* primitive = area.primitive;
    if (pdf_pos)
      *pdf_pos = 1./primitive->Area();
    if (pdf_dir)
      *pdf_dir = 1./UnitHalfSphereSurfaceArea;
    // Cos of angle between exitant dir and normal is dealt with elsewhere.
    return Take(spectrum, context.lambda_idx);
  }
};


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
  
  DirectionalSample TakeDirectionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    DirectionalSample s {
      dir_out,
      Take(col, context.lambda_idx),
      1.
    };
    SetPmfFlag(s);
    return s;
  }
  
  Spectral3 Evaluate(const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    assert (Length(dir_out - this->dir_out) <= Epsilon);
    if (pdf) *pdf = 0.;
    return Spectral3{0.};
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
  
  DirectionalSample TakeDirectionSample(Sampler &sampler, const LightPathContext &context) const override
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
  
  Spectral3 Evaluate(const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    if (pdf) *pdf = 1./UnitHalfSphereSurfaceArea;
    // Similar rationale as above: light comes from the top hemisphere if
    // the direction vector points down.
    auto above = Dot(dir_out, down_dir);
    return above > 0 ? Take(col, context.lambda_idx) : Spectral3{0.};
  }
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
  
  DirectionalSample TakeDirectionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    auto dir_out = SampleTrafo::ToUniformSphereSection(cos_opening_angle, sampler.UniformUnitSquare());
    DirectionalSample s {
      frame * dir_out,
      Take(emission_spectrum, context.lambda_idx),
      pdf_val };
    return s;
  }

  Spectral3 Evaluate(const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    ASSERT_NORMALIZED(dir_out);
    Double3 dir_out_center = frame.col(2);
    bool is_in_cone = Dot(dir_out, dir_out_center) > cos_opening_angle;
    if (pdf) 
      *pdf = is_in_cone ? pdf_val : 0.;
    return is_in_cone ? Take(emission_spectrum, context.lambda_idx) : Spectral3{0.};
  }
};

} // namespace