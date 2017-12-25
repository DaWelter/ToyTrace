#ifndef LIGHT_HXX
#define LIGHT_HXX

#include "vec3f.hxx"
#include "ray.hxx"
#include "sampler.hxx"
#include "radianceorimportance.hxx"



class Light : public RadianceOrImportance::EmitterSensor
{
public:
  bool is_environmental_radiance_distribution = false;
  using Sample = RadianceOrImportance::Sample;
  using DirectionalSample = RadianceOrImportance::DirectionalSample;
  using LightPathContext = RadianceOrImportance::LightPathContext;
};

// class DirectionalLight : public Light
// {
// 	Double3 dir;
// 	Double3 col;
// public:
// 	DirectionalLight(const Double3 &col,const Double3 &dir) 
// 		: dir(dir) , col(col)
// 	{}
// 
//   LightSample TakePositionSampleTo(const Double3 &org) const override
//   {
//     LightSample s;
//     // We evaluate Le(x0->dir) = col * delta[dir * (org-x0)] by importance sampling, where
//     // x0 is on the light source, delta is the dirac delta distribution.
//     // Taking the position as below importance samples the delta distribution, i.e. we take
//     // the sample deterministically.
//     s.pos_on_light = org - LargeNumber * dir;
//     s.radiance = col;
//     // This should be the delta function. However it cannot be represented by a regular float number.
//     // Fortunately the calling code should only ever use the quotient of radiance and pdf. Thus
//     // the deltas cancel out and I can set this to one.
//     s.pdf_of_pos = 1.;
//     s.is_extremely_far_away = true;
//   }
//   
//   Light::DirectionalSample TakeDirectionalSample() const override
//   {
//     // I need a random position where the ray with direction this->dir has a chance to hit the world. 
//     // TODO: implement
//   }
// };



class PointLight : public Light
{
  SpectralN col; // Total power distributed uniformely over the unit sphere.
  Double3 pos;
public:
  PointLight(const SpectralN &col,const Double3 &pos)
    : col{col},pos(pos)
  {}

  Sample TakePositionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    Sample s {
      pos,
      1.,
      Take(col, context.lambda_idx),
      false };
    return s;
  }
  
  DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const override
  {
    constexpr double one_over_unit_sphere_surface_area = 1./(4.*Pi);
    DirectionalSample s{
      { pos, SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare()) },
      one_over_unit_sphere_surface_area,
      Spectral3{1._sp}
    };
    return s;
  }
  
  Spectral3 EvaluatePositionComponent(const Double3 &pos, const LightPathContext &context, double *pdf) const override
  {
    assert (Length(pos - this->pos) <= Epsilon);
    if (pdf) *pdf = 1.;
    return Take(col, context.lambda_idx);
  }
  
  Spectral3 EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    constexpr double one_over_unit_sphere_surface_area = 1./(4.*Pi);
    if (pdf) *pdf = one_over_unit_sphere_surface_area;
    return Spectral3{1.};
  }
};


class DistantDirectionalLight : public Light
{
  SpectralN col;
  Double3 dir_out;
public:
  DistantDirectionalLight(const SpectralN &_col,const Double3 &_dir_out)
    : col{_col},dir_out(_dir_out)
    {}

  Sample TakePositionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    Sample s {
      dir_out,
      1.,
      Take(col, context.lambda_idx),
      true };
    return s;
  }
  
  DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const override
  {
    assert(false && !"not implemented");
    return DirectionalSample{};
  }
  
  Spectral3 EvaluatePositionComponent(const Double3 &pos, const LightPathContext &context, double *pdf) const override
  {
    assert (Length(pos - this->dir_out) <= Epsilon);
    if (pdf) *pdf = 1.;
    return Take(col, context.lambda_idx);
  }
  
  Spectral3 EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    if (pdf) *pdf = 1.;
    return Spectral3{1.};
  }
};


class DistantDomeLight : public Light
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

  Sample TakePositionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    // Generate directions pointing away from the light by
    // sampling the opposite hemisphere!
    auto dir_out = frame * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    double pdf = 1./UnitHalfSphereSurfaceArea;
    Sample s {
      dir_out,
      pdf,
      Take(col, context.lambda_idx),
      true };
    return s;
  }
  
  DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const override
  {
    assert(false && !"not implemented");
    return DirectionalSample{};
  }
  
  Spectral3 EvaluatePositionComponent(const Double3 &pos, const LightPathContext &context, double *pdf) const override
  {
    if (pdf) *pdf = 1./UnitHalfSphereSurfaceArea;
    // Similar rationale as above: light comes from the top hemisphere if
    // the direction vector (here pos) points down.
    auto above = Dot(pos, down_dir);
    return above > 0 ? Take(col, context.lambda_idx) : Spectral3{0.};
  }
  
  Spectral3 EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    if (pdf) *pdf = 1.;
    return Spectral3{1.};
  }
};


// It is super large and super far away so it is best modeled as angular distribution of radiance.
class Sun : public Light
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
    Light::is_environmental_radiance_distribution = true;
    cos_opening_angle = std::cos(Pi/180.*opening_angle);
    pdf_val = 1./(2.*Pi*(1.-cos_opening_angle));
    frame = OrthogonalSystemZAligned(_dir_out);
    // Values should be normalized so that the sum over all bins equals 1.
    emission_spectrum << 0.0182455226337, 0.0163991962465, 0.0190541811123, 0.027969972257, 0.0285963511606, 0.0275706142789, 0.0277031812426, 0.0319734945613, 0.0336952080025, 0.0334350453362, 0.0333240205041, 0.0319734945613, 0.0323827950617, 0.0315857361924, 0.0303628059521, 0.0309195871997, 0.031393514095, 0.0308284474122, 0.0306859379262, 0.0305169150474, 0.0304423461304, 0.029999903889, 0.0296204309553, 0.02907359223, 0.0281870506601, 0.0279368305161, 0.0273452504405, 0.0265896187474, 0.0254197152926, 0.0257411901796, 0.0250319569237, 0.0243724362792, 0.0237725707684, 0.0231097359498, 0.0225562688763, 0.0221850813779;
    emission_spectrum *= _total_power;
    emission_spectrum *= pdf_val; // I need it per solid angle.
  }

  Sample TakePositionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    auto dir_out = SampleTrafo::ToUniformSphereSection(cos_opening_angle, sampler.UniformUnitSquare());
    Sample s {
      frame * dir_out,
      pdf_val,
      Take(emission_spectrum, context.lambda_idx),
      true };
    return s;
  }
  
  DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const override
  {
    assert(false && !"not implemented");
    return DirectionalSample{};
  }
  
  Spectral3 EvaluatePositionComponent(const Double3 &dir_out_eval, const LightPathContext &context, double *pdf) const override
  {
    ASSERT_NORMALIZED(dir_out_eval);
    Double3 dir_out_center = frame.col(2);
    bool is_in_cone = Dot(dir_out_eval, dir_out_center) > cos_opening_angle;
    if (pdf) 
      *pdf = is_in_cone ? pdf_val : 0.;
    return is_in_cone ? 
           Take(emission_spectrum, context.lambda_idx) :
           Spectral3::Constant(0.);
  }
  
  Spectral3 EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf) const override
  {
    if (pdf) *pdf = 1.;
    return Spectral3{1.};
  }
};

#endif