#pragma once

#include "shader_util.hxx"
#include "primitive.hxx"

namespace RadianceOrImportance
{

struct LightPathContext
{
  explicit LightPathContext(const Index3 &_lambda_idx) :
    lambda_idx(_lambda_idx)
  {}
  Index3 lambda_idx;
};

struct TagPositionSample {};
struct TagDirectionSample {};
using PositionSample = Sample<Double3, Spectral3, TagPositionSample>;
using DirectionalSample = Sample<Double3, Spectral3, TagDirectionSample>;


class Emitter
{
public:
  virtual ~Emitter() {}
};

class EnvironmentalRadianceField : public Emitter
{
public:
  virtual DirectionalSample TakeDirectionSample(Sampler &sampler, const LightPathContext &context) const = 0;
  virtual Spectral3 Evaluate(const Double3 &emission_dir, const LightPathContext &context) const = 0;
  virtual double EvaluatePdf(const Double3 &emission_dir, const LightPathContext &context) const = 0;
};


using PosSampleCoordinates = HitId;
struct TagAreaSample {};
struct Nothing {};
using AreaSample = Sample<PosSampleCoordinates, Nothing, TagAreaSample>;


class PointEmitter : public Emitter
{
public:
  virtual Double3 Position() const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual Spectral3 Evaluate(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf_direction) const = 0;
};


class AreaEmitter : public Emitter
{
public:
  virtual AreaSample TakeAreaSample(const Primitive& primitive, Sampler &sampler, const LightPathContext &context) const = 0; 
  virtual DirectionalSample TakeDirectionSampleFrom(const PosSampleCoordinates &area, Sampler &sampler, const LightPathContext &context) const
  {
    assert(!"Not Implemented");
    return DirectionalSample{};
  }
  virtual Spectral3 Evaluate(const PosSampleCoordinates &area, const Double3 &dir_out, const LightPathContext &context, double *pdf_direction) const = 0;
  virtual double EvaluatePdf(const PosSampleCoordinates &area, const LightPathContext &context) const = 0;
};


class PointEmitterArray : public Emitter
{
public:
  const int num_units;
  struct Response
  {
    int unit_index = { -1 };
    Spectral3 weight = Spectral3{ 0. };
  };
  PointEmitterArray(int _num_units) : num_units(_num_units) {}
  // Position sample is needed for physical camera.
  virtual PositionSample TakePositionSample(int unit_index, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(int unit_index, const Double3 &pos, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual Response Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, const LightPathContext &context, double *pdf_direction) const = 0;
};


}
