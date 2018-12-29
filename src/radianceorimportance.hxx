#pragma once

#include "shader_util.hxx"
#include "primitive.hxx"

class Scene;

namespace RadianceOrImportance
{

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
  virtual DirectionalSample TakeDirectionSample(Sampler &sampler, const PathContext &context) const = 0;
  virtual Spectral3 Evaluate(const Double3 &emission_dir, const PathContext &context) const = 0;
  virtual double EvaluatePdf(const Double3 &emission_dir, const PathContext &context) const = 0;
};


using PosSampleCoordinates = HitId;
struct TagAreaSample {};
struct Nothing {};
using AreaSample = Sample<PosSampleCoordinates, Nothing, TagAreaSample>;


class PointEmitter : public Emitter
{
public:
  virtual Double3 Position() const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const PathContext &context) const = 0;
  virtual Spectral3 Evaluate(const Double3 &pos, const Double3 &dir_out, const PathContext &context, double *pdf_direction) const = 0;
};


class AreaEmitter : public Emitter
{
public:
  virtual AreaSample TakeAreaSample(const PrimRef& primitive, Sampler &sampler, const PathContext &context) const = 0; 
  virtual DirectionalSample TakeDirectionSampleFrom(const PosSampleCoordinates &area, Sampler &sampler, const PathContext &context) const
  {
    assert(!"Not Implemented");
    return DirectionalSample{};
  }
  virtual Spectral3 Evaluate(const PosSampleCoordinates &area, const Double3 &dir_out, const PathContext &context, double *pdf_direction) const = 0;
  virtual double EvaluatePdf(const PosSampleCoordinates &area, const PathContext &context) const = 0;
};


class PointEmitterArray : public Emitter
{
public:
  const int num_units;
  struct Response
  {
    int unit_index = { -1 };
    Spectral3 weight = Spectral3{ 0. };
    operator bool() const { return unit_index>=0; }
  };
  PointEmitterArray(int _num_units) : num_units(_num_units) {}
  // Position sample is needed for physical camera.
  virtual PositionSample TakePositionSample(int unit_index, Sampler &sampler, const PathContext &context) const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(int unit_index, const Double3 &pos, Sampler &sampler, const PathContext &context) const = 0;
  virtual Response Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, const PathContext &context, double *pdf_direction) const = 0;
};


}
