#pragma once
#include "vec3f.hxx"
#include "spectral.hxx"
#include "ray.hxx"
#include "sampler.hxx"
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
  virtual Spectral3 Evaluate(const Double3 &emission_dir, const LightPathContext &context, double *pdf_dir) const = 0;  
};


class PointEmitter : public Emitter
{
public:
  virtual PositionSample TakePositionSample(Sampler &sampler, const LightPathContext &context) const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const = 0;
  //virtual Spectral3 Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, const LightPathContext &context, double *pdf_pos, double *pdf_dir) const = 0;
  virtual Spectral3 EvaluatePositionComponent(const Double3 &pos, const LightPathContext &context, double *pdf) const = 0;
  virtual Spectral3 EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf) const = 0;
};


struct AreaSampleCoordinates
{
  Double3 pos;
  Double3 normal;
  HitId hit;
};

inline AreaSampleCoordinates MakeAreaSampleCoordinatesFrom(const HitId &hit)
{
  AreaSampleCoordinates area;
  area.hit = hit;
  Double3 dummy1;
  hit.primitive->GetLocalGeometry(hit, area.pos, area.normal, dummy1);
  return area;
}

struct TagAreaSample {};
using AreaSample = Sample<AreaSampleCoordinates, Spectral3, TagAreaSample>;


class AreaEmitter : public Emitter
{
public:
  virtual AreaSample TakeAreaSample(const Primitive& primitive, Sampler &sampler, const LightPathContext &context) const = 0; 
  virtual DirectionalSample TakeDirectionSampleFrom(const AreaSampleCoordinates &area, Sampler &sampler, const LightPathContext &context) const
  {
    assert(!"Not Implemented");
    return DirectionalSample{};
  }
  virtual Spectral3 Evaluate(const AreaSampleCoordinates &area, const Double3 &dir_out, const LightPathContext &context, double *pdf_pos, double *pdf_dir) const = 0;
};


class PointEmitterArray : public Emitter
{
public:
  const int num_units;
  struct Response
  {
    int unit_index;
    Spectral3 value;
    double pdf_pos;
    double pdf_dir;
  };
  PointEmitterArray(int _num_units) : num_units(_num_units) {}
  virtual PositionSample TakePositionSample(int unit_index, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(int unit_index, const Double3 &pos, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual void Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, std::vector<Response> &responses, const LightPathContext &context) const = 0;
};


}
