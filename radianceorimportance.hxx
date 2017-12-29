#pragma once
#include "vec3f.hxx"
#include "spectral.hxx"
#include "ray.hxx"
#include "sampler.hxx"

namespace RadianceOrImportance
{

struct LightPathContext
{
  explicit LightPathContext(const Index3 &_lambda_idx) :
    lambda_idx(_lambda_idx)
  {}
  Index3 lambda_idx;
};

struct Sample
{
  // If "is_direction" is true then pos represents a solid angle.
  Double3 pos;
  double pdf;
  Spectral3 measurement_contribution;
  bool is_direction;
};

struct DirectionalSample
{
  Ray ray_out;
  double pdf;
  Spectral3 measurement_contribution;
};

class EmitterSensor
{
public:
  virtual ~EmitterSensor() {}
  virtual Sample TakePositionSample(Sampler &sampler, const LightPathContext &context) const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual Spectral3 EvaluatePositionComponent(const Double3 &pos, const LightPathContext &context, double *pdf) const = 0;
  virtual Spectral3 EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, const LightPathContext &context, double *pdf) const = 0;
};


class EmitterSensorArray
{
public:
  const int num_units;
  struct Response
  {
    Spectral3 measurement_contribution;
    int unit_index;
    double pdf;
  };
  EmitterSensorArray(int _num_units) : num_units(_num_units) {}
  virtual ~EmitterSensorArray() {}
  virtual Sample TakePositionSample(int unit_index, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(int unit_index, const Double3 &pos, Sampler &sampler, const LightPathContext &context) const = 0;
  virtual void Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, std::vector<Response> &responses, const LightPathContext &context) const = 0;
};



}
