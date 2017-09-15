#pragma once
#include "vec3f.hxx"
#include "ray.hxx"
#include "sampler.hxx"

namespace RadianceOrImportance
{

struct Sample
{
  // If "is_direction" is true then pos represents a "position" on the unit sphere ...
  Double3 pos;
  double pdf_of_pos;
  Spectral measurement_contribution;
  bool is_direction;
};

struct DirectionalSample
{
  Ray ray_out;
  double pdf;
  Spectral measurement_contribution;
};

class EmitterSensor
{
public:
  virtual ~EmitterSensor() {}
  virtual Sample TakePositionSample(Sampler &sampler) const = 0;
  virtual DirectionalSample TakeDirectionSampleFrom(const Double3 &pos, Sampler &sampler) const = 0;
  virtual Spectral EvaluatePositionComponent(const Double3 &pos, double *pdf) const = 0;
  virtual Spectral EvaluateDirectionComponent(const Double3 &pos, const Double3 &dir_out, double *pdf) const = 0;
};


}