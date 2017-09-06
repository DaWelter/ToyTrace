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
  Double3 measurement_contribution;
  bool is_direction;
};

struct DirectionalSample
{
  Ray ray_out;
  double pdf_of_dir_given_pos;
  Double3 measurement_contribution;
};


class PathEndPoint
{
public:
  virtual ~PathEndPoint() {}
  virtual Sample TakePositionSample(Sampler &sampler) const = 0;
  virtual DirectionalSample TakeDirectionalSampleFrom(const Double3 &pos, Sampler &sampler) const = 0;
  //virtual void EvaluateMeasurementContribution(const Double3 &pos) const = 0;
  //virtual void EvaluateMeasurementContribution(const Double3 &pos, const Double3 &dir_out) const = 0;
};


}