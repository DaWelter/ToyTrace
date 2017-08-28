#pragma once
#include "vec3f.hxx"
#include "ray.hxx"
#include "sampler.hxx"

namespace RadianceOrImportance
{
  
struct Sample
{
  Double3 sample_pos;
  Double3 value;
  double pdf_of_pos; // conditioned on 'org' (see below) is okay.
};

struct DirectionalSample : public Sample
{
  Double3 emission_dir;
  double pdf_of_dir_given_pos;
};


class PathEndPoint
{
public:
  virtual ~PathEndPoint() {}
  virtual Sample TakePositionSampleTo(const Double3 &org, Sampler &sampler) const = 0;
  virtual DirectionalSample TakeDirectionalSample(Sampler &sampler) const = 0;
};


}