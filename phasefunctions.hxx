#pragma once

#include"vec3f.hxx"

class Sampler;


namespace PhaseFunctions
{

struct Sample
{
  Double3 dir;
  Spectral value;
  double pdf;
};
  
  
class PhaseFunction
{
  public:
    virtual Sample SampleDirection(const Double3 &reverse_incident_dir, const Double3 &pos, Sampler &sampler) const = 0;
    virtual Spectral Evaluate(const Double3 &reverse_indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const = 0;
};


class Uniform : public PhaseFunction
{
  public:
    Sample SampleDirection(const Double3 &reverse_incident_dir, const Double3 &pos, Sampler &sampler) const override;
    Spectral Evaluate(const Double3 &reverse_indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const override;
};


/* see http://glossary.ametsoc.org/wiki/Rayleigh_phase_function */
class Rayleigh : public PhaseFunction
{
  public:
    Sample SampleDirection(const Double3 &reverse_incident_dir, const Double3 &pos, Sampler &sampler) const override;
    Spectral Evaluate(const Double3 &reverse_indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const override;
};


// PBRT pg's. 682 and 899.
class HenleyGreenstein : public PhaseFunction
{
    double g;
  public:
    HenleyGreenstein(double _g)
      : g(_g) 
      {}
    Sample SampleDirection(const Double3 &reverse_incident_dir, const Double3 &pos, Sampler &sampler) const override;
    Spectral Evaluate(const Double3 &reverse_indcident_dir, const Double3 &pos, const Double3 &out_direction, double *pdf) const override;
};

}