#include "sampler.hxx"

inline Double3 TransformToUniformSphere(double r1, double r2)
{
  double z = 1. - 2.*r2;
  double s = sqrt(r2*(1.-r2));
  double omega = 2.*Pi*r1;
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  double x = 2.*cs*s;
  double y = 2.*sn*s;
  return Double3(x,y,z);
}


Sampler::Sampler()
  : random_engine(),
    uniform(0., 1.)
{
}

double Sampler::Uniform01()
{
  return uniform(random_engine);
}


Double3 Sampler::UniformSphere()
{
  return TransformToUniformSphere(Uniform01(), Uniform01());
}


int Sampler::UniformInt(int a, int b)
{
  return std::uniform_int_distribution<int>(a, b)(random_engine);
}


Double3 Sampler::UniformHemisphere()
{
  Double3 v = UniformSphere();
  v[2] = v[2]>=0. ? v[2] : -v[2];
  return v;
}