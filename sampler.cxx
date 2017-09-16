#include "sampler.hxx"

namespace SampleTrafo
{
Double3 ToUniformSphere(Double2 r)
{
  double z = 1. - 2.*r[1];
  double s = sqrt(r[1]*(1.-r[1]));
  double omega = 2.*Pi*r[0];
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  double x = 2.*cs*s;
  double y = 2.*sn*s;
  return Double3(x,y,z);
}

Double3 ToUniformHemisphere(Double2 r)
{
  Double3 v = ToUniformSphere(r);
  v[2] = v[2]>=0. ? v[2] : -v[2];
  return v;
}

Double3 ToCosHemisphere(Double2 r)
{
  double rho = std::sqrt(1.-r[0]);
  double z   = std::sqrt(r[0]);
  double omega = 2.*Pi*r[1];
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  return Double3{cs*rho, sn*rho, z};
}

}


Sampler::Sampler()
  : random_engine(),
    uniform(0., 1.)
{

}


void Sampler::Uniform01(double* dest, int count)
{
  for (int i=0; i<count; ++i)
    dest[i] = uniform(random_engine);
}


int Sampler::UniformInt(int a, int b_inclusive)
{
  return std::uniform_int_distribution<int>(a, b_inclusive)(random_engine);
}



// Double3 Sampler::UniformSphere()
// {
//   double r[2];
//   Uniform01(r, 2);
//   return TransformToUniformSphere(r[0], r[1]);
// }
// 
// 
// Double3 Sampler::UniformHemisphere()
// {
//   Double3 v = UniformSphere();
//   v[2] = v[2]>=0. ? v[2] : -v[2];
//   return v;
// }
