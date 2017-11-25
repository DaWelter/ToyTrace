#pragma once

#include "vec3f.hxx"
#include <cmath>
#include <random>
#include <array>

namespace SampleTrafo
{
Double3 ToUniformSphere(Double2 r);
Double3 ToUniformHemisphere(Double2 r);
Double3 ToCosHemisphere(Double2 r);
Double3 ToPhongHemisphere(Double2 r, double alpha);
Double3 ToBeckmanHemisphere(Double2 r, double alpha);
}



class Sampler
{
  std::mt19937 random_engine;
  std::uniform_real_distribution<double> uniform;
public: 
  Sampler();
  void Uniform01(double *dest, int count);
  int UniformInt(int a, int b_inclusive);
  
  inline double Uniform01() 
  {
    double r;
    Uniform01(&r, 1);
    return r;
  }
  
  inline Double2 UniformUnitSquare()
  {
    Double2 r;
    Uniform01(r.data(), 2);
    return r;
  }
};


class Stratified2DSamples
{
    int nx, ny;
    inline void FastIncrementCoordinates()
    {
      // Avoid branching. Use faster conditional moves.
      ++current_x;
      current_y = (current_x >= nx) ? current_y+1 : current_y;
      current_y = (current_y >= ny) ? 0 : current_y;
      current_x = (current_x >= nx) ? 0 : current_x;
    }
  public:
    int current_x, current_y;
    
    Stratified2DSamples(int _nx, int _ny)
      : nx(_nx), ny(_ny),
        current_x(0), current_y(0)
      {}
    
    Double2 UniformUnitSquare(Double2 r)
    {
      Double2 ret{
        (current_x + r[0]) / nx,
        (current_y + r[1]) / ny
      };
      FastIncrementCoordinates();
      return ret;
    }
};
