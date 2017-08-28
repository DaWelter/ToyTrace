#pragma once

#include "vec3f.hxx"
#include <cmath>
#include <random>


class Sampler
{
  std::mt19937 random_engine;
  std::uniform_real_distribution<double> uniform;
public:
  Sampler();
  double Uniform01();
  Double3 UniformSphere();
  Double3 UniformHemisphere();
  int UniformInt(int a, int b);
};