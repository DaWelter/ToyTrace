#pragma once

#include "vec3f.hxx"

namespace Color
{

using Scalar = double;
typedef Eigen::Array<Scalar, 3, 1> Spectral;


// Applies gamma correction. See https://en.wikipedia.org/wiki/SRGB
inline Scalar SRGBToLinear(Scalar x)
{
  using S = Scalar;
  return (x <= S(0.04045)) ? 
         (x/S(12.92)) : 
         (std::pow((x+S(0.055))/S(1.055f), S(2.4)));
}


// Applies gamma correction. See https://en.wikipedia.org/wiki/SRGB
inline Scalar LinearToSRGB(Scalar x)
{    
  using S = Scalar;
  return   (x <= S(0.0031308)) ? 
           (S(12.92)*x) : 
           (S(1.055)*std::pow(x, S(1.0/2.4)) - S(0.055));
}

}


using Spectral = Color::Spectral;

template<>
constexpr int static_size<Spectral>()
{
  return Spectral::RowsAtCompileTime;
}
