#pragma once

#include "vec3f.hxx"

namespace Color
{

using Scalar = double; // TODO: Use float?

constexpr Scalar operator"" _sp (long double d) { return Scalar(d); }
constexpr Scalar operator"" _sp (unsigned long long int d) { return Scalar(d); }

static constexpr Scalar lambda_min = 380;
static constexpr Scalar lambda_max = 720;
static constexpr int NBINS = 10;

using Spectral3 = Eigen::Array<Scalar, 3, 1>;
using SpectralN = Eigen::Array<Scalar, NBINS, 1>;

// Applies gamma correction. See https://en.wikipedia.org/wiki/SRGB
inline Scalar SRGBToLinear(Scalar x)
{
  return (x <= 0.04045_sp) ? 
         (x/12.92_sp) : 
         (std::pow((x+0.055_sp)/1.055_sp, 2.4_sp));
}


// Applies gamma correction. See https://en.wikipedia.org/wiki/SRGB
inline Scalar LinearToSRGB(Scalar x)
{    
  return   (x <= 0.0031308_sp) ? 
           (12.92_sp*x) : 
           (1.055_sp*std::pow(x, 1.0_sp/2.4_sp) - 0.055_sp);
}


Scalar GetWavelength(int bin);
Scalar RGBToSpectrum(int bin, const Spectral3 &rgb);
SpectralN RGBToSpectrum(const Spectral3 &rgb);
Spectral3 SpectrumToRGB(const SpectralN &val);

/*
  the spectral intensity emitted per area and steradian
  [ W / m^2 / sr / m / wavelength]
  http://csep10.phys.utk.edu/astr162/lect/light/radiation.html
  https://upload.wikimedia.org/wikipedia/commons/1/19/Black_body.svg
  I normalized the result by the using the sum of the vector entries
*/
SpectralN MaxwellBoltzmanDistribution(double temp);

}


using Spectral3 = Color::Spectral3;

template<>
constexpr int static_size<Spectral3>()
{
  return Spectral3::RowsAtCompileTime;
}
