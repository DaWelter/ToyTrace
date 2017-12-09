#pragma once

#include "vec3f.hxx"

namespace Color
{

using Scalar = double; // TODO: Use float?

constexpr Scalar operator"" _sp (long double d) { return Scalar(d); }
constexpr Scalar operator"" _sp (unsigned long long int d) { return Scalar(d); }

// Left and right bounds of the binned range.
static constexpr Scalar lambda_min = 380;
static constexpr Scalar lambda_max = 720;
// And the number of bins.
static constexpr int NBINS = 10;

using Spectral3 = Eigen::Array<Scalar, 3, 1>;
using SpectralN = Eigen::Array<Scalar, NBINS, 1>;
using RGB       = Eigen::Array<Scalar, 3, 1>;  // Alias of Spectral3 which is not ideal because they can be mixed.

inline Scalar GetWavelength(int bin)
{
  static constexpr Scalar wavelengths[NBINS] = 
    {397.0, 431.0, 465.0, 499.0, 533.0, 567.0, 601.0, 635.0, 669.0, 703.0};
    //{380.0, 418, 456, 493, 531, 569, 607, 644, 682, 720.0};
  return wavelengths[bin];
}

// Obtained by comparing wavelengths of bins with the RGB primaries
// displayed in the chromacity diagram on wikipedia https://en.wikipedia.org/wiki/SRGB
inline Index3 LambdaIdxClosestToRGBPrimaries()
{
  return Index3{6, 5, 2};
}

Scalar RGBToSpectrum(int bin, const RGB &rgb);
SpectralN RGBToSpectrum(const RGB &rgb);
RGB SpectrumToRGB(const SpectralN &val);
RGB SpectralSelectionToRGB(const Spectral3 &val, const Index3 &idx);

/*
  The spectral intensity emitted per area and steradian.
  [ W / m^2 / sr / m / wavelength ]
  http://csep10.phys.utk.edu/astr162/lect/light/radiation.html
  https://upload.wikimedia.org/wikipedia/commons/1/19/Black_body.svg
  I normalized the result by dividing by the sum of the vector entries.
*/
SpectralN MaxwellBoltzmanDistribution(double temp);


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

}

using RGB = Color::RGB;
using Spectral3 = Color::Spectral3;
using SpectralN = Color::SpectralN;

template<>
constexpr int static_size<Spectral3>()
{
  return Spectral3::RowsAtCompileTime;
}
