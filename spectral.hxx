#pragma once

#include "vec3f.hxx"

namespace Color
{

using Scalar = double; // TODO: Use float?

constexpr Scalar operator"" _sp (long double d) { return Scalar(d); }
constexpr Scalar operator"" _sp (unsigned long long int d) { return Scalar(d); }

// Left and right bounds of the binned range.
static constexpr Scalar lambda_min = 370;
static constexpr Scalar lambda_max = 730;
// And the number of bins.
static constexpr int NBINS = 36;

// TODO: Use strong typedef to differentiate between RGB values and spectral triplel in a type safe way.
// http://www.boost.org/doc/libs/1_61_0/libs/serialization/doc/strong_typedef.html
using Spectral3 = Eigen::Array<Scalar, 3, 1>;
using SpectralN = Eigen::Array<Scalar, NBINS, 1>;
using RGB       = Eigen::Array<Scalar, 3, 1>;  // Alias of Spectral3 which is not ideal because they can be mixed.

inline Scalar GetWavelength(int bin)
{
  static constexpr Scalar wavelengths[NBINS] = {
    375.00000, 385.00000, 395.00000, 405.00000, 415.00000, 425.00000, 435.00000, 445.00000, 455.00000, 465.00000, 475.00000, 485.00000, 495.00000, 505.00000, 515.00000, 525.00000, 535.00000, 545.00000, 555.00000, 565.00000, 575.00000, 585.00000, 595.00000, 605.00000, 615.00000, 625.00000, 635.00000, 645.00000, 655.00000, 665.00000, 675.00000, 685.00000, 695.00000, 705.00000, 715.00000, 725.00000
  };
  return wavelengths[bin];
}

// Obtained by comparing wavelengths of bins with the RGB primaries
// displayed in the chromacity diagram on wikipedia https://en.wikipedia.org/wiki/SRGB
inline Index3 LambdaIdxClosestToRGBPrimaries()
{
  return Index3{23, 17, 9};
}

SpectralN RGBToSpectrum(const RGB &rgb);
RGB SpectrumToRGB(const SpectralN &val);
RGB SpectralSelectionToRGB(const Spectral3 &val, const Index3 &idx);
Spectral3 RGBToSpectralSelection(const RGB &rgb, const Index3 &idx);

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
