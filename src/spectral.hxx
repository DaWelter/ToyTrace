#pragma once

#include "very_strong_typedef.hxx"
#include "vec3f.hxx"

namespace Color
{

using Scalar = double; // TODO: Use float?
struct tag_RGBScalar {};
using RGBScalar = very_strong_typedef<Scalar, tag_RGBScalar>;

//https://codereview.stackexchange.com/questions/49502/user-defined-string-literals-and-namespace-use
namespace Literals
{
constexpr Scalar operator"" _sp (long double d) { return Scalar(d); }
constexpr Scalar operator"" _sp (unsigned long long int d) { return Scalar(d); }
constexpr RGBScalar operator"" _rgb(long double d) { return RGBScalar(d); }
}

using namespace Literals;

// Left and right bounds of the binned range.
static constexpr Scalar lambda_min = 370;
static constexpr Scalar lambda_max = 730;
// And the number of bins.
static constexpr int NBINS = 36;

// TODO: Use strong typedef to differentiate between RGB values and spectral triplel in a type safe way.
// http://www.boost.org/doc/libs/1_61_0/libs/serialization/doc/strong_typedef.html
using Spectral3    = Eigen::Array<Scalar, 3, 1>;
using SpectralN    = Eigen::Array<Scalar, NBINS, 1>;
using RGB          = Eigen::Array<RGBScalar, 3, 1>;  // Alias of Spectral3 which is not ideal because they can be mixed.
using Wavelengths3 = Eigen::Array<Scalar, 3, 1>; // The actual wavelengths
using Spectral3f   = Eigen::Array<float, 3, 1>;


inline auto GetWavelengths()
{
  static const Scalar wavelengths[NBINS] = {
    375.00000, 385.00000, 395.00000, 405.00000, 415.00000, 425.00000, 435.00000, 445.00000, 455.00000, 465.00000, 475.00000, 485.00000, 495.00000, 505.00000, 515.00000, 525.00000, 535.00000, 545.00000, 555.00000, 565.00000, 575.00000, 585.00000, 595.00000, 605.00000, 615.00000, 625.00000, 635.00000, 645.00000, 655.00000, 665.00000, 675.00000, 685.00000, 695.00000, 705.00000, 715.00000, 725.00000
  };
  return Eigen::Map<const Eigen::Array<Scalar, NBINS, 1>, Eigen::Unaligned>(wavelengths, NBINS);
}


inline Scalar GetWavelength(int bin)
{
  return GetWavelengths()[bin];
}

inline std::pair<Scalar,Scalar> GetWavelengthBinBounds(int bin) 
{
  const auto center = GetWavelength(bin);
  static constexpr Scalar half_width = 0.5*(lambda_max-lambda_min)/NBINS;  // Evaluated at compile time!(?)
  return std::make_pair(center-half_width, center+half_width);
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

}

using RGB = Color::RGB;
using Spectral3 = Color::Spectral3;
using Spectral3f = Color::Spectral3f;
using SpectralN = Color::SpectralN;
using RGBScalar = Color::RGBScalar;
using Wavelengths3 = Color::Wavelengths3;

using namespace Color::Literals;

template<>
constexpr int static_size<Spectral3>()
{
  return Spectral3::RowsAtCompileTime;
}



namespace Eigen 
{
template<> 
struct NumTraits<Color::RGBScalar>  : GenericNumTraits<Color::RGBScalar> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef Color::RGBScalar Real;
  typedef Color::RGBScalar NonInteger;
  typedef Color::RGBScalar Nested;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};
} // namespace Eigen


namespace Color
{

extern const RGBScalar srgb_to_rgb[9];
extern const RGBScalar rgb_to_srgb[9];  

inline auto SRGBToRGBMatrix()
{
  return Eigen::Map<const Matrix33<RGBScalar>>{srgb_to_rgb};
}

inline auto RGBToSRGBMatrix()
{
  return Eigen::Map<const Matrix33<RGBScalar>>{rgb_to_srgb};
}

inline RGB SRGBToRGB(const RGB &x)
{
  return (SRGBToRGBMatrix()*x.matrix()).array();
}



// Applies gamma correction. See https://en.wikipedia.org/wiki/SRGB
inline RGBScalar SRGBToLinear(RGBScalar x)
{
  return (x <= 0.04045_rgb) ? 
         (x/12.92_rgb) : 
         (pow((x+0.055_rgb)/1.055_rgb, 2.4_rgb));
}


// Applies gamma correction. See https://en.wikipedia.org/wiki/SRGB
inline RGBScalar LinearToSRGB(RGBScalar x)
{    
  return   (x <= 0.0031308_rgb) ? 
           (12.92_rgb*x) : 
           (1.055_rgb*pow(x, 1.0_rgb/2.4_rgb) - 0.055_rgb);
}

} // namespace Color
