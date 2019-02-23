#include "spectral.hxx"

namespace Color
{
/* from "An RGB to Spectrum Conversion for Reflectances" Smits (2000)
 see also
 https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_xy_chromaticity_diagram_and_the_CIE_xyY_color_space
 https://en.wikipedia.org/wiki/List_of_color_spaces_and_their_uses
 https://en.wikipedia.org/wiki/Color_space
 https://en.wikipedia.org/wiki/RGB_color_space
 https://en.wikipedia.org/wiki/SRGB
*/

static const double spectra[7][NBINS] = {
{ 0.99015, 0.99015, 0.99019, 0.99035, 0.99079, 0.99177, 0.99265, 0.99292, 0.99256, 0.99194, 0.99145, 0.99115, 0.99117, 0.99114, 0.99111, 0.99139, 0.99166, 0.99179, 0.99221, 0.99262, 0.99291, 0.99314, 0.99328, 0.99305, 0.99228, 0.99148, 0.99061, 0.98966, 0.98882, 0.98824, 0.98784, 0.98756, 0.98745, 0.98742, 0.98741, 0.98742 },
{ 0.07894, 0.07896, 0.07901, 0.07899, 0.07834, 0.07335, 0.05738, 0.03310, 0.01144, 0.00000, 0.00000, 0.00000, 0.00000, -0.00000, 0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.08029, 0.28157, 0.55071, 0.80631, 0.96563, 0.99212, 0.99225, 0.99172, 0.99119, 0.99070, 0.99042, 0.99019, 0.99010, 0.99005, 0.98999, 0.98996, 0.98996 },
{ -0.00000, -0.00000, 0.00000, -0.00000, -0.00000, 0.00000, -0.00000, 0.00000, -0.00000, 0.12933, 0.37098, 0.54123, 0.75558, 0.85512, 0.95864, 0.98004, 0.99022, 0.98551, 0.98150, 0.86049, 0.62797, 0.33627, 0.15643, 0.00262, 0.00000, 0.02230, 0.14723, 0.19397, 0.19709, 0.21106, 0.21874, 0.21938, 0.22409, 0.22507, 0.22482, 0.22530 },
{ 0.99010, 0.99013, 0.99016, 0.99045, 0.99104, 0.99170, 0.99195, 0.98999, 0.89888, 0.75161, 0.58216, 0.44397, 0.34394, 0.26371, 0.18685, 0.10719, 0.03746, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.00178, 0.01318, 0.02161, 0.02507, 0.02646, 0.02725, 0.02718, 0.02684, 0.02656, 0.02646, 0.02641, 0.02640, 0.02640 },
{ -0.00000, -0.00000, 0.00000, -0.00000, 0.00000, -0.00000, -0.00000, 0.00919, 0.10549, 0.25615, 0.42425, 0.56295, 0.66293, 0.74427, 0.82242, 0.90032, 0.96668, 0.99118, 0.99191, 0.99181, 0.99167, 0.99141, 0.99112, 0.99094, 0.99061, 0.99047, 0.99032, 0.98889, 0.98658, 0.98439, 0.98275, 0.98200, 0.98167, 0.98153, 0.98148, 0.98147 },
{ 0.97029, 0.96996, 0.96873, 0.96766, 0.96084, 0.94531, 0.93921, 0.95239, 0.97858, 0.99026, 0.99064, 0.99205, 0.99225, 0.99265, 0.99473, 0.99503, 0.99462, 0.99490, 0.99268, 0.93095, 0.73666, 0.46266, 0.19337, 0.02553, -0.00000, -0.00000, 0.00000, -0.00000, -0.00000, 0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.00001 },
{ 0.98932, 0.98941, 0.98958, 0.99042, 0.99114, 0.99222, 0.99319, 0.99287, 0.99055, 0.85987, 0.65043, 0.45366, 0.31224, 0.20031, 0.10024, 0.02593, 0.00000, -0.00000, 0.02384, 0.14459, 0.33845, 0.57514, 0.79025, 0.93277, 0.99107, 0.99095, 0.99048, 0.99079, 0.99114, 0.99080, 0.98887, 0.98746, 0.98673, 0.98646, 0.98636, 0.98635 },
};

enum Colors {
  WHITE = 0, RED = 1, GREEN = 2, BLUE = 3, YELLOW = 4, CYAN = 5, MAGENTA = 6
};


static const double mat_spectrum_to_rgb[3][NBINS] = {
{ 0.00015, 0.00029, 0.00112, 0.00337, 0.00825, 0.01392, 0.01456, 0.01075, 0.00378, -0.00469, -0.01567, -0.02578, -0.03540, -0.04812, -0.05878, -0.05685, -0.04093, -0.01575, 0.01789, 0.05861, 0.10271, 0.14358, 0.17307, 0.18256, 0.17029, 0.13885, 0.10156, 0.06811, 0.04148, 0.02288, 0.01223, 0.00622, 0.00303, 0.00154, 0.00078, 0.00039 },
{ -0.00020, -0.00040, -0.00157, -0.00476, -0.01190, -0.02074, -0.02312, -0.01975, -0.01248, -0.00361, 0.01136, 0.02931, 0.05020, 0.07984, 0.11629, 0.14399, 0.15490, 0.15321, 0.14080, 0.11889, 0.08917, 0.05553, 0.02385, 0.00035, -0.01261, -0.01619, -0.01444, -0.01079, -0.00694, -0.00393, -0.00214, -0.00110, -0.00054, -0.00027, -0.00014, -0.00007 },
{ 0.00143, 0.00282, 0.01107, 0.03375, 0.08511, 0.15284, 0.18344, 0.18173, 0.15864, 0.14171, 0.10619, 0.06303, 0.03199, 0.01217, -0.00444, -0.01509, -0.02053, -0.02300, -0.02305, -0.02129, -0.01815, -0.01418, -0.01012, -0.00666, -0.00414, -0.00242, -0.00135, -0.00072, -0.00038, -0.00019, -0.00010, -0.00005, -0.00002, -0.00001, -0.00001, -0.00000 },
};

const RGBScalar srgb_to_rgb[9] = {
   0.78694858234_rgb, 0.0647045767649_rgb, -0.00909263161801_rgb,
   -0.0248551264252_rgb, 1.10299325274_rgb, -0.0177619102543_rgb,
   -0.00104684863467_rgb, -0.0352378150562_rgb, 1.13449582707_rgb
};

const RGBScalar rgb_to_srgb[9] = {
   1.26840200577_rgb, -0.0741202032818_rgb, 0.00900540622489_rgb,
   0.0286156491301_rgb, 0.905405377142_rgb, 0.0144045665207_rgb,
   0.00205922120113_rgb, 0.0280537960894_rgb, 0.88190453316_rgb
};



RGB SpectrumToRGB(int bin, Scalar intensity)
{
  RGB x;
  for (int i=0; i<3; ++i)
    x[i] = RGBScalar(intensity * mat_spectrum_to_rgb[i][bin]);
  return x;
}

#ifdef _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__// Because I can!
#endif

inline void LinComb(double *RESTRICT dst, double a, const double *RESTRICT sa, double b, const double *RESTRICT sb, double c, const double *RESTRICT sc)
{
  for (int i=0; i<NBINS; ++i)
  {
    dst[i] = a*sa[i] + b*sb[i] + c*sc[i];
  }
}


inline void LinComb(double *RESTRICT dst, double a, const double *RESTRICT sa, double b, const double *RESTRICT sb, double c, const double *RESTRICT sc, const int *RESTRICT idx)
{
  for (int i=0; i<3; ++i)
  {
    dst[i] = a*sa[idx[i]] + b*sb[idx[i]] + c*sc[idx[i]];
  }
}


SpectralN RGBToSpectrum(const RGB &rgb)
{
  const double red = value(rgb[0]), green = value(rgb[1]), blue = value(rgb[2]);
  SpectralN ret;

  if(red <= green && red <= blue)
  {
    if(green <= blue) // cyan = green+blue
    {
      LinComb(ret.data(), red, spectra[WHITE], green-red, spectra[CYAN], blue-green, spectra[BLUE]);
    } else {
      LinComb(ret.data(), red, spectra[WHITE], blue - red, spectra[CYAN], green-blue, spectra[GREEN]);
    } 
  }
  else if(green <= red && green <= blue)
  {
    if (red <= blue) // red+blue = magenta
    {
      LinComb(ret.data(), green, spectra[WHITE], red-green, spectra[MAGENTA], blue - red, spectra[BLUE]);
    } else {
      LinComb(ret.data(), green, spectra[WHITE], blue - green, spectra[MAGENTA], red - blue, spectra[RED]);
    }       
  }
  else // blue < all others
  {
    if (red <= green) // red+green = YELLOW
    {
      LinComb(ret.data(), blue, spectra[WHITE], red - blue, spectra[YELLOW], green - red, spectra[GREEN]);
    }
    else
    {
      LinComb(ret.data(), blue, spectra[WHITE], green - blue, spectra[YELLOW], red - green, spectra[RED]);
    }
  }
  return ret;
}

RGB SpectrumToRGB(const SpectralN &val)
{
  RGB x;
  for (int i=0; i<3; ++i)
  {
    x[i] = 0._rgb;
    for (int bin=0; bin<NBINS; ++bin)
      x[i] += RGBScalar(val[bin] * mat_spectrum_to_rgb[i][bin]);
  }
  return x;
}

RGB SpectralSelectionToRGB(const Spectral3 &val, const Index3 &idx)
{
  RGB x;
  for (int i=0; i<3; ++i)
  {
    x[i] = 0._rgb;
    for (int j=0; j<idx.size(); ++j)
      x[i] += RGBScalar(val[j] * mat_spectrum_to_rgb[i][idx[j]]);
  }
  return x;
}

Spectral3 RGBToSpectralSelection(const RGB &rgb, const Index3 &idx)
{
  const double red = value(rgb[0]), green = value(rgb[1]), blue = value(rgb[2]);
  Spectral3 ret;

  if(red <= green && red <= blue)
  {
    if(green <= blue)
    {
      LinComb(ret.data(), red, spectra[WHITE], green-red, spectra[CYAN], blue-green, spectra[BLUE], idx.data());
    } else {
      LinComb(ret.data(), red, spectra[WHITE], blue - red, spectra[CYAN], green-blue, spectra[GREEN], idx.data());
    } 
  }
  else if(green <= red && green <= blue)
  {
    if (red <= blue) // red+blue = magenta
    {
      LinComb(ret.data(), green, spectra[WHITE], red-green, spectra[MAGENTA], blue - red, spectra[BLUE], idx.data());
    } else {
      LinComb(ret.data(), green, spectra[WHITE], blue - green, spectra[MAGENTA], red - blue, spectra[RED], idx.data());
    }       
  }
  else // blue < all others
  {
    if (red <= green)
    {
      LinComb(ret.data(), blue, spectra[WHITE], red - blue, spectra[YELLOW], green - red, spectra[GREEN], idx.data());
    }
    else
    {
      LinComb(ret.data(), blue, spectra[WHITE], green - blue, spectra[YELLOW], red - green, spectra[RED], idx.data());
    }
  }
  return ret;
}

SpectralN MaxwellBoltzmanDistribution(double temp)
{
  SpectralN ret;
  // c = 299 792 458 m / s
  // h = 6.626 069 57 x 10−34 J s
  // kb = 1.380 6488(13) × 10−23 J K−1
  // lambda = v c
  // compute spectral density sigma
  // v = c lambda

  Scalar hc_over_kt = 14.38e6/temp;  // [nm K]
  Scalar hcc = 59.55; // [nm^2 / s J]

  Scalar sum = 0.;
  for (int i=0; i<NBINS; ++i)
  {
    Scalar l = GetWavelength(i);
    Scalar l5 = l*l*l*l*l;
    Scalar t1 = 2*hcc/l5;
    Scalar t2 = 1.0 / (std::exp(hc_over_kt/l) - 1.);
    ret[i] = t1*t2;
    sum += ret[i];
  }
  sum = NBINS/sum;
  for (int i=0; i<NBINS; ++i)
    ret[i] *= sum;
  return ret;
}


}