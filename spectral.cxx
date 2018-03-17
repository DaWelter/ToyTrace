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
{ 1.00000, 0.99987, 0.99989, 0.99984, 0.99995, 0.99993, 0.99986, 0.99956, 0.99965, 1.00000, 1.00000, 0.99995, 0.99976, 0.99977, 0.99937, 0.99855, 0.99790, 0.99830, 0.99909, 0.99971, 1.00000, 1.00000, 1.00000, 1.00000, 0.99996, 0.99989, 0.99966, 0.99907, 0.99866, 0.99745, 0.99564, 0.99490, 0.99494, 0.99491, 0.99570, 0.99710 },
{ 0.00129, 0.00146, 0.00157, 0.00152, 0.00125, 0.00112, 0.00052, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.00000, -0.00000, -0.00000, -0.00000, 0.00000, 0.00000, 0.05907, 0.17491, 0.32966, 0.50214, 0.66837, 0.80927, 0.91172, 0.97286, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99954, 0.99964, 1.00000 },
{ 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, -0.00000, 0.00000, 0.00000, 0.05065, 0.16468, 0.32833, 0.51896, 0.70761, 0.86859, 0.97558, 1.00000, 1.00000, 1.00000, 0.96276, 0.84612, 0.67440, 0.47849, 0.29140, 0.13970, 0.04045, 0.00000, 0.00000, -0.00000, -0.00000, -0.00000, 0.00000, 0.00000, -0.00000, -0.00000, -0.00000, -0.00000 },
{ 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.96846, 0.88765, 0.76591, 0.61639, 0.45727, 0.30447, 0.17037, 0.06916, 0.01110, 0.00000, -0.00000, 0.00000, 0.00000, 0.00483, 0.02545, 0.05289, 0.07835, 0.09743, 0.11134, 0.12108, 0.12651, 0.12919, 0.13069, 0.13073, 0.12945, 0.12799, 0.12704, 0.12673 },
{ -0.00000, -0.00000, 0.00000, -0.00000, 0.00000, -0.00000, 0.00000, 0.00000, 0.03114, 0.11068, 0.23198, 0.38047, 0.53907, 0.69309, 0.82802, 0.92979, 0.98890, 1.00000, 1.00000, 1.00000, 1.00000, 0.99599, 0.97595, 0.94898, 0.92188, 0.89955, 0.88309, 0.87163, 0.86469, 0.86108, 0.85946, 0.85968, 0.86100, 0.86240, 0.86428, 0.86634 },
{ 0.99583, 0.99646, 0.99694, 0.99670, 0.99655, 0.99770, 0.99953, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.93973, 0.82359, 0.66910, 0.49793, 0.33149, 0.19051, 0.08851, 0.02665, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00004, 0.00087, 0.00026, 0.00000 },
{ 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.94902, 0.83495, 0.67115, 0.48109, 0.29269, 0.13144, 0.02469, -0.00000, 0.00000, 0.00000, 0.03615, 0.15336, 0.32578, 0.52156, 0.70911, 0.86123, 0.96000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99966, 0.99914 },
};

enum Colors {
  WHITE = 0, RED = 1, GREEN = 2, BLUE = 3, YELLOW = 4, CYAN = 5, MAGENTA = 6
};


static const double mat_spectrum_to_rgb[3][NBINS] = {
{ 0.00004, 0.00008, 0.00029, 0.00084, 0.00188, 0.00253, 0.00105, -0.00236, -0.00725, -0.01389, -0.02140, -0.02733, -0.03332, -0.04245, -0.04924, -0.04459, -0.02769, -0.00275, 0.02961, 0.06815, 0.10937, 0.14705, 0.17361, 0.18104, 0.16778, 0.13631, 0.09948, 0.06662, 0.04054, 0.02235, 0.01195, 0.00608, 0.00296, 0.00150, 0.00076, 0.00038 },
{ -0.00000, -0.00000, -0.00001, -0.00001, 0.00003, 0.00040, 0.00153, 0.00340, 0.00582, 0.00981, 0.01637, 0.02463, 0.03633, 0.05534, 0.08013, 0.10059, 0.11068, 0.11273, 0.10802, 0.09723, 0.08107, 0.06153, 0.04188, 0.02559, 0.01418, 0.00713, 0.00327, 0.00137, 0.00056, 0.00023, 0.00009, 0.00004, 0.00002, 0.00001, 0.00001, 0.00000 },
{ 0.00118, 0.00233, 0.00914, 0.02786, 0.07026, 0.12623, 0.15166, 0.15055, 0.13187, 0.11854, 0.09030, 0.05605, 0.03238, 0.01919, 0.00964, 0.00434, 0.00165, 0.00009, -0.00058, -0.00074, -0.00066, -0.00050, -0.00033, -0.00020, -0.00012, -0.00006, -0.00003, -0.00001, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000 },
};


RGB SpectrumToRGB(int bin, Scalar intensity)
{
  RGB x;
  for (int i=0; i<3; ++i)
    x[i] = RGBScalar(intensity * mat_spectrum_to_rgb[i][bin]);
  return x;
}

#define RESTRICT __restrict__// Because I can!

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