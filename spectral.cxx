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

/*
The data presented here is based on a 10 bin spectral representation from 380nm to
720nm. The bins are all equal size. The data for theXYZ to RGB conversion areRxy =
(.64; .33);Gxy = (.3; .6);Bxy = (.15; .06);Wxy = (.333; .333);WY = 106.8. The
phosphor chromaticities are based on the IUT-R BT.709 standard as discussed by a
W3C recommendation describing a default color space for the Internet[13].
*/
static const double spectra[7][NBINS] = {
 {1.0000, 1.0000, 0.9999, 0.9993, 0.9992, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000},
 {0.9710, 0.9426, 1.0007, 1.0007, 1.0007, 1.0007, 0.1564, 0.0000, 0.0000, 0.0000},
 {1.0000, 1.0000, 0.9685, 0.2229, 0.0000, 0.0458, 0.8369, 1.0000, 1.0000, 0.9959},
 {0.0001, 0.0000, 0.1088, 0.6651, 1.0000, 1.0000, 0.9996, 0.9586, 0.9685, 0.9840},
 {0.1012, 0.0515, 0.0000, 0.0000, 0.0000, 0.0000, 0.8325, 1.0149, 1.0149, 1.0149},
 {0.0000, 0.0000, 0.0273, 0.7937, 1.0000, 0.9418, 0.1719, 0.0000, 0.0000, 0.0025},
 {1.0000, 1.0000, 0.8916, 0.3323, 0.0000, 0.0000, 0.0003, 0.0369, 0.0483, 0.0496}
};

enum Colors {
  WHITE = 0,CYAN = 1,MAGENTA = 2,YELLOW = 3,RED = 4,GREEN = 5,BLUE = 6
};


static const double mat_spectrum_to_rgb[3][NBINS] = {
  {0.00306014357538,0.0463958543401,-0.0192977270864,-0.171471339886,-0.168278091835,0.200744766262,0.631887889885,0.393086222687,0.0764148853588,0.00741322824585},
  {-0.00337971853567,-0.0566423249985,-0.00567607879861,0.22674246694,0.496745871265,0.322923516056,0.0682239078675,-0.0371997329563,-0.0100663816127,-0.00105764374},
  {0.0248663212935,0.4718527533,0.535330818287,0.0935674544332,-0.0469072509475,-0.046685819402,-0.0256623726658,-0.00570804182428,-0.000742966635938,-6.14705584196e-05}
};


double RGBToSpectrum(int bin, const Spectral3 &rgb)
{
  const double red = rgb[0], green = rgb[1], blue = rgb[2];
  double ret = 0.;
  if(red <= green && red <= blue)
  {
    ret += red * spectra[WHITE][bin];
    if(green <= blue) // cyan = green+blue
    {
      ret += (green - red) * spectra[CYAN][bin];
      ret += (blue - green) * spectra[BLUE][bin];
    } else {
      ret += (blue - red) * spectra[CYAN][bin];
      ret += (green - blue) * spectra[GREEN][bin];
    } 
  }
  else if(green <= red && green <= blue)
  {
    ret += green * spectra[WHITE][bin];
    if (red <= blue) // red+blue = magenta
    {
      ret += (red  - green) * spectra[MAGENTA][bin];
      ret += (blue -   red) * spectra[BLUE][bin];
    } else {
      ret += (blue - green) * spectra[MAGENTA][bin];
      ret += (red - blue) * spectra[RED][bin];
    }       
  }
  else // blue < all others
  {
    ret += blue * spectra[WHITE][bin];
    if (red <= green) // red+green = YELLOW
    {
      ret += (red - blue) * spectra[YELLOW][bin];
      ret += (green - red) * spectra[GREEN][bin];
    }
    else
    {
      ret += (green - blue) * spectra[YELLOW][bin];
      ret += (red - green) * spectra[RED][bin];
    }
  }
  return ret;
}

double GetWavelength(int bin)
{
  return (bin+0.5)/(NBINS)*(lambda_max-lambda_min) + lambda_min;
}


Spectral3 SpectrumToRGB(int bin, double intensity)
{
  Spectral3 x;
  for (int i=0; i<3; ++i)
    x[i] = intensity * mat_spectrum_to_rgb[i][bin];
  return x;
}


inline void LinComb(double *dst, double a, const double *sa, double b, const double *sb, double c, const double *sc)
{
  for (int i=0; i<NBINS; ++i)
  {
    dst[i] = a*sa[i] + b*sb[i] + c*sc[i];
  }
}


SpectralN RGBToSpectrum(const Spectral3 &rgb)
{
  const double red = rgb[0], green = rgb[1], blue = rgb[2];
  SpectralN ret;

  if(red <= green && red <= blue)
  {
    //ret += red * spectra[WHITE][bin];
    if(green <= blue) // cyan = green+blue
    {
      LinComb(ret.data(), red, spectra[WHITE], green-red, spectra[CYAN], blue-green, spectra[BLUE]);
      //ret += (green - red) * spectra[CYAN][bin];
      //ret += (blue - green) * spectra[BLUE][bin];
    } else {
      LinComb(ret.data(), red, spectra[WHITE], blue - red, spectra[CYAN], green-blue, spectra[GREEN]);
      //ret += (blue - red) * spectra[CYAN][bin];
      //ret += (green - blue) * spectra[GREEN][bin];
    } 
  }
  else if(green <= red && green <= blue)
  {
    //ret += green * spectra[WHITE][bin];
    if (red <= blue) // red+blue = magenta
    {
      //ret += (red  - green) * spectra[MAGENTA][bin];
      //ret += (blue -   red) * spectra[BLUE][bin];
      LinComb(ret.data(), green, spectra[WHITE], red-green, spectra[MAGENTA], blue - red, spectra[BLUE]);
    } else {
      //ret += (blue - green) * spectra[MAGENTA][bin];
      //ret += (red - blue) * spectra[RED][bin];
      LinComb(ret.data(), green, spectra[WHITE], blue - green, spectra[MAGENTA], red - blue, spectra[RED]);
    }       
  }
  else // blue < all others
  {
    //ret += blue * spectra[WHITE][bin];
    if (red <= green) // red+green = YELLOW
    {
      //ret += (red - blue) * spectra[YELLOW][bin];
      //ret += (green - red) * spectra[GREEN][bin];
      LinComb(ret.data(), blue, spectra[WHITE], red - blue, spectra[YELLOW], green - red, spectra[GREEN]);
    }
    else
    {
      //ret += (green - blue) * spectra[YELLOW][bin];
      //ret += (red - green) * spectra[RED][bin];
      LinComb(ret.data(), blue, spectra[WHITE], green - blue, spectra[YELLOW], red - green, spectra[RED]);
    }
  }
  return ret;
}

Spectral3 SpectrumToRGB(const SpectralN &val)
{
  Spectral3 x;
  for (int i=0; i<3; ++i)
  {
    x[i] = 0.;
    for (int bin=0; bin<NBINS; ++bin)
      x[i] += val[bin] * mat_spectrum_to_rgb[i][bin];
  }
  return x;
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