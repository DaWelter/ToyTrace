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
{ 0.84304, 0.84367, 0.84342, 0.84326, 0.84253, 0.84819, 0.85251, 0.84757, 0.84481, 0.84438, 0.84368, 0.85026, 0.85483, 0.84944, 0.83998, 0.82948, 0.81741, 0.79710, 0.77961, 0.75817, 0.73290, 0.71342, 0.69316, 0.67175, 0.65574, 0.64920, 0.64821, 0.65004, 0.64929, 0.64085, 0.62511, 0.60944, 0.60158, 0.60427, 0.60847, 0.60809 },
{ -0.00000, -0.00000, 0.00431, 0.00861, 0.01150, 0.01665, 0.02312, 0.02760, 0.02421, 0.01077, 0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.00000, 0.00000, 0.00000, -0.00000, 0.02131, 0.11007, 0.23197, 0.37099, 0.50608, 0.61128, 0.68276, 0.73357, 0.76713, 0.78532, 0.79529, 0.80405, 0.81307, 0.81057, 0.80717, 0.81201, 0.81975 },
{ 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, -0.00000, 0.00000, 0.00000, 0.01237, 0.08025, 0.19234, 0.34631, 0.50606, 0.64736, 0.76672, 0.85056, 0.87797, 0.85424, 0.78755, 0.68170, 0.54859, 0.40516, 0.26260, 0.14640, 0.06481, 0.01303, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.00014, -0.00000, 0.00012, 0.00186 },
{ 0.91889, 0.92017, 0.91791, 0.91300, 0.90546, 0.89193, 0.87091, 0.83100, 0.76307, 0.68071, 0.58133, 0.45748, 0.33106, 0.21690, 0.11245, 0.03606, -0.00000, 0.00000, -0.00000, 0.00000, -0.00000, -0.00000, -0.00000, 0.00000, 0.00000, -0.00000, -0.00000, 0.00000, -0.00000, 0.00000, 0.00000, 0.00002, 0.00286, 0.00136, -0.00000, -0.00000 },
{ 0.00801, 0.00911, 0.00993, 0.00425, 0.00000, 0.00000, 0.00000, 0.01583, 0.06629, 0.13531, 0.23380, 0.34059, 0.45294, 0.56513, 0.66431, 0.74644, 0.80062, 0.83008, 0.84059, 0.82642, 0.78607, 0.73613, 0.69097, 0.65275, 0.62509, 0.60927, 0.60036, 0.59359, 0.58624, 0.57844, 0.56919, 0.55298, 0.53211, 0.51452, 0.50318, 0.49754 },
{ 0.78040, 0.77919, 0.77346, 0.76938, 0.77089, 0.77554, 0.78730, 0.80811, 0.83696, 0.86932, 0.89793, 0.92663, 0.94634, 0.94955, 0.94849, 0.92842, 0.87905, 0.80078, 0.70859, 0.60286, 0.48856, 0.37599, 0.27156, 0.18314, 0.11730, 0.07839, 0.06097, 0.05072, 0.04072, 0.03578, 0.03706, 0.03317, 0.02157, 0.01137, 0.00407, 0.00000 },
{ 0.99225, 1.00000, 0.99939, 0.99633, 0.99055, 0.96817, 0.93032, 0.86545, 0.76320, 0.64780, 0.52637, 0.38537, 0.24764, 0.12881, 0.04168, -0.00000, -0.00000, -0.00000, 0.02841, 0.09543, 0.18346, 0.28602, 0.39501, 0.48652, 0.56140, 0.61726, 0.65755, 0.68069, 0.68868, 0.69074, 0.69471, 0.69872, 0.69790, 0.69385, 0.69085, 0.68861 },
};

enum Colors {
  WHITE = 0, RED = 1, GREEN = 2, BLUE = 3, YELLOW = 4, CYAN = 5, MAGENTA = 6
};


static const double mat_spectrum_to_rgb[3][NBINS] = {
{ 0.00028, 0.00055, 0.00215, 0.00646, 0.01587, 0.02695, 0.02860, 0.02191, 0.00934, -0.00582, -0.02598, -0.04506, -0.06346, -0.08776, -0.10927, -0.10888, -0.08350, -0.04145, 0.01577, 0.08571, 0.16206, 0.23343, 0.28566, 0.30381, 0.28467, 0.23270, 0.17046, 0.11442, 0.06973, 0.03847, 0.02057, 0.01046, 0.00510, 0.00259, 0.00131, 0.00065 },
{ -0.00021, -0.00042, -0.00162, -0.00494, -0.01232, -0.02141, -0.02368, -0.01983, -0.01183, -0.00188, 0.01491, 0.03516, 0.05916, 0.09369, 0.13647, 0.16941, 0.18292, 0.18180, 0.16826, 0.14368, 0.10993, 0.07139, 0.03477, 0.00715, -0.00872, -0.01415, -0.01344, -0.01033, -0.00673, -0.00384, -0.00209, -0.00108, -0.00053, -0.00027, -0.00013, -0.00007 },
{ 0.00160, 0.00316, 0.01239, 0.03779, 0.09529, 0.17115, 0.20547, 0.20366, 0.17795, 0.15921, 0.11980, 0.07190, 0.03773, 0.01651, -0.00086, -0.01180, -0.01747, -0.02025, -0.02067, -0.01939, -0.01683, -0.01350, -0.01003, -0.00699, -0.00466, -0.00294, -0.00177, -0.00103, -0.00057, -0.00030, -0.00016, -0.00008, -0.00004, -0.00002, -0.00001, -0.00000 },
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
  for (int i=0; i<Spectral3::RowsAtCompileTime; ++i)
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