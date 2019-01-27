#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "spectral.hxx"

class Spectral3ImageBuffer
{
  int count;
  long splat_count;
  ToyVector<RGB>  accumulator;
  ToyVector<RGB>  light_accum;
public:
  const int xres, yres;
  
  Spectral3ImageBuffer(int _xres, int _yres)
    : count{0}, splat_count{0}, xres(_xres), yres(_yres)
  {
    int sz = _xres * _yres;
    assert(sz > 0);
    accumulator.resize(sz, RGB::Zero());
    light_accum.resize(sz, RGB::Zero());
  }
  
  void AddSampleCount(int samples_per_pixel)
  {
    count += samples_per_pixel;
  }
  
  void Splat(int pixel_index, const RGB &value)
  {
    light_accum[pixel_index] += value;
  }
  
  void Insert(int pixel_index, const RGB &value)
  {
    accumulator[pixel_index] += value; 
    // Splat count increased here because formally each sampled pixel creates a 
    // eye connection which makes a contribution to all of the pixels. At least 
    // formally. In reality it depends on the pixel filter support.
    // So, after n-pixels processed, I have n samples for *each* pixel in the 
    // light image. Hence a weight of 1/n. See below.
    ++splat_count;
  }
  
  void ToImage(Image &dest, int ystart, int yend) const
  {
    assert (ystart >= 0 && yend>= ystart && yend <= dest.height());
    Color::RGBScalar splat_weight(splat_count>0 ? double(xres*yres)/(splat_count) : 0.); // Multiply with xres*yres because I divided it out in the path tracer code.
    for (int y=ystart; y<yend; ++y)
      for (int x=0; x<xres; ++x)
      {
        int pixel_index = xres * y + x;
        RGB average = accumulator[pixel_index]/Color::RGBScalar(count);
            average += splat_weight*light_accum[pixel_index];
        Image::uchar rgb[3];
        bool isfinite = average.isFinite().all();
        assert (isfinite);
        average = average.max(0._rgb).min(1._rgb);
        if (isfinite)
        {
          for (int i=0; i<3; ++i)
            rgb[i] = value(Color::LinearToSRGB(average[i])*255.999_rgb);
          dest.set_pixel(x, dest.height() - 1 - y, rgb[0], rgb[1], rgb[2]);
        }
      }
  }
};
