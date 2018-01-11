#pragma once

#include "vec3f.hxx"
#include "util.hxx"

class Spectral3ImageBuffer
{
  int count;
  std::vector<RGB, boost::alignment::aligned_allocator<RGB, 128> >  accumulator;
public:
  const int xres, yres;
  
  Spectral3ImageBuffer(int _xres, int _yres)
    : xres(_xres), yres(_yres)
  {
    int sz = _xres * _yres;
    assert(sz > 0);
    count = 0;
    accumulator.resize(sz, RGB::Zero());
  }
  
  void AddSampleCount(int more)
  {
    count += more;
  }
  
  void Insert(int pixel_index, const RGB &value)
  {
    assert (pixel_index >= 0 && pixel_index < accumulator.size());
    accumulator[pixel_index] += value; 
  }
  
  void ToImage(Image &dest, int ystart, int yend) const
  {
    assert (ystart >= 0 && yend>= ystart && yend <= dest.height());
    for (int y=ystart; y<yend; ++y)
    for (int x=0; x<xres; ++x)
    {
      int pixel_index = xres * y + x;
      RGB average = accumulator[pixel_index]/Color::RGBScalar(count);
      Image::uchar rgb[3];
      bool isfinite = average.isFinite().all();
      //bool iszero = (accumulator[pixel_index]==0.).all();
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