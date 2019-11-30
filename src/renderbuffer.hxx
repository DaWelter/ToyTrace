#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "spectral.hxx"


class ImageTileSet
{
public:
    struct Tile
    {
        Int2 corner;
        Int2 shape;
    };

    ImageTileSet(Int2 im_shape_)
        : im_shape_{im_shape_}
    {
        // Careful with rounding ...

        // Image shape  |    Tiles
        //    0                0
        //    1                1
        //    32               1
        //    33               2
        //  ...
        shape_ = (im_shape_ + basicTileSize() - 1) / basicTileSize();
    }

    int size() const { return shape_.prod(); }
    Int2 shape() const { return shape_; }
    Tile operator[](const int idx) const
    {
        // Grid coordinates
        int j = idx / shape_[0];
        int i = idx - shape_[0] * j;
        // Pixel coordinates
        Int2 xy{ i, j };
        xy *= basicTileSize();
        Int2 s = (im_shape_ - xy).cwiseMin(basicTileSize());
        return { xy, s };
    }

    static constexpr int basicTileSize() { return 16;  }
    static constexpr int basicPixelsPerTile() { return basicTileSize()*basicTileSize(); }

private:
    Int2 shape_; // Size of the tile grid
    Int2 im_shape_;
};

#if 0
//TODO: maybe finish later or delete
template<class T>
class Buffer2d
{
public:
    using Int2 = Eigen::Array2i;

    Buffer2d(Int2 shape_)
        : shape_{ shape_ }, data(size())
    {}

    template<class U>
    Buffer2d(Int2 shape_, U &&arg)
        : shape_{ shape_ }, data(size(), std::forward(arg))
    {}

    int size() const { return shape_.prod(); }
    Int2 shape() const { return shape_; }


    //auto operator[](int i) { return data[i]; }
    //auto operator[](int i) const { return data[i]; }

    auto operator[](const Int2 ij) {
        return data[idx(ij)];
    }

    auto operator[](const Int2 ij) {

    }

    int idx(Int2 ij) const {
        return ij[0] + ij[1] * shape_[0];
    }

private:
    // Using std::vector because std::unique_ptr<T[]> does only allow default initialization
    std::vector<T> data;
    Int2 shape_;
};
#endif

namespace framebuffer
{

inline void ToImage(Image &dest, int xstart, int xend, int ystart, int yend, Span<const RGB> framebuffer, std::uint64_t sampleCount, const bool convert_linear_to_srgb = true)
{
  assert(ystart >= 0 && yend >= ystart && yend <= dest.height());
  const int xres = dest.width();
  for (int y = ystart; y < yend; ++y)
  {
    for (int x = xstart; x < xend; ++x)
    {
      int pixel_index = xres * y + x;
      RGB average = framebuffer[pixel_index] / Color::RGBScalar(sampleCount);
      bool isfinite = average.isFinite().all();
      assert(isfinite);

      Image::uchar rgb[3];

      average = average.max(0._rgb).min(1._rgb);
      if (isfinite)
      {
        for (int i = 0; i < 3; ++i)
        {
          rgb[i] = Image::uchar(convert_linear_to_srgb ? value(Color::LinearToSRGB(average[i])*255.999_rgb) : value(average[i] * 255.999_rgb));
        }
        dest.set_pixel(x, dest.height() - 1 - y, rgb[0], rgb[1], rgb[2]);
      }
    }
  }
}

inline void ToImage(Image &dest, const ImageTileSet::Tile &tile, Span<const RGB> framebuffer, std::uint64_t sampleCount, const bool convert_linear_to_srgb = true)
{
  ToImage(
    dest,
    tile.corner[0],
    tile.corner[0] + tile.shape[0],
    tile.corner[1],
    tile.corner[1] + tile.shape[1],
    framebuffer, sampleCount,
    convert_linear_to_srgb);
}

}




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
  
  void ToImage(Image &dest, int xstart, int xend, int ystart, int yend, const bool convert_linear_to_srgb = true) const
  {
    assert (ystart >= 0 && yend>= ystart && yend <= dest.height());
    Color::RGBScalar splat_weight(splat_count>0 ? double(xres*yres)/(splat_count) : 0.); // Multiply with xres*yres because I divided it out in the path tracer code.
    for (int y = ystart; y < yend; ++y)
    {
        for (int x = xstart; x < xend; ++x)
        {
            int pixel_index = xres * y + x;
            RGB average = accumulator[pixel_index] / Color::RGBScalar(count);
            average += splat_weight * light_accum[pixel_index];
            Image::uchar rgb[3];
            bool isfinite = average.isFinite().all();
            assert(isfinite);
            average = average.max(0._rgb).min(1._rgb);
            if (isfinite)
            {
                for (int i = 0; i < 3; ++i)
                {
                    rgb[i] = Image::uchar(convert_linear_to_srgb ? value(Color::LinearToSRGB(average[i])*255.999_rgb) : value(average[i] * 255.999_rgb));
                }
                dest.set_pixel(x, dest.height() - 1 - y, rgb[0], rgb[1], rgb[2]);
            }
        }
    }
  }

  void ToImage(Image &dest, const ImageTileSet::Tile &tile, const bool convert_linear_to_srgb = true) const
  {
      ToImage(
          dest,
          tile.corner[0],
          tile.corner[0] + tile.shape[0],
          tile.corner[1],
          tile.corner[1] + tile.shape[1],
          convert_linear_to_srgb);
  }

  void ToImage(Image &dest, int ystart, int yend, const bool convert_linear_to_srgb = true) const
  {
      ToImage(dest, 0, xres, ystart, yend, convert_linear_to_srgb);
  }
};
