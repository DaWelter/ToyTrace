#ifndef TEXTURE_HXX
#define TEXTURE_HXX

#include"image.hxx"
#include"vec3f.hxx"
#include"spectral.hxx"
#include"util.hxx"

namespace boost { namespace filesystem { 
  class path;
}}


class Texture
{
  ToyVector<std::uint8_t> data;
  int w, h;
  // Always 3 channels
  enum Type {
    BYTE, // 1 byte per channel
    FLOAT // 4 byte per channel, data really contains floats
  } type;
 
  //void MakeDefaultImage();
  void ReadFile(const std::string &filename);
 
public:
  Texture(const boost::filesystem::path &filename);
  int Width() const { return w; }
  int Height() const { return h; }
  inline RGB GetPixel(int x, int y) const;
  inline RGB GetPixel(std::pair<int,int> xy) const;
};


inline std::pair<int, int> UvToPixel(const Texture &tex, Float2 uv)
{
  const int w = tex.Width();
  const int h = tex.Height();
  float u = uv[0];
  float v = uv[1];
  float dummy;
  u = std::modf(u, &dummy);
  v = std::modf(v, &dummy);
  if (u < 0.f) u = 1. + u;
  if (v < 0.f) v = 1. + v;
  if (u > 1.f - Epsilon) u -= Epsilon;
  if (v > 1.f - Epsilon) v -= Epsilon;
  int x = u * w;
  int y = v * h;
  y = h - y - 1;
  return std::make_pair(x,y);
}


inline Float2 PixelCenterToUv(const Texture &tex, std::pair<int,int> xy)
{
  const int w = tex.Width();
  const int h = tex.Height();
  Float2 uv( xy.first,
             h - 1 - xy.second );
  uv.array() += 0.5f;
  uv[0] /= w;
  uv[1] /= h;
  return uv;
}

inline std::pair<Float2,Float2> PixelToUvBounds(const Texture &tex, std::pair<int,int> xy)
{
  const int w = tex.Width();
  const int h = tex.Height();
  Float2 uv_lower( 
    xy.first,
    xy.second );
  uv_lower[0] /= w;
  uv_lower[1] /= h;
  Float2 uv_upper = uv_lower;
  uv_upper[0] += 1.f/w;
  uv_upper[1] += 1.f/h;
  uv_lower[1] = 1.f-uv_lower[1];
  uv_upper[1] = 1.f-uv_upper[1];
  return std::make_pair(uv_lower, uv_upper);
}


inline RGB Texture::GetPixel(std::pair<int,int> xy) const
{
  return GetPixel(xy.first, xy.second);
}


inline RGB Texture::GetPixel(int x, int y) const
{
  assert (x>=0 && x<Width());
  assert (y>=0 && y<Height());
  constexpr int num_channels = 3;
  const int idx = (y * w + x)*num_channels;
  if (type == BYTE)
  {
    return RGB{
      Color::SRGBToLinear(Color::RGBScalar(data[idx  ] / 255.0)),
      Color::SRGBToLinear(Color::RGBScalar(data[idx+1] / 255.0)),
      Color::SRGBToLinear(Color::RGBScalar(data[idx+2] / 255.0))
    };
  }
  else
  {
    const auto* pix = reinterpret_cast<const float*>(&data[idx*sizeof(float)]);
    return RGB{
      Color::RGBScalar{pix[0]},
      Color::RGBScalar{pix[1]},
      Color::RGBScalar{pix[2]}
    };
  }
}


#endif
