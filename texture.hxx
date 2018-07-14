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
  
  inline RGB GetTexel(Float2 uv) const
  {
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
    y = h - y - 1.f;
    return GetPixel(x, y);
  }
};


inline RGB Texture::GetPixel(int x, int y) const
{
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
