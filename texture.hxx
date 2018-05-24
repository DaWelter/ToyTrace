#ifndef TEXTURE_HXX
#define TEXTURE_HXX

#include"image.hxx"
#include"vec3f.hxx"
#include"spectral.hxx"

namespace boost { namespace filesystem { 
  class path;
}}


class Texture
{
	Image bm;
public:
  Texture(const boost::filesystem::path &filename);

  RGB GetTexel(float u, float v) const
  {
    float dummy;
    u = std::modf(u, &dummy);
    v = std::modf(v, &dummy);
    if (u < 0.f) u = 1. + u;
    if (v < 0.f) v = 1. + v;
    if (u > 1.f - Epsilon) u -= Epsilon;
    if (v > 1.f - Epsilon) v -= Epsilon;
    int x = u * bm.width();
    int y = v * bm.height();
    y = bm.height() - y - 1.f;
    //x = bm.width() - x - 1; // Nope. I don't think this is it.
    auto rgb = bm.get_pixel_uc3(x, y);
    RGB c;
    c[0] = Color::SRGBToLinear(Color::RGBScalar(std::get<0>(rgb) / 255.0));
    c[1] = Color::SRGBToLinear(Color::RGBScalar(std::get<1>(rgb) / 255.0));
    c[2] = Color::SRGBToLinear(Color::RGBScalar(std::get<2>(rgb) / 255.0));
    //c = (Color::SRGBToRGBMatrix()*c.matrix()).array();
    return c;
  }
};

#endif
