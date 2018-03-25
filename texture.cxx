#include "texture.hxx"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace fs = boost::filesystem;

Texture::Texture(const fs::path &filename)
{
  if (fs::exists(filename))
  {
    std::cout << "Texture: " << filename << std::endl;
    bm.Read(filename.string());
  }
  else
  {
    std::cerr << "Texture file does not exist: " << filename << std::endl;
    bm.init(1,1);
    bm.set_pixel(0, 0, 255, 0, 0);
  }
}