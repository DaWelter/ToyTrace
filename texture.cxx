#include "texture.hxx"
#include "span.hxx"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <OpenImageIO/imageio.h>

namespace fs = boost::filesystem;


Texture::Texture(const fs::path &filename)
{
  if (fs::exists(filename))
  {
    std::cout << "Texture: " << filename << std::endl;
    ReadFile(filename.string());
  }
  else
  {
    throw std::invalid_argument(strconcat("Texture file does not exist: ",filename.string()));;
  }
}


namespace {
void MakeThreeChannels(Span<std::uint8_t> dst, Span<const std::uint8_t> src, int num_pixels, int bytes_per_channel, int dst_channels, int src_channels); 
}

void Texture::ReadFile(const std::string &filename)
{
  auto *in = OIIO::ImageInput::open (filename);
  if (! in)
    throw std::invalid_argument(strconcat("Failed to open image file: ", filename));
  SCOPE_EXIT(OIIO::ImageInput::destroy (in););
  
  const OIIO::ImageSpec &spec = in->spec();
  w = spec.width;
  h = spec.height;
  int channels = spec.nchannels;
  int bytes_per_channel = 1;
  if (spec.format.is_floating_point())
  {
    type = FLOAT;
    bytes_per_channel = sizeof(float);
    data.resize(w*h*bytes_per_channel*channels);
    in->read_image (OIIO::TypeDesc::FLOAT, (float*)&data[0]);
  }
  else
  {
    type = BYTE;
    data.resize(w*h*channels);
    in->read_image (OIIO::TypeDesc::UINT8, &data[0]);
  }
  in->close ();
  
  if (channels != 3)
  {
    ToyVector<std::uint8_t> bad; std::swap(bad, data);
    data.resize(w*h*bytes_per_channel*3);
    MakeThreeChannels(AsSpan(data), AsSpan(bad), w*h, bytes_per_channel, 3, channels);
  }
}


namespace 
{
void MakeThreeChannels(Span<std::uint8_t> dst_span, Span<const std::uint8_t> src_span, int num_pixels, int bytes_per_channel, int dst_channels, int src_channels)
{
  const int dst_incr = dst_channels*bytes_per_channel;
  const int src_incr = src_channels*bytes_per_channel;
  const int num_common_channels = std::min(src_channels, dst_channels);
  auto* dst = dst_span.begin();
  auto* src = src_span.begin();
  for (int pixel = 0; pixel < num_pixels; pixel++)
  {
    
    auto CopyChannel = [=](int ch_dst, int ch_src) {
      assert(IsInRange(dst_span, dst+ch_dst*bytes_per_channel + bytes_per_channel-1));
      assert(IsInRange(src_span, src+ch_src*bytes_per_channel + bytes_per_channel-1));
      memcpy(dst + ch_dst*bytes_per_channel, src + ch_src*bytes_per_channel, bytes_per_channel);
    };
    for (int i=0; i<num_common_channels; ++i)
      CopyChannel(i, i);
    for (int i=num_common_channels; i<dst_channels; ++i)
      CopyChannel(i, num_common_channels-1);
    dst += dst_incr;
    src += src_incr;
  }
}
}