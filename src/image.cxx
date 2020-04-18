#include <stdlib.h>
#include <new>

#include "zlib.h"
#define cimg_use_png 1
#define cimg_use_jpeg 1
#define cimg_debug 0
#ifdef _MSC_VER 
#undef NOGDI // Make API available.
#define cimg_OS 2
#else
#define cimg_OS 1
#define cimg_display 1
#endif


#include "image.hxx"

#undef APIENTRY
#undef WINGDIAPI

#include <cassert>
#include "CImg.h"


typedef cimg_library::CImg<Image::uchar> CImgType;

#define TOCIMG(p) reinterpret_cast<CImgType*>(&p)
#define TOCIMG_CONST(p) reinterpret_cast<CImgType const *>(&p)

Image::Image()
{
  col[0]=col[1]=col[2] = 0;
  opacity=1.0f;
  static_assert(sizeof(priv)==sizeof(CImgType), "memory reserve size must equal size of CImg<T>");
  new(TOCIMG(priv)) CImgType( (unsigned)0, (unsigned)0, (unsigned)1, (unsigned)3 );
}

Image::Image( const Image &other )
{
  for (int i=0; i<3; ++i)
    col[i] = other.col[i];
  opacity = other.opacity;
  new (TOCIMG(priv)) CImgType(*TOCIMG_CONST(other.priv));
}

Image::Image( int w, int h )
{
  //printf("sizeof(CImgType)=%u",sizeof(CImgType));
  col[0]=col[1]=col[2] = 0;
  opacity=1.0f;
  static_assert(sizeof(priv)==sizeof(CImgType),"must have same size");
  new(TOCIMG(priv)) CImgType( w, h, 1, 3, 0 );  
}

void Image::init( int w, int h )
{
  col[0]=col[1]=col[2] = 0;
  opacity=1.0f;
  TOCIMG(priv)->~CImgType();
  new(TOCIMG(priv)) CImgType( w, h, 1, 3, 0 );
}

Image::~Image()
{
  TOCIMG(priv)->~CImgType();
}

int Image::width() const
{
  return TOCIMG_CONST(priv)->width();
}

int Image::height() const
{
  return TOCIMG_CONST(priv)->height();
}

void Image::MyRescale( int w, int h, int interpolation )
{
  TOCIMG(priv)->resize( w, h, -100, -100, interpolation==0 ? 1 : 5 );
}

void Image::Fill( uchar r, uchar g, uchar b )
{
  TOCIMG(priv)->fill( r, g, b );
}

void Image::DrawLine( int x1, int y1, int x2, int y2 )
{
  TOCIMG(priv)->draw_line( x1,y1, x2,y2, col, opacity );
}

void Image::DrawRect( int x1, int y1, int x2, int y2 )
{
  TOCIMG(priv)->draw_rectangle( x1,y1,x2,y2, col,opacity );
}

void Image::DrawRectOutline( int x1, int y1, int x2, int y2 )
{
  TOCIMG(priv)->draw_line( x1+1,y1, x2-1,y1, col, opacity );
  TOCIMG(priv)->draw_line( x2,y1, x2,y2, col, opacity );
  TOCIMG(priv)->draw_line( x2-1,y2, x1+1,y2, col, opacity );
  TOCIMG(priv)->draw_line( x1,y2, x1,y1, col, opacity );
}

void Image::DrawPixel( int x, int y )
{
  assert (x >= 0 && x < width() && y >= 0 && y < height());
  TOCIMG(priv)->draw_point( x,y, col,opacity );
}

void Image::DrawImage( const Image &img, int x, int y )
{
  if (img.width() <= 0 || img.height() <= 0) return;
  TOCIMG(priv)->draw_image(x, y, 0, 0, *TOCIMG_CONST(img.priv), opacity );
}

void Image::SetColor( uchar r, uchar g, uchar b )
{
  col[0]=r; col[1]=g; col[2]=b;
}

void Image::set_pixel(int x, int y, Image::uchar r, Image::uchar g, Image::uchar b)
{
  Image::uchar col[3] = { r, g, b};
  TOCIMG(priv)->draw_point( x,y, col );
}


// void Image::SetColor( const uchar* c )
// {
//   col[0]=c[0]; col[1]=c[1]; col[2]=c[2];
// }

// void Image::SetColor( uchar rgb )
// {
//   col[0]=col[1]=col[2]=rgb;
// }

// void Image::SetPixel( int x, int y, uchar r, uchar g, uchar b )
// {
//   SetColor(r,g,b);
//   DrawPixel(x,y);
// }

// void Image::SetPixel( int x, int y, uchar rgb )
// {
//   SetColor(rgb);
//   DrawPixel(x,y);
// }

void Image::SetOpacity( double a )
{
  opacity = (float)a;
}

// bool Image::WritePNM( const std::string &filename ) const
// {
// 	TOCIMG_CONST(priv)->save_pnm( filename.c_str() );
//   return true;
// }
// 
// bool Image::WritePng( const std::string &filename ) const
// {
//     TOCIMG_CONST(priv)->save_png( filename.c_str() );
//     return true;
// }

bool Image::write( const std::string &filename ) const
{  
  TOCIMG_CONST(priv)->save( filename.c_str() );
  return true;  
}

bool Image::Read( const std::string &filename )
{
  TOCIMG(priv)->load( filename.c_str() );
  return true;
}


// Note for auto return type without appended -> <return type> we need c++14
std::array<Image::uchar, 3> Image::get_pixel_uc3( int x, int y) const
{
  return std::array<Image::uchar, 3>{
    TOCIMG_CONST(priv)->operator()( x, y, 0,0 ),
    TOCIMG_CONST(priv)->operator()( x, y, 0,1 ),
    TOCIMG_CONST(priv)->operator()( x, y, 0,2 )
  };
}

//static const int g_wmin[] = { 0, 0, 0, -1, -1, -2, -2, -4 }; // ....
//static const int g_wmax[] = { 0, 0, 1,  1,  2,  2,  3,  3 }; // ....

void Image::DrawLine( int x1, int y1, int x2, int y2, int w )
{
  if( w<0 ) w=0;
  int max = w/2;
  int min = (1-w)/2;
  //cout <<"line: w"<<w<<" max"<<max<<" min"<<min<<endl;
  int dx,dy;
  for( dx=min; dx<=max; ++dx )
    for( dy=min; dy<=max; ++dy )
    {
      DrawLine( x1+dx,y1+dy,x2+dx,y2+dy );
    }
}

void Image::DrawPoint( int x, int y, int w )
{
	if( w<0 ) w=0;
	int max = w/2;
	int min = (1-w)/2;
	int dx,dy;
  for( dx=min; dx<=max; ++dx ) {
		for( dy=min; dy<=max; ++dy )
		{
			if( dx*dx + dy*dy <= w*w )  TOCIMG(priv)->draw_point( x+dx,y+dy, col,opacity );
		}
  }
}


void Image::MyDrawText( int x, int y, const char* text, const Image::uchar* colbg )
{
	TOCIMG(priv)->draw_text(x, y, text, col, colbg, opacity, 11);
}

#if 1
Image::uchar* Image::GetDataAddress( int x, int y, int c )
{
  return (uchar*)(TOCIMG(priv)->data( x, y, 0, c ));
}
#endif


Image& Image::operator*=( double x )
{
  TOCIMG(priv)->operator*=( x );
  return *this;
}

Image& Image::operator+=( Image &img )
{
  TOCIMG(priv)->operator+=( *TOCIMG(img.priv) );
  return *this;
}

Image& Image::operator=( const Image &img )
{
  TOCIMG(priv)->operator=( *TOCIMG_CONST(img.priv) );
  return *this;
}


void DrawImageGrid(Image &dst, const std::vector<Image> &images, int dir)
{
  int dir2 = (dir+1)%2;
  int s[2] = {0, 0};
  for (int i=0; i<images.size(); ++i)
  {
    const int t[2] = { images[i].width(), images[i].height() };
    s[dir] += t[dir];
    s[dir2] = std::max(s[dir2], t[dir2]);
  }
  dst.init(s[0], s[1]);
  s[0] = s[1] = 0;
  for (int i=0; i<images.size(); ++i)
  {
    const int t[2] = { images[i].width(), images[i].height() };
    dst.DrawImage(images[i], s[0], s[1]);
    s[dir] += t[dir];
  }
}

void DrawImageGrid(Image &dst, const std::vector<std::vector<Image> > &images)
{
  std::vector<Image> rows(images.size());
  for (int i=0; i<images.size(); ++i)
  {
    DrawImageGrid(rows[i], images[i], 0);
  }
  DrawImageGrid(dst, rows, 1);
}


ImageDisplay::ImageDisplay()
{
  static_assert(sizeof(priv)>=sizeof(cimg_library::CImgDisplay), "memory reserve size must equal size of CImgDisplay");
  new(impl()) cimg_library::CImgDisplay();
}


void ImageDisplay::show(const Image& img)
{
  const auto* img_impl = TOCIMG_CONST(img.priv);
  impl()->display(*img_impl);
//   impl()->show();
//   while (!impl()->is_closed()) 
//   {
//     impl()->wait();
//   }
}


bool ImageDisplay::is_open() const
{
  return !impl()->is_closed();
}


const cimg_library::CImgDisplay* ImageDisplay::impl() const
{
  return reinterpret_cast<const cimg_library::CImgDisplay*>(priv.buffer);
}

cimg_library::CImgDisplay* ImageDisplay::impl()
{
  return reinterpret_cast<cimg_library::CImgDisplay*>(priv.buffer);
}
