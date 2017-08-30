#ifndef DRAWING_H_
#define DRAWING_H_

#include <string>
#include <vector>
#include <array>

#if (((defined __GNUC__) && (defined __LP64__)) || (defined _WIN64))
  #define IMG_PRIVATE_DATA_SIZE 32
#else
  #define IMG_PRIVATE_DATA_SIZE 24
#endif

#define MAX_IMG_SIZE 16384 //2^14

class Image
{
public:
  typedef unsigned char uchar;
  
  Image();
  Image( const Image &img );
  Image( int w, int h );

  void init( int w, int h );
  
  ~Image();
  void clear() { init(0,0); }

  int width() const;
  int height() const;
  bool empty() const { return width()<=0 || height()<=0; }

  void Fill( uchar r, uchar g, uchar b );
  void Rescale( int w, int h, int interpolation=1 ) { MyRescale(w,h,interpolation); }

  void DrawLine( int x1, int y1, int x2, int y2 );
  void DrawLine( int x1, int y1, int x2, int y2, int width );
  void DrawRect( int x1, int y1, int x2, int y2 );
  void DrawRectOutline( int x1, int y1, int x2, int y2 );
  void DrawPixel( int x, int y );
  void DrawPoint( int x, int y, int width );
  void DrawImage( const Image &img, int x, int y );
  void DrawText( int x, int y, const char* text, const uchar* colbg = NULL )         { MyDrawText(x,y,text,colbg); }
  void DrawText( int x, int y, const std::string &text, const uchar* colbg = NULL )  { MyDrawText(x,y,text.c_str(),colbg); }

  void SetColor( uchar r, uchar g, uchar b );
  void SetColorF( double r, double g, double b )                       { SetColor((uchar)(255.99*r), (uchar)(255.99*g), (uchar)(255.99*b)); }
  void SetOpacity( double a );

  std::array<uchar, 3> get_pixel_uc3( int x, int y) const;
  void set_pixel( int x, int y, uchar r, uchar g, uchar b )            { SetColor(r, g, b); DrawPixel(x, y); }

//   bool WritePNM( const std::string &filename ) const;
//   bool WritePng( const std::string &filename ) const;
  bool write( const std::string &filename ) const;

  bool Read( const std::string &filename );

  Image& operator*=( double x );
  Image& operator+=( Image &img );
  Image& operator=( const Image &img );

  uchar* GetDataAddress( int x, int y, int c );

  template<class Array>  void DrawPixel( const Array &p )                    { DrawPixel(p[0],p[1]); }
  template<class Array>  void SetColor( const Array &v )                     { SetColor(v[0], v[1], v[2]); }
  template<class Array>  void SetColorF( const Array &v )                     { SetColorF(v[0], v[1], v[2]); }
  template<class Array, class Array2>  void SetPixel(const Array &p, const Array2 &v) { SetColor(v); DrawPixel(p); }
  template<class Array, class Array2>  void SetPixelF(const Array &p, const Array2 &v) { SetColorF(v); DrawPixel(p); }
  template<class Array, class Array2>  void GetPixel(const Array &p, Array2 &v) { GetPixel(p[0], p[1], v[0], v[1], v[2]); }
  
private:
  friend class ImageDisplay;
  
  // hack around stupid macros in windows headers that redefine DrawText and Rescale
  void MyDrawText( int x, int y, const char* text, const uchar* colbg );
  void MyRescale( int w, int h, int interpolation );

  struct CImgInstanceHolder
  {
    uchar buffer[IMG_PRIVATE_DATA_SIZE];
  } __attribute__((__may_alias__)) __attribute__ ((aligned (sizeof(void*)))) priv;
  
  uchar col[3];
  double opacity;
};


void DrawImageGrid(Image& dst, const std::vector< Image >& images, int dir = 0);
void DrawImageGrid(Image &dst, const std::vector<std::vector<Image> > &images);

namespace cimg_library {
class CImgDisplay;
}

class ImageDisplay
{
public:
  ImageDisplay();
  void show(const Image &img);
  
private:
  struct Impl
  {
    static constexpr int CIMG_DISPLAY_SIZE = 4096; // sufficiently large
    char buffer[CIMG_DISPLAY_SIZE];
  } __attribute__((__may_alias__)) __attribute__ ((aligned (sizeof(void*)))) priv;
  cimg_library::CImgDisplay* impl();
  const cimg_library::CImgDisplay* impl() const;
};


#endif
