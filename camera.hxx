#ifndef PERSPECTIVECAMERA_HXX
#define PERSPECTIVECAMERA_HXX


#include "vec3f.hxx"
#include "ray.hxx"
#include "radianceorimportance.hxx"


class Camera : public RadianceOrImportance::PointEmitterArray
{
public:
  using PositionSample = RadianceOrImportance::PositionSample;
  using DirectionalSample = RadianceOrImportance::DirectionalSample;
  
  int xres,yres;
  
	Camera(int _xres,int _yres)
		: RadianceOrImportance::PointEmitterArray{_xres * _yres},
      xres(_xres),yres(_yres)
	{}
    
  std::pair<int, int> UnitToPixel(int unit) const
  {
    int y = unit / xres;
    int x = unit - y * xres;
    assert(y >= 0 && y < yres);
    assert(x >= 0 && x < xres);
    return std::make_pair(x,y); 
  }
  
  int PixelToUnit(std::pair<int,int> coord) const
  {
    
    return coord.second * xres + coord.first;
  }
  
  struct Frame
  {
    Double3 up, dir, right;
  };
  
  static Frame MakeFrame(const Double3 &_dir, const Double3 &_up)
  {
    auto up = Normalized(_up);
    auto dir = Normalized(_dir);
    double r = Dot(dir,up); 
    if(std::abs(r)>1.0-Epsilon) {
      std::cerr<<"PerspectiveCamera::PerspectiveCamera: up dir ambiguous"<<std::endl;
    }
    else 
    {
      up = up - r * dir;
      Normalize(up);
    }
    auto right = Cross(up,dir);
    return Frame{up, dir, right};
  }
};


class PerspectiveCamera : public Camera
{
  Double3 pos;
  double fov;
  Camera::Frame frame;
  double xmin, xperpixel;
  double ymin, yperpixel;
public: 
	PerspectiveCamera(const Double3 &_pos,
                    const Double3 &_dir,
                    const Double3 &_up,
                    double _fov,
                    int _xres,int _yres)
  : Camera(_xres,_yres),
    pos(_pos),fov(_fov)
  {
    frame = Camera::MakeFrame(_dir, _up);
    fov *= Pi/180.;
    double a = xres;
    double b = yres;
    double aspect = b/a;
    a = tan(fov*0.5);
    b = a*aspect;
    xmin = -a;
    ymin = -b;
    xperpixel = 2.*a / xres;
    yperpixel = 2.*b / yres;
    std::cout << "Perspective Camera Frame: " << std::endl;
    std::cout << "up=" << frame.up << std::endl;
    std::cout << "right=" << frame.right << std::endl;
    std::cout << "fwd=" << frame.dir << std::endl;
	}

	double PixelPdfWrtSolidAngle(double x, double y) const
  {
    double screen_surface_area = xperpixel*yperpixel;
    Double3 v{x, y, 1.};
    double  l = Length(v);
    v *= 1./l;
    double pdf = 1./screen_surface_area* PdfConversion::AreaToSolidAngle(l, v, Double3{0., 0., 1.});
    return pdf;
  }
	
  virtual PositionSample TakePositionSample(int unit_index, Sampler &sampler, const PathContext &context) const override
  {
    PositionSample s{this->pos, Spectral3{1.}, Pdf::MakeFromDelta(1.)};    
    return s;
  }
  
  virtual DirectionalSample TakeDirectionSampleFrom(int unit_index, const Double3 &pos, Sampler &sampler, const PathContext &context) const override
  {
    auto pixel = this->UnitToPixel(unit_index);
    double r1 = sampler.Uniform01();
    double r2 = sampler.Uniform01();
    Double3 v(
      xmin + (r1 + std::get<0>(pixel)) * xperpixel,
      ymin + (r2 + std::get<1>(pixel)) * yperpixel,
      1.0);
    Normalize(v);
    Double3 emission_dir = frame.right*v[0] + frame.up*v[1] + frame.dir*v[2];
    double pdf = PixelPdfWrtSolidAngle(v[0], v[1]);
    DirectionalSample s{emission_dir, Spectral3{pdf}, pdf};
    return s;
  }
  
  virtual Response Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, const PathContext &context, double *pdf_direction) const override
  {
    ASSERT_NORMALIZED(dir_out);
    // Oh yeah why didn't I just use a matrix multiply ...
    double z = Dot(dir_out, frame.dir);
    double x = Dot(dir_out, frame.right);
    double y = Dot(dir_out, frame.up);
    if (z <= 0.)
    {
      if (pdf_direction) *pdf_direction = 0.;
      return {};
    }
    x /= z;
    y /= z;
    int pix_x = (x-xmin)/xperpixel;
    int pix_y = (y-ymin)/yperpixel;
    if (pix_x<0 || pix_x>=xres || pix_y<0 || pix_y>=yres)
    {
      if (pdf_direction) *pdf_direction = 0.;
      return {};
    }
    // Units don't overlap (yet). Therefore there can only respond one of them at most.
    double pdf = PixelPdfWrtSolidAngle(x, y);
    if (pdf_direction)
      *pdf_direction = pdf; // TODO: correct or not???
    return {
      PixelToUnit({pix_x, pix_y}),
      Spectral3{pdf}, // value
    };
  }
};


class FisheyeHemisphereCamera : public Camera
{
  Double3 pos;
  Camera::Frame frame;
  double per_pixel_delta;
public: 
  FisheyeHemisphereCamera(const Double3 &_pos,
                          const Double3 &_dir,
                          const Double3 &_up,
                          int _xres,int _yres)
  : Camera(_xres,_yres), pos(_pos)
  {
    frame = Camera::MakeFrame(_dir, _up);
    double smallest_side = std::min(xres, yres);
    per_pixel_delta = 2./smallest_side;
  }

  virtual PositionSample TakePositionSample(int unit_index, Sampler &sampler, const PathContext &context) const override
  {
    PositionSample s{this->pos, Spectral3{1.}, Pdf::MakeFromDelta(1.)};    
    return s;
  }
  
  virtual DirectionalSample TakeDirectionSampleFrom(int unit_index, const Double3 &pos, Sampler &sampler, const PathContext &context) const override
  {
    auto pixel = this->UnitToPixel(unit_index);
    auto r = sampler.UniformUnitSquare();
    double u = (r[0] + std::get<0>(pixel))*per_pixel_delta - 1.0;
    double v = (r[1] + std::get<1>(pixel))*per_pixel_delta - 1.0;
    // Ref: https://en.wikipedia.org/wiki/Stereographic_projection
    // It is one of several fisheye projections 
    // https://wiki.panotools.org/Fisheye_Projection
    double rho = u*u + v*v;
    Double3 w{ 2.*u, 2.*v, rho-1. };
    w *= 1./(1.+rho);
    // Up to here it is according to the refs. But z points down for rho<1. So just flip it.
    w[2] = -w[2];
    if (rho <= 1.)
    {
      Double3 emission_dir = frame.right*w[0] + frame.up*w[1] + frame.dir*w[2];
      ASSERT_NORMALIZED(emission_dir);
      DirectionalSample s{ emission_dir, Spectral3{1.}, 1.0};
      SetPmfFlag(s);
      return s;
    }
    else
    {
      DirectionalSample s{frame.dir, Spectral3{0.}, 1.0};
      SetPmfFlag(s);
      return s;
    }
  }
  
  virtual Response Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, const PathContext &context, double *pdf_direction) const override
  {
//     ASSERT_NORMALIZED(dir_out);
//     double w2 = Dot(dir_out, frame.dir);
//     double w0 = Dot(dir_out, frame.right);
//     double w1 = Dot(dir_out, frame.up);
//     if (w2 <= 0.)
//       return {};
//     w2 = -w2;
//     double u = w0/(1.-w2+Epsilon);
//     double v = w1/(1.-w2+Epsilon);
//     
//     int pix_x = (x-xmin)/xperpixel;
//     int pix_y = (y-ymin)/yperpixel;
//     if (pix_x<0 || pix_x>=xres || pix_y<0 || pix_y>=yres)
//       return {};
    if (pdf_direction)
      *pdf_direction = 0;
    return {};
  }
};


#endif
