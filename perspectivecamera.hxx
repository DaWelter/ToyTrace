#ifndef PERSPECTIVECAMERA_HXX
#define PERSPECTIVECAMERA_HXX


#include "vec3f.hxx"
#include "ray.hxx"
#include "radianceorimportance.hxx"


class Camera : public RadianceOrImportance::PathEndPoint
{
public:
	Camera(int xres,int yres)
		: xres(xres),yres(yres),
      current_pixel_x(0), current_pixel_y(0)
	{}
	int xres,yres;
  int current_pixel_x, current_pixel_y;
};


class PerspectiveCamera : public Camera
{
	double fov;
	Double3 up, dir, pos, right;
  double xmin, xperpixel;
  double ymin, yperpixel;
public:
  using Sample = RadianceOrImportance::Sample;
  using DirectionalSample = RadianceOrImportance::DirectionalSample;
  
	PerspectiveCamera(const Double3 &_pos,
                    const Double3 &_dir,
                    const Double3 &_up,
                    double _fov,
                    int _xres,int _yres)
  : Camera(_xres,_yres),
    up(_up),dir(_dir),pos(_pos),fov(_fov)
  {
    Normalize(up);
    Normalize(dir);
    double r = Dot(dir,up); 
    if(abs(r)>1.0-Epsilon) {
      std::cerr<<"PerspectiveCamera::PerspectiveCamera: up dir ambiguous"<<std::endl;
    }
    else 
    {
      fov *= Pi/180.;
      up = up - r * dir;
      Normalize(up);
    }
    right = Cross(up,dir);

    double a = xres;
    double b = yres;
    double aspect = b/a;
    a = tan(fov*0.5);
    b = a*aspect;
    xmin = -a;
    ymin = -b;
    xperpixel = 2.*a / xres;
    yperpixel = 2.*b / yres;
	}

  Sample TakePositionSample(Sampler &sampler) const override
  {
    Sample s{this->pos, 1., Double3::Constant(1.), false};    
    return s;
  }
  
  DirectionalSample TakeDirectionalSampleFrom(const Double3 &pos, Sampler &sampler) const override
  {
    double r1 = sampler.Uniform01();
    double r2 = sampler.Uniform01();
    Double3 v(
      xmin + (r1 + current_pixel_x) * xperpixel,
      ymin + (r2 + current_pixel_y) * yperpixel,
      1.0);
    Normalize(v);
    Double3 emission_dir = right*v[0] + up*v[1] + dir*v[2];
    DirectionalSample s{{pos, emission_dir}, 1.0, Double3::Constant(1.)};
    return s;
  }
};



#endif