#ifndef PERSPECTIVECAMERA_HXX
#define PERSPECTIVECAMERA_HXX


#include "vec3f.hxx"
#include "ray.hxx"
#include "radianceorimportance.hxx"


class Camera : public RadianceOrImportance::EmitterSensorArray
{
public:
  int xres,yres;
  
	Camera(int _xres,int _yres)
		: RadianceOrImportance::EmitterSensorArray{_xres * _yres},
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

  virtual Sample TakePositionSample(int unit_index, Sampler &sampler) const override
  {
    Sample s{this->pos, 1., Double3::Constant(1.), false};    
    return s;
  }
  
  virtual DirectionalSample TakeDirectionSampleFrom(int unit_index, const Double3 &pos, Sampler &sampler) const override
  {
    auto pixel = this->UnitToPixel(unit_index);
    double r1 = sampler.Uniform01();
    double r2 = sampler.Uniform01();
    Double3 v(
      xmin + (r1 + std::get<0>(pixel)) * xperpixel,
      ymin + (r2 + std::get<1>(pixel)) * yperpixel,
      1.0);
    Normalize(v);
    Double3 emission_dir = right*v[0] + up*v[1] + dir*v[2];
    DirectionalSample s{{pos, emission_dir}, 1.0, Double3::Constant(1.)};
    return s;
  }
  
  virtual void Evaluate(const Double3 &pos_on_this, const Double3 &dir_out, std::vector<Response> &responses) const override
  {
    assert(false && "not implemented");
  }
};



#endif