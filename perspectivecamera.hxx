#ifndef PERSPECTIVECAMERA_HXX
#define PERSPECTIVECAMERA_HXX


#include "vec3f.hxx"


class Camera
{
public:
	Camera(int xres,int yres) 
		: xres(xres),yres(yres) 
	{}
	int xres,yres;
	virtual void InitRay(double x,double y,Ray &r) = 0;
};

class PerspectiveCamera : public Camera
{
	double fov;
	Double3 up,dir,pos;
public:
	PerspectiveCamera(const Double3 &_pos,
					  const Double3 &_dir,
					  const Double3 &_up,
					  double _fov,
					  int _xres,int _yres) : Camera(_xres,_yres),
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
	}

	virtual void InitRay(double x,double y,Ray &r)
	{
		Double3 right = Cross(up,dir);

		Double3 v;
		double a = xres; 
		double b = yres;
		double s = b/a;
		x /= a;
		y /= b;
		a = tan(fov*0.7);
		b = a*s;
		v[0] = (1.0-2.0*x)*a;
		v[1] = -(1.0-2.0*y)*b;
		v[2] = 1.0;
		Normalize(v);

		r.dir = right*v[0]+ up*v[1] + dir*v[2];
		r.org = pos;
		r.t   = 1000.;
	}
};



#endif