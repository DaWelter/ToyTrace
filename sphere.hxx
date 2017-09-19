#ifndef SPHERE_HXX
#define SPHERE_HXX

#include "primitive.hxx"

class Sphere : public Primitive
{
  Double3 center;
  double radius;
public:
	Sphere(Double3 _center,double _radius,Shader *_shader)
	: center(_center),radius(_radius),Primitive(_shader)
	{};

	bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const override
	{
		Double3 q = center-ray.org;
		double t = Dot(q,ray.dir);
		if(t>ray_length+Epsilon+radius || t<-radius+Epsilon) return false;
		Double3 p = ray.org + t*ray.dir;
		double d = Length(p-center);
		if(d>=radius) return false;
		double dt = sqrt(radius*radius-d*d);
		double t1=t+dt,t2=t-dt;
		if(t2>Epsilon)
			t = t2;
		else if(t1>Epsilon) 
			t = t1;
		else return false;
		hit.primitive = this;
    ray_length = t;
    hit.barry = ray.org + t * ray.dir;
		return true;
	};

	virtual Double3 GetNormal(const HitId &hit) const
	{
	  Double3 n(hit.barry-center);
	  Normalize(n);
	  return n;
	}

	virtual Box CalcBounds() const
	{
		Box box;
		box.min = center - Double3(radius+Epsilon);
		box.max = center + Double3(radius+Epsilon);
		return box;
	}
};

#endif
