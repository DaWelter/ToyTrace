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

	virtual bool Intersect(Ray &ray)
	{
		Double3 q = center-ray.org;
		double t = Dot(q,ray.dir);
		if(t>ray.t+Epsilon+radius || t<-radius+Epsilon) return false;
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
		ray.t = t;
		ray.hit = this;
		return true;
	};

	virtual Double3 GetNormal(Ray &ray)
	{
	  Double3 n(ray.org+ray.t*ray.dir-center);
	  Normalize(n);
	  return n;
	}

	virtual Box CalcBounds()
	{
		Box box;
		box.min = center - Double3(radius+Epsilon);
		box.max = center + Double3(radius+Epsilon);
		return box;
	}

	virtual bool Occluded(Ray &ray) 
	{  return Intersect(ray);  }

	virtual bool InBox(const Box &box)
	{
		Box boundingBox = CalcBounds();
		return boundingBox.Intersect(box);
	}

	virtual void ListPrimitives(std::vector<Primitive *> &list)
	{	list.push_back(this); }
};

#endif
