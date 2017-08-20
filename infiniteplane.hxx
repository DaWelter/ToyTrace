#ifndef INFINITEPLANE_HXX
#define INFINITEPLANE_HXX

#include"primitive.hxx"

class InfinitePlane : public Primitive
{
  Double3 normal;
  Double3 origin;
public:
  InfinitePlane(Double3 origin, Double3 normal,Shader *shader)
    : Primitive(shader),normal(normal),origin(origin)
  { 
	  Normalize(normal); 
  };

  virtual bool Intersect(Ray &ray)
  {
		double s = Dot(origin,normal);
		double nv = Dot(ray.dir,normal);
		if(nv*nv < 1.0e-9) return false;
		double t = (s-Dot(ray.org,normal))/(nv);
		if(t<Epsilon || t>ray.t+Epsilon) return false;
		//p_hit = r.o+t*r.v;
		ray.t = t;
		ray.hit = this;
		return true;
  };

  virtual Double3 GetNormal(Ray &ray)
  {
	  return normal;
  }

  virtual Box CalcBounds()
  {
	  Box box;
	  box.min = -Double3(Infinity);
	  box.max = Double3(Infinity);
	  return box;
  }

  virtual bool Occluded(Ray &ray) 
  {  return Intersect(ray);  }

  virtual bool InBox(const Box &box)
  {  return false; }

  virtual void ListPrimitives(std::vector<Primitive *> &list)
  {	 list.push_back(this); }
};

#endif
