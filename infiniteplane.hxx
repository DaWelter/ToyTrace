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

  bool Intersect(const Ray &ray, double &ray_length, SurfaceHit &hit) const override
  {
		double s = Dot(origin,normal);
		double nv = Dot(ray.dir,normal);
		if(nv*nv < 1.0e-9) 
      return false;
		double t = (s-Dot(ray.org,normal))/(nv);
		if(t<Epsilon || t>ray_length+Epsilon) 
      return false;
    hit.primitive = this;
    ray_length = t;
    return true;
  };

  virtual Double3 GetNormal(const SurfaceHit &hit) const
  {
	  return normal;
  }

  virtual Box CalcBounds() const
  {
	  Box box;
	  box.min = -Double3(Infinity);
	  box.max = Double3(Infinity);
	  return box;
  }

  virtual bool InBox(const Box &box) const
  {  
    return false; 
  }
};

#endif
