#ifndef PRIMITIVE_HXX
#define PRIMITIVE_HXX


class Shader;

#include<vector>
#include"ray.hxx"
#include"box.hxx"

class Primitive
{
public:
  Primitive(Shader *shader) 
	  : shader(shader)
  { }
  Primitive() 
	  : shader(0) 
  {}
  virtual ~Primitive() {}
  
  Shader *shader;
  
  virtual bool Intersect(const Ray &ray, double &ray_length, SurfaceHit &hit) const = 0;
  
  virtual Double3 GetNormal(const SurfaceHit &hit) const = 0;
  
  virtual Box   CalcBounds() const = 0;
  
  virtual bool  Occluded(const Ray &ray, double t) const
  {
    SurfaceHit hit;
    return Intersect(ray, t, hit);
  }
  
  virtual Double3 GetUV(const SurfaceHit &hit) const
  { 
	  return Double3{0, 0, 0};
  }
  
  virtual bool InBox(const Box &box) const
  {
    Box boundingbox = CalcBounds();
    return boundingbox.Intersect(box);
  }
  
  virtual void ListPrimitives(std::vector<Primitive *> &list)
  { 
    list.push_back(this); 
  }
};

#endif
