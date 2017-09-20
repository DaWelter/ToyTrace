#ifndef PRIMITIVE_HXX
#define PRIMITIVE_HXX


class Shader;
class Medium;

#include<vector>
#include"ray.hxx"
#include"box.hxx"

class Primitive
{
public:
  Primitive() 
	  : shader(nullptr), medium(nullptr)
  {}
  virtual ~Primitive() {}
  
  Shader *shader;
  Medium *medium;
  
  virtual bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const = 0;
  
  virtual Double3 GetNormal(const HitId &hit) const = 0;
  
  virtual Box   CalcBounds() const = 0;
  
  virtual bool  Occluded(const Ray &ray, double t) const
  {
    HitId hit;
    return Intersect(ray, t, hit);
  }
  
  virtual Double3 GetUV(const HitId &hit) const
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
