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
  Shader *shader;
  virtual bool  Intersect(Ray &ray) = 0;
  virtual Double3 GetNormal(Ray &ray) = 0;
  virtual Box   CalcBounds() = 0;
  virtual bool  Occluded(Ray &ray) = 0;
  virtual Double3 GetUV(Ray &ray) { 
	  return Double3{0, 0, 0};
  }
  virtual bool InBox(const Box &box) = 0;
  virtual void ListPrimitives(std::vector<Primitive *> &list) = 0;
};

#endif
