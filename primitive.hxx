#ifndef PRIMITIVE_HXX
#define PRIMITIVE_HXX


class Shader;
class Medium;
namespace RadianceOrImportance {
class AreaEmitter;
}
class Sampler;

#include<vector>
#include"ray.hxx"
#include"box.hxx"

class Primitive
{
public:
  Primitive() 
	  : shader(nullptr), medium(nullptr), emitter{nullptr}
  {}
  virtual ~Primitive() {}
  
  const Shader *shader;
  const Medium *medium;
  const RadianceOrImportance::AreaEmitter *emitter;
  
  virtual bool Intersect(const Ray &ray, double tnear, double &tfar, HitId &hit) const = 0;
  
//   virtual void Intersect(const Ray &ray, double ray_length, HitVector &hits) const
//   {
//     HitId hit;
//     if (Intersect(ray, ray_length, hit))
//       hits.push_back(HitRecord{hit, ray_length});
//   }
  
  virtual void GetLocalGeometry(
      const HitId &hit,
      Double3 &hit_point,
      Double3 &normal,
      Double3 &shading_normal) const = 0;
  
  virtual Box   CalcBounds() const = 0;
  
  virtual bool  Occluded(const Ray &ray, double t) const
  {
    HitId hit;
    return Intersect(ray, 0, t, hit);
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
  
  virtual HitId SampleUniformPosition(Sampler &sampler) const
  {
    assert(!"Not Implemented");
    return HitId{};
  };
  
  virtual double Area() const
  {
    assert(!"Not Implemented");
    return 0;
  }
};

#endif
