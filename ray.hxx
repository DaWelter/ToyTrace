#ifndef RAY_HXX
#define RAY_HXX

#include <limits>
#include "vec3f.hxx"

class Primitive;

static constexpr int MAX_RAY_DEPTH = 10;

struct Ray
{
  Ray() : level(0) {}
  Ray(Double3 org, Double3 dir) : org(org), dir(dir) {}

  char level; // 0 = primary ray, 1,2,3... secondary ray
  
  Double3 org; // origin
  Double3 dir; // direction
};


inline std::ostream &operator<<(std::ostream &o,const Ray &ray)
{ o << "Ray[" << ray.org << "+t*" << ray.dir << "]"; return o; }


struct SurfaceHit
{
  const Primitive *primitive = { nullptr }; // primitive that was hit
  Double3 barry;
  
  bool isValid() const
  {
    return primitive != nullptr;
  }
  
  Double3 Normal() const;
};


#endif
