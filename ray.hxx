#ifndef RAY_HXX
#define RAY_HXX

#include "vec3f.hxx"

class Primitive;

#define MaxRayDepth 3

class Ray
{
public:
  Ray() : hit(0),t(0.),level(0),barry(0) {}

  char level; // 0 = primary ray, 1,2,3... secondary ray
  
  Primitive *hit; // primitive that was hit
  Double3 barry;

  Double3 org; // origin
  Double3 dir; // direction
  double t;   // current/maximum hit distance
};

inline std::ostream &operator<<(std::ostream &o,const Ray &ray)
{ o << "Ray[" << ray.org << "+t*" << ray.dir << "]"; return o; }

#endif
