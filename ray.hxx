#ifndef RAY_HXX
#define RAY_HXX

#include <limits>
#include "vec3f.hxx"

class Primitive;

static constexpr int MAX_RAY_DEPTH = 10;

struct Ray
{
  Ray() {}
  Ray(Double3 _org, Double3 _dir) : org(_org), dir(_dir) {}
  Double3 org; // origin
  Double3 dir; // direction
};


struct RaySegment
{
  Ray ray;
  double length;
  
  RaySegment() : length(NaN) {}
  RaySegment(const Ray &_ray, double _length) : ray(_ray), length(_length) {}
  static RaySegment FromTo(const Double3 &src, const Double3 &dest) 
  {
    Double3 delta = dest-src;
    double l = Length(delta);
    delta = l>0 ? (delta / l).eval() : Double3(NaN, NaN, NaN);
    return RaySegment{{src, delta}, l};
  }
  
  auto EndPoint() const -> decltype(ray.org + length * ray.dir) 
  { 
    // Want to return some expression template construct. Not an actual Double3. To facilitate optimization.
    return ray.org + length * ray.dir; 
  }
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
  
  Double3 Normal(const Double3 &up_dir) const;
};


#endif
