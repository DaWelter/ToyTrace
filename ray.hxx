#ifndef RAY_HXX
#define RAY_HXX

#include <limits>
#include "vec3f.hxx"

class Primitive;
class Shader;

struct Ray
{
  Ray() {}
  Ray(Double3 _org, Double3 _dir) : org(_org), dir(_dir) {}
  Double3 org; // origin
  Double3 dir; // direction
  
  auto PointAt(double t) const -> decltype(org + t * dir)
  {
    return org + t * dir;
  }
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
  
  auto EndPoint() const -> decltype(ray.PointAt(length))
  { 
    // Want to return some expression template construct. Not an actual Double3. To facilitate optimization.
    return ray.PointAt(length);
  }
};


inline std::ostream &operator<<(std::ostream &o,const Ray &ray)
{ o << "Ray[" << ray.org << "+t*" << ray.dir << "]"; return o; }


struct HitId
{
  const Primitive *primitive = { nullptr }; // primitive that was hit
  Double3 barry;
  
  operator bool() const
  {
    return primitive != nullptr;
  }
};


struct RaySurfaceIntersection
{
  HitId hitid;
  Double3 normal;        // Used for shading and tracing. Oriented toward the incomming ray.
  Double3 shading_normal;
  Double3 volume_normal; // Used to determine if entering or leaving a medium.
  Double3 pos;
  
  const Primitive& primitive() const;
  const Shader& shader() const;
  RaySurfaceIntersection(const HitId &hitid, const RaySegment &inbound);
  RaySurfaceIntersection() = default;
};


#endif
