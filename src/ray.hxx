#pragma once

#include <limits>
#include <vector>

#include "types.hxx"
#include "vec3f.hxx"
//#include "scene.hxx"
#include "spectral.hxx"

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
  RaySegment(const Ray &ray_, double tnear, double tfar)
    : ray{ ray_ }, length{ tfar - tnear }
  {
    ray.org += tnear * ray.dir;
  }

  static RaySegment FromTo(const Double3 &src, const Double3 &dest);

  auto EndPoint() const -> decltype(ray.PointAt(length))
  { 
    // Want to return some expression template construct. Not an actual Double3. To facilitate optimization.
    return ray.PointAt(length);
  }
  
  RaySegment Reversed() const
  {
    return {{ray.org + length * ray.dir, -ray.dir}, length };
  }
};

inline void MoveOrg(RaySegment &seg, float t)
{
  seg.ray.org += seg.ray.dir;
  seg.length -= t;
}



inline std::ostream &operator<<(std::ostream &o,const Ray &ray)
{ o << "Ray[" << ray.org << "+t*" << ray.dir << "]"; return o; }

