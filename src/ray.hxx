#pragma once

#include <limits>
#include <vector>

#include "types.hxx"
#include "vec3f.hxx"
#include "scene.hxx"
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



struct InteractionPoint
{
  Double3 pos;
};


struct SurfaceInteraction : public InteractionPoint
{
  HitId hitid;
  Double3 geometry_normal;
  Double3 smooth_normal;
  Double3 normal;    // Geometry normal, oriented toward the incomming ray, if result of ray-surface intersection.
  Double3 shading_normal; // Same for smooth normal.
  Float2 tex_coord;
  Float3 pos_bounds { 0. }; // Bounds within which the true hitpoint (computed without roundoff errors) lies. See PBRT chapt 3.

  SurfaceInteraction(const HitId &hitid, const RaySegment &_incident_segment);
  SurfaceInteraction(const HitId &hitid);
  SurfaceInteraction() = default;
  void SetOrientedNormals(const Double3 &incident);
};


struct VolumeInteraction : public InteractionPoint
{
  const Medium *_medium = nullptr;
  Spectral3 radiance;
  Spectral3 sigma_s; // Scattering coefficient. Use in evaluate functions and scatter sampling. Kernel defined as phase function times sigma_s. 
  VolumeInteraction() = default;
  VolumeInteraction(const Double3 &_position, const Medium &_medium, const Spectral3 &radiance_, const Spectral3 &sigma_s_)
    : InteractionPoint{_position}, _medium{&_medium}, radiance{radiance_}, sigma_s{sigma_s_}
    {}
  const Medium& medium() const { return *_medium; }
};


Double3 AntiSelfIntersectionOffset(const SurfaceInteraction &interaction, const Double3 &exitant_dir);