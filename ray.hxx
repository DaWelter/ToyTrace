#pragma once

#include <limits>
#include <vector>

#include "types.hxx"
#include "vec3f.hxx"
#include "scene.hxx"


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
  Float2 tex_coord;
  
  SurfaceInteraction(const HitId &hitid);
  SurfaceInteraction() = default;
};

struct RaySurfaceIntersection : public SurfaceInteraction
{
  Double3 normal;    // Geometry normal, oriented toward the incomming ray, if result of ray-surface intersection.
  Double3 shading_normal; // Same for smooth normal.
  
  RaySurfaceIntersection(const HitId &hitid, const RaySegment &_incident_segment);
  RaySurfaceIntersection() = default;
  void SetOrientedNormals(const Double3 &incident);
};


struct VolumeInteraction : public InteractionPoint
{
  const Medium *_medium = nullptr;
  VolumeInteraction() = default;
  VolumeInteraction(const Double3 &_position, const Medium &_medium)
    : InteractionPoint{_position}, _medium{&_medium}
    {}
    
  const Medium& medium() const { return *_medium; }
};


inline Double3 AntiSelfIntersectionOffset(const SurfaceInteraction &interaction, const Double3 &exitant_dir)
{
  static constexpr float EXPERIMENTALLY_DETERMINED_MAGIC_NUMBER = 512.f;
  const auto pos = interaction.pos.cast<float>();
  const auto normal = interaction.geometry_normal;
  float val = pos.cwiseAbs().maxCoeff();
  float eps = EXPERIMENTALLY_DETERMINED_MAGIC_NUMBER*
              val*std::numeric_limits<float>::epsilon();
  assert(eps > 0.f);
  return eps * (Dot(exitant_dir, normal) > 0. ? 1. : -1.) * normal;
}

