#pragma once

#include <embree3/rtcore.h>

#include "util.hxx"

class Scene;
class Spheres;
class Geometry;
class Mesh;
struct Ray;
struct SurfaceInteraction;
class Box;

struct RTCHit;

class EmbreeAccelerator
{
private:
  RTCDevice rtdevice = nullptr;
  RTCScene rtscene = nullptr;
  void FirstIntersectionTriangle(const RTCHit &rthit, const Ray &, SurfaceInteraction &intersection) const;
  void FirstIntersectionSphere(const RTCHit &rthit, const Ray &, SurfaceInteraction &intersection) const;
  static void SphereBoundsFunc(const RTCBoundsFunctionArguments*);
  static void SphereIntersectFunc(const RTCIntersectFunctionNArguments*);
public:
  EmbreeAccelerator();
  ~EmbreeAccelerator();
  void Add(Mesh &mesh);
  void Add(Spheres &spheres);
  void Add(Geometry &geo);
  void Build();
  bool FirstIntersection(const Ray &ray, double tnear, double &ray_length, SurfaceInteraction &intersection) const;
  Box GetSceneBounds() const;
};
