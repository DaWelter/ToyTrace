#pragma once


#include <optional>

#include "util.hxx"
#include "span.hxx"
#include "primitive.hxx"

class Scene;
class Spheres;
class Geometry;
class Mesh;
struct Ray;
struct SurfaceInteraction;
class Box;

struct RTCHit;
struct RTDevice_;
struct RTScene_;
typedef struct RTCDeviceTy* RTCDevice;
typedef struct RTCSceneTy* RTCScene;
struct RTCBoundsFunctionArguments;
struct RTCIntersectFunctionNArguments;
struct RTCOccludedFunctionNArguments;

struct BoundaryIntersection
{
  scene_index_t geom;
  scene_index_t prim;
  float t;
  Float3 n;
};

static_assert(sizeof(BoundaryIntersection) == 24);


class EmbreeAccelerator
{
private:
  RTCDevice rtdevice = nullptr;
  RTCScene rtscene = nullptr;
  static thread_local ToyVector<BoundaryIntersection> intersections_result;

  void FirstIntersectionTriangle(const RTCHit &rthit, const Ray &, SurfaceInteraction &intersection) const;
  void FirstIntersectionSphere(const RTCHit &rthit, const Ray &, SurfaceInteraction &intersection) const;
  static void SphereBoundsFunc(const RTCBoundsFunctionArguments*);
  static void SphereIntersectFunc(const RTCIntersectFunctionNArguments*);
  static void SphereOccludedFunc(const RTCOccludedFunctionNArguments*);
public:
  EmbreeAccelerator();
  ~EmbreeAccelerator();
  void InsertRefTo(Mesh &mesh);
  void InsertRefTo(Spheres &spheres);
  void InsertRefTo(Geometry &geo);
  void Build(bool enable_intersections_in_order_call = false);
  bool FirstIntersection(const Ray &ray, double tnear, double &ray_length, SurfaceInteraction &intersection) const;
  // Note: Calling this twice will invalidate the array from the first call!
  Span<BoundaryIntersection> IntersectionsInOrder(const Ray &ray, double tnear, double tfar) const;
  bool IsOccluded(const Ray &ray, double tnear, double tfar) const;
  Box GetSceneBounds() const;
};


namespace EmbreeAcceleratorDetail
{
template<class IterT, class Less, class Equal>
std::pair<IterT, bool> FindPlaceIfUnique(IterT begin, IterT end, const typename IterT::value_type &item, Less less, Equal equal)
{
  const auto backup = end;
  while (end != begin)
  {
    --end; // Can decrement because not at the beginning.
    if (less(item, *end))
      continue;

    const bool eq = equal(*end, item);
      
    return { end + (eq ? 0 : 1), eq };
  }

  return { end, false };
}
}
