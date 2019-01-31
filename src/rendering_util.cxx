#include "rendering_util.hxx"

double UpperBoundToBoundingBoxDiameter(const Scene &scene)
{
  Box bb = scene.GetBoundingBox();
  double diameter = Length(bb.max - bb.min);
  return diameter;
}


bool IterateIntersectionsBetween::Next(RaySegment &seg, SurfaceInteraction &intersection)
{
#if 0
  tfar = this->seg.length;
  auto offset_tnear = 0; 
  // Not clear how to set the offset because the uncertainty tnear is known if it stems from a previous intersection.
  // This is because the accuracy of tfar as computed by Embree is unkown. Maybe it is as found in PBRT pg  234. But
  // still in order to compute that I would have to need all the data about the triangles and do a lot of redundant
  // computation, essentially having to implement my own ray-triangle intersection routine.
  // Moreover, Embree reports intersection where tfar<tnear!
  // Therefore this approach is not feasible.
  bool hit = scene.FirstIntersectionEmbree(this->seg.ray, offset_tnear, tfar, intersection);
  seg.ray = this->seg.ray;
  seg.ray.org += tnear*this->seg.ray.dir;
  seg.length = tfar - tnear;
  tnear = tfar;
  return hit;
#else
  // Embree does not appear to report intersections where tfar<0 though. 
  // So like in my previous approach, I let the origin jump to the last intersection point.
  seg.ray.org = this->current_org;
  seg.ray.dir = this->original_seg.ray.dir;
  seg.length = this->original_seg.length-current_dist;
  if (seg.length > 0)
  {
    bool hit = scene.FirstIntersectionEmbree(seg.ray, 0, seg.length, intersection);
    if (hit)
    {
      auto offset = AntiSelfIntersectionOffset(intersection, this->original_seg.ray.dir);
      this->current_org = intersection.pos + offset;
      // The distance taken so far is computed by projection of intersection point.
      this->current_dist = Dot(this->current_org-this->original_seg.ray.org, this->original_seg.ray.dir);
      return hit;
    }
    else
      this->current_dist = this->original_seg.length; // Done!
  }
  return false;
#endif
}


MediumTracker::MediumTracker(const Scene& _scene)
  : current{nullptr},
    media{}, // Note: Ctor zero-initializes the media array.
    scene{_scene}
{
}


void MediumTracker::initializePosition(const Double3& pos)
{
  std::fill(media.begin(), media.end(), nullptr);
  current = &scene.GetEmptySpaceMedium();
  {
    const Box bb = scene.GetBoundingBox();
    // The InBox check is important. Otherwise I would not know how long to make the ray.
    if (bb.InBox(pos))
    {
      double distance_to_go = 2. * (bb.max-bb.min).maxCoeff(); // Times to due to the diagonal.
      Double3 start = pos;
      start[0] += distance_to_go;
      IterateIntersectionsBetween iter{
        {{start, {-1., 0., 0.}}, distance_to_go}, scene};
      RaySegment seg;
      SurfaceInteraction intersection;
      while (iter.Next(seg, intersection))
      {
        goingThroughSurface(seg.ray.dir, intersection);
      }
    }
  }
}

