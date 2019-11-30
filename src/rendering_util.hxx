#pragma once

#include "scene.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "camera.hxx"
#include "util.hxx"
#include "shader_util.hxx"


/* This thing tracks overlapping media volumes. Since it is a bit complicated
 * to physically correctly handle mixtures of media that would occur
 * in overlapping volumes, I take a simpler approach. This hands over the
 * medium with the highest associated priority. In case multiple volumes with the
 * same medium material overlap it will be as if there was the union of all of
 * those volumes.
 */
class MediumTracker
{
  static constexpr int MAX_INTERSECTING_MEDIA = 4;
  const Medium* current;
  std::array<const Medium*, MAX_INTERSECTING_MEDIA> media;
  const Scene &scene;
  void enterVolume(const Medium *medium);
  void leaveVolume(const Medium *medium);
  const Medium* findMediumOfHighestPriority() const;
  bool remove(const Medium *medium);
  bool insert(const Medium *medium);
public:
  explicit MediumTracker(const Scene &_scene);
  MediumTracker(const MediumTracker &_other) = default;
  void initializePosition(const Double3 &pos);
  void goingThroughSurface(const Double3 &dir_of_travel, const SurfaceInteraction &intersection);
  void goingThroughSurface(const Double3 &dir_of_travel, const BoundaryIntersection &intersection);
  const Medium& getCurrentMedium() const;
};


inline const Medium& MediumTracker::getCurrentMedium() const
{
  assert(current);
  return *current;
}


inline void MediumTracker::goingThroughSurface(const Double3 &dir_of_travel, const SurfaceInteraction& intersection)
{
  const Medium *m = scene.GetMaterialOf(intersection.hitid).medium;
  if (!m)
    return;
  if (Dot(dir_of_travel, intersection.geometry_normal) < 0)
    enterVolume(m);
  else
    leaveVolume(m);
}


inline void MediumTracker::goingThroughSurface(const Double3 &dir_of_travel, const BoundaryIntersection& intersection)
{
  const Medium *m = scene.GetMaterialOf(intersection.geom, intersection.prim).medium;
  if (!m)
    return;
  if (Dot(dir_of_travel.cast<float>(), intersection.n) < 0)
    enterVolume(m);
  else
    leaveVolume(m);
}


inline const Medium* MediumTracker::findMediumOfHighestPriority() const
{
  const Medium* medium_max_prio = &scene.GetEmptySpaceMedium();
  for (int i = 0; i < media.size(); ++i)
  {
    medium_max_prio = (media[i] &&  medium_max_prio->priority < media[i]->priority) ?
                        media[i] : medium_max_prio;
  }
  return medium_max_prio;
}


inline bool MediumTracker::remove(const Medium *medium)
{
  bool is_found = false;
  for (int i = 0; (i < media.size()) && !is_found; ++i)
  {
    is_found = media[i] == medium;
    media[i] = is_found ? nullptr : media[i];
  }
  return is_found;
}


inline bool MediumTracker::insert(const Medium *medium)
{
  bool is_place_empty = false;
  for (int i = 0; (i < media.size()) && !is_place_empty; ++i)
  {
    is_place_empty = media[i]==nullptr;
    media[i] = is_place_empty ? medium : media[i];
  }
  return is_place_empty;
}


inline void MediumTracker::enterVolume(const Medium* medium)
{
  // Set one of the array entries that is currently nullptr to the new medium pointer.
  // But only if there is room left. Otherwise the new medium is ignored.
  bool was_inserted = insert(medium);
  current = (was_inserted && medium->priority > current->priority) ? medium : current;
}


inline void MediumTracker::leaveVolume(const Medium* medium)
{
  // If the medium is in the media stack, remove it.
  // And also make the medium of highest prio the current one.
  remove(medium);
  if (medium == current)
  {
    current = findMediumOfHighestPriority();
  }
}

// For scattered rays, which can either be reflected or transmitted. But it is not clear which of them happened.
inline void MaybeGoingThroughSurface(MediumTracker &mt, const Double3& dir_of_travel, const SurfaceInteraction &surf)
{
    // Determine by geometry normal. Should always point outside of the volume.
    if(Dot(dir_of_travel, surf.normal) < 0.)
    {
      mt.goingThroughSurface(dir_of_travel, surf);
    }
}


#if 0
template<class Func>
inline bool ForEachVolumeSegments(const Scene &scene, const Ray &ray, double tnear, const double tfar, MediumTracker &medium_tracker, Func func)
{
  const Span<BoundaryIntersection> intersections = scene.IntersectionsWithVolumes(ray, tnear, tfar);
  for (const auto &intersection : intersections)
  {
    bool continue_ = func(tnear, intersection.t, medium_tracker.getCurrentMedium());
    if (!continue_)
      return false;

    medium_tracker.goingThroughSurface(ray.dir, intersection);
    tnear = intersection.t;
  }

  return func(tnear, tfar, medium_tracker.getCurrentMedium());
}
#endif


class SegmentIterator
{
  BoundaryIntersection const *start, *end;
  double tnear, tfar;
public:
  SegmentIterator(const Span<BoundaryIntersection> is, double tnear, double tfar)
    : start{ is.begin() }, end{ is.end() }, tnear{ tnear }, tfar{ tfar }
  {
  }

  operator bool() const
  {
    return tnear != tfar;
  }

  void Next(const Ray& ray, MediumTracker &medium_tracker)
  {
    assert(static_cast<float>(tnear) < static_cast<float>(tfar)); // Not done yet
    if (start == end)
    {
      // Mark as invalid.
      tnear = tfar;
    }
    else
    {
      medium_tracker.goingThroughSurface(ray.dir, *start);
      tnear = start->t;
      ++start;
    }
  }

  std::pair<double, double> Interval() const
  {
    return std::make_pair(tnear, (start==end ? tfar : start->t));
  }
};

inline SegmentIterator VolumeSegmentIterator(const Scene &scene, const Ray &ray, double tnear, double tfar)
{
  return SegmentIterator{ scene.IntersectionsWithVolumes(ray, tnear, tfar), tnear, tfar };
}



/* Ray is the ray to shoot. It must already include the anti-self-intersection offset.
  */
template<class VolumeHitVisitor, class SurfaceHitVisitor, class EscapeVisitor>
decltype(((EscapeVisitor*)nullptr)->operator()(Spectral3{}))  // Return type of the escape visitor
inline TrackToNextInteraction(
  const Scene &scene,
  const Ray &ray,
  const PathContext &context,
  const Spectral3 &initial_weight,
  Sampler &sampler,
  MediumTracker &medium_tracker,
  VolumePdfCoefficients *volume_pdf_coeff,
  SurfaceHitVisitor &&surface_visitor,
  VolumeHitVisitor &&volume_visitor,
  EscapeVisitor &&escape_visitor)
{
  Spectral3 weight{1.};
  
  double tfar = LargeNumber;
  const auto hit = scene.FirstIntersection(ray, 0., tfar);

  // The factor by which tfar is decreased is meant to prevent intersections which lie close to the end of the query segment.
  auto iter = VolumeSegmentIterator(scene, ray, 0., tfar * 0.9999);
  for (; iter; iter.Next(ray, medium_tracker))
  {
    auto[snear, sfar] = iter.Interval();
    const Medium& medium = medium_tracker.getCurrentMedium();
    RaySegment segment{ ray, snear, sfar };
    
    const auto medium_smpl = medium.SampleInteractionPoint(segment, weight*initial_weight, sampler, context);
    weight *= medium_smpl.weight;

    const bool interacted_w_medium = medium_smpl.t < segment.length;
    segment.length = interacted_w_medium ? medium_smpl.t : segment.length;

    if (volume_pdf_coeff != nullptr)
    {
      VolumePdfCoefficients local_coeff = medium.ComputeVolumePdfCoefficients(segment, context);
      Accumulate(*volume_pdf_coeff, local_coeff, snear == 0., interacted_w_medium);
    }

    if (interacted_w_medium)
    {
      tfar = snear + medium_smpl.t;
      VolumeInteraction vi{ ray.PointAt(tfar), medium, Spectral3{ 0. }, medium_smpl.sigma_s };
      return volume_visitor(vi, tfar, weight);
    }
  }

  // If we get here, there was no scattering in the medium.
  if (hit)
  {
    return surface_visitor(*hit, tfar, weight);
  }
  else
  {
    return escape_visitor(weight);
  }
};



template<class SurfaceHitVisitor, class EscapeVisitor, class SegmentVisitor>
inline void TrackBeam(
  const Scene &scene,
  const Ray &ray,
  const PathContext &context,
  Sampler &sampler,
  MediumTracker &medium_tracker,
  SurfaceHitVisitor &&surface_visitor,
  SegmentVisitor &&segment_visitor,
  EscapeVisitor &&escape_visitor)
{
  Spectral3 weight{1.};
  
  double tfar = LargeNumber;
  const auto hit = scene.FirstIntersection(ray, 0., tfar);

  PiecewiseConstantTransmittance pct;
  
  // TODO: if the active medium does not change at a boundary, we can aggregate the adjacent segments.
  auto iter = VolumeSegmentIterator(scene, ray, 0., tfar);
  for (; iter; iter.Next(ray, medium_tracker))
  {
    const auto[snear, sfar] = iter.Interval();
    const RaySegment segment{ ray, snear, sfar };
    const auto &medium = medium_tracker.getCurrentMedium();
    
    medium.ConstructShortBeamTransmittance(segment, sampler, context, pct);
    
    segment_visitor(segment, medium, pct, weight);
    
    weight *= pct(segment.length);
    pct.Clear();
    
    if (weight.maxCoeff() <= 0.)
      break;
  }

  if (hit)
  {
    surface_visitor(*hit, weight);
  }
  else
  {
    escape_visitor(weight);
  }
};



Spectral3 TransmittanceEstimate(const Scene &scene, RaySegment seg, MediumTracker &medium_tracker, 
                                const PathContext &context, Sampler &sampler, 
                                VolumePdfCoefficients *volume_pdf_coeff = nullptr);


/* Straight forwardly following Cpt 5.3 Veach's thesis.
  * Basically when multiplied, it turns fs(wi -> wo, Nshd) into the corrected BSDF fs(wi->wo, Nsh)*Dot(Nshd,wi)/Dot(N,wi),
  * where wi refers to the incident direction of light as per Eq. 5.17.
  * 
  * Except that my conventions are different. In/out directions refer to the random walk. */
inline double BsdfCorrectionFactor(const Double3 &reverse_incident_dir, const SurfaceInteraction &intersection, const Double3 &exitant_dir, TransportType transport)
{
  const Double3 &light_incident = transport==RADIANCE ? exitant_dir : reverse_incident_dir;
  const double correction =  std::abs(Dot(intersection.shading_normal, light_incident))/
                            (std::abs(Dot(intersection.normal, light_incident))+Epsilon);
  return std::min(correction, 10.); // Don't allow weights to grow arbitrarily large
}


inline double BsdfCorrectionFactorPBRT(const Double3 &reverse_incident_dir, const SurfaceInteraction &intersection, const Double3 &exitant_dir, double clamp)
{
  double nom = std::abs(Dot(intersection.shading_normal, reverse_incident_dir)) * std::abs(Dot(intersection.normal, exitant_dir));
  //           '-------------------------^------------------------------------'   '------------------^--------------------------'
  //                                From shading correction.                                  The D-factor, which uses geometry normal. Often part of geometry factor.
  double denom = std::abs(Dot(intersection.normal, reverse_incident_dir)) * std::abs(Dot(intersection.shading_normal, exitant_dir));
  //           '-------------------------^-------------------------------------'   '------------------^-------------------------'
  //                                From shading correction                                   Cancel out "fake" D-factor, which uses the shading normal.
  //                                                                                          Note the fake D-factor should be multiplied separately to cancel
  //                                                                                          the corresponding 1/wr.ns term of specular BSDFs. (*)
  //  (*)  This is the important difference to my non-PBRT style correction. Here we cancel wr.ns of the specular BSDF exactly, whereas the other correction
  //  routine can let wr.ng / wr.ns grow a lot. Here we have  wr.ns / wr.ns * clamp(wr.ns * ....).
  //
  //  One more note: For photon mapping, Veach suggests to split particles. See pg. 160. 
  return std::min(nom/(denom + Epsilon), clamp);
}

inline double DFactorPBRT(const SurfaceInteraction &intersection, const Double3 &exitant_dir)
{
  // By definition the factor is 1 for Volumes.
  // This one uses the shading normal. Use with PBRT style shading normal correction!
  return std::abs(Dot(intersection.shading_normal, exitant_dir));
}


class RayTermination
{
public:
  int min_node_count = 2;
  int max_node_count = 10;

  RayTermination(const RenderingParameters &params)
  {
    max_node_count = params.max_ray_depth;
  }
  
  bool SurvivalAtNthScatterNode(Spectral3 &scatter_coeff, int node_count, Sampler &sampler) const
  {
    return SurvivalAtNthScatterNode(scatter_coeff, scatter_coeff, node_count, sampler);
  }
  
  bool SurvivalAtNthScatterNode(Spectral3 &weight, const Spectral3 &scatter_coeff, int node_count, Sampler &sampler) const
  {
    if (node_count <= max_node_count)
    {
      if (node_count <= min_node_count)
      {
        return true;
      }
      //return RouletteSurvival(weight, scatter_coeff, sampler);
      return true;
    }
    // Else
    return false;
  }
/*  
  bool TerminateDueToNodeCount(int node_count) const
  {
    return node_count > max_node_count;
  }*/
  
private: 
  bool RouletteSurvival(Spectral3 &weight, const Spectral3 &coeff, Sampler &sampler) const
  {
    assert (coeff.maxCoeff() >= 0.);
    // Clip at 0.9 to make the path terminate in high-albedo scenes.
    double p_survive = std::min(0.9, coeff.maxCoeff());
    if (sampler.Uniform01() > p_survive)
      return false;
    weight *= 1./p_survive;
    return true;
  }
};
