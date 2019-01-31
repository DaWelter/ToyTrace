#pragma once

#include "scene.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "camera.hxx"
#include "util.hxx"
#include "shader_util.hxx"


class IterateIntersectionsBetween
{
#if 0
  RaySegment seg;
  double tnear, tfar;
#else
  const RaySegment original_seg;
  Double3 current_org;
  double current_dist;
#endif
  const Scene &scene;
public:
  IterateIntersectionsBetween(const RaySegment &seg, const Scene &scene)
    : original_seg{seg}, current_org{seg.ray.org}, current_dist{0.}, scene{scene}
  {}
  bool Next(RaySegment &seg, SurfaceInteraction &intersection);
  double GetT() const { return current_dist; }
};


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
  const Medium& getCurrentMedium() const;
};


inline const Medium& MediumTracker::getCurrentMedium() const
{
  assert(current);
  return *current;
}


inline void MediumTracker::goingThroughSurface(const Double3 &dir_of_travel, const SurfaceInteraction& intersection)
{
  if (Dot(dir_of_travel, intersection.geometry_normal) < 0)
    enterVolume(&GetMediumOf(intersection, scene));
  else
    leaveVolume(&GetMediumOf(intersection, scene));
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


// TODO: At least some of the data could be precomputed.
// TODO: For improved precision it would make sense to move the scene center to the origin.
struct EnvLightPointSamplingBeyondScene
{
  double diameter;
  double sufficiently_long_distance_to_go_outside_the_scene_bounds;
  Double3 box_center;
  
  EnvLightPointSamplingBeyondScene(const Scene &scene)
  {
    Box bb = scene.GetBoundingBox();
    box_center = 0.5*(bb.max+bb.min);
    diameter = Length(bb.max - bb.min);
    sufficiently_long_distance_to_go_outside_the_scene_bounds = 10.*diameter;
  }
  
  Double3 Sample(const Double3 &exitant_direction, Sampler &sampler) const
  {
    Double3 disc_sample = SampleTrafo::ToUniformDisc(sampler.UniformUnitSquare());
    Eigen::Matrix3d frame = OrthogonalSystemZAligned(exitant_direction);
    Double3 org = 
      box_center +
      -sufficiently_long_distance_to_go_outside_the_scene_bounds*exitant_direction
      + frame * 0.5 * diameter * disc_sample;
    return org;
  }
  
  double Pdf(const Double3 &) const
  {
    return 1./(Pi*0.25*Sqr(diameter));
  }
};



/* Ray is the ray to shoot. It must already include the anti-self-intersection offset.
  */
template<class VolumeHitVisitor, class SurfaceHitVisitor, class EscapeVisitor>
decltype(((EscapeVisitor*)nullptr)->operator()(Spectral3{}))  // Return type of the escape visitor
inline TrackToNextInteraction(
  const Scene &scene,
  const Ray &ray,
  const PathContext &context,
  Sampler &sampler,
  MediumTracker &medium_tracker,
  VolumePdfCoefficients *volume_pdf_coeff,
  SurfaceHitVisitor &&surface_visitor,
  VolumeHitVisitor &&volume_visitor,
  EscapeVisitor &&escape_visitor)
{
  static constexpr int MAX_ITERATIONS = 100; // For safety, in case somethign goes wrong with the intersections ...
  
  Spectral3 total_weight{1.};
  
  RaySegment segment;
  SurfaceInteraction intersection;
  IterateIntersectionsBetween iter{RaySegment{ray, LargeNumber}, scene};
  
  for (int interfaces_crossed=0; interfaces_crossed < MAX_ITERATIONS; ++interfaces_crossed)
  {
    double prev_t = iter.GetT();
    
    bool hit = iter.Next(segment, intersection);

    const Medium& medium = medium_tracker.getCurrentMedium();
    const auto medium_smpl = medium.SampleInteractionPoint(segment, sampler, context);
    total_weight *= medium_smpl.weight;
    
    const bool interacted_w_medium = medium_smpl.t < segment.length;
    const bool hit_invisible_wall =
      hit && 
      scene.GetMaterialOf(intersection.hitid).shader==&scene.GetInvisibleShader() &&
      scene.GetMaterialOf(intersection.hitid).emitter==nullptr;
    const bool cont = !interacted_w_medium && hit_invisible_wall;
    
    segment.length = std::min(segment.length, medium_smpl.t);
    
    if (volume_pdf_coeff != nullptr)
    {
      VolumePdfCoefficients local_coeff = medium_tracker.getCurrentMedium().ComputeVolumePdfCoefficients(segment, context);
      Accumulate(*volume_pdf_coeff, local_coeff, interfaces_crossed==0, !cont);
    }

    // If there was a medium interaction, it came before the next surface, and we can surely return.
    // Else it is possible to hit a perfectly transparent wall. I want to pass through it and keep on going.
    // If we hit something else then we surely want to return to process it further.
    // Finally there is the possibility of the photon escaping out of the scene to infinity.
    if (cont)
    {
      medium_tracker.goingThroughSurface(segment.ray.dir, intersection);
    }
    else
    {
      if (interacted_w_medium)
      {
        VolumeInteraction interaction{segment.EndPoint(), medium, Spectral3{0.}, medium_smpl.sigma_s};
        return volume_visitor(interaction, prev_t+medium_smpl.t, total_weight);
      }
      else if (hit)
      {
        return surface_visitor(intersection, iter.GetT(), total_weight);
      }
      break; // Escaped the scene.
    }
    assert (interfaces_crossed < MAX_ITERATIONS-1);
  }
  
  return escape_visitor(total_weight);
};


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
  return correction;
}


inline double DFactorOf(const SurfaceInteraction &intersection, const Double3 &exitant_dir)
{
  // By definition the factor is 1 for Volumes.
  return std::abs(Dot(intersection.geometry_normal, exitant_dir));
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
    if (node_count <= max_node_count)
    {
      return !RouletteTerminationAllowedAtLength(node_count) || RouletteSurvival(scatter_coeff, sampler);
    }
    else
      return false;
  }
  
private:
  bool RouletteTerminationAllowedAtLength(int n) const
  {
    return n > min_node_count;
  }
  
  bool RouletteSurvival(Spectral3 &coeff, Sampler &sampler) const
  {
    assert (coeff.maxCoeff() >= 0.);
    // Clip at 0.9 to make the path terminate in high-albedo scenes.
    double p_survive = std::min(0.9, coeff.maxCoeff());
    if (sampler.Uniform01() > p_survive)
      return false;
    coeff *= 1./p_survive;
    return true;
  }
};
