#pragma once

#include <fstream>
#include <cstring> // memcopy, yes memcopy
#include <type_traits>

#include "scene.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "renderbuffer.hxx"

#pragma GCC diagnostic warning "-Wunused-parameter" 
#pragma GCC diagnostic warning "-Wunused-variable"

using  AlgorithmParameters = RenderingParameters;
namespace ROI = RadianceOrImportance;



inline static Double3 Position(const Medium::InteractionSample &s, const RaySegment &which_is_a_point_on_this_segment)
{
  return which_is_a_point_on_this_segment.ray.PointAt(s.t);
}

double UpperBoundToBoundingBoxDiameter(const Scene &scene);



// WARNING: THIS IS NOT THREAD SAFE. DON'T WRITE TO THE SAME FILE FROM MULTIPLE THREADS!
class PathLogger
{
  std::ofstream file;
  int max_num_paths;
  int num_paths_written;
  int total_path_index;
  void PreventLogFromGrowingTooMuch();
public:
  enum SegmentType : int {
    EYE_SEGMENT = 'e',
    LIGHT_PATH = 'l',
    EYE_LIGHT_CONNECTION = 'c',
  };
  enum ScatterType : int {
    SCATTER_VOLUME = 'v',
    SCATTER_SURFACE = 's'
  };
  PathLogger();
  void AddSegment(const Double3 &x1, const Double3 &x2, const Spectral3 &beta_at_end_before_scatter, SegmentType type);
  void AddScatterEvent(const Double3 &pos, const Double3 &out_dir, const Spectral3 &beta_after, ScatterType type);
  void NewTrace(const Spectral3 &beta_init);
};
#ifndef NDEBUG
#  define PATH_LOGGING(x)
#else
#  define PATH_LOGGING(x)
#endif


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
  void initializePosition(const Double3 &pos, IntersectionCalculator &intersector);
  void initializePosition(const Double3 &pos, IntersectionCalculator &&intersector)
  { // Take control of the temporary that was passed in here. Now it has a name and
    // therefore repeated calling of initializePosition will invoke the one with the
    // lvalue in the parameter list.
    initializePosition(pos, intersector);
  }
  void goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection &intersection);
  const Medium& getCurrentMedium() const;
};


inline const Medium& MediumTracker::getCurrentMedium() const
{
  assert(current);
  return *current;
}


inline void MediumTracker::goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection& intersection)
{
  if (Dot(dir_of_travel, intersection.geometry_normal) < 0)
    enterVolume(intersection.primitive().medium);
  else
    leaveVolume(intersection.primitive().medium);
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


struct LightPicker
{
  const Scene &scene;
  Sampler &sampler;
  static constexpr int IDX_PROB_ENV = 0;
  static constexpr int IDX_PROB_AREA = 1;
  static constexpr int IDX_PROB_POINT = 2;
  std::array<double, 3> emitter_type_selection_probabilities;
  
  LightPicker(const Scene &_scene, Sampler &_sampler) : scene{_scene}, sampler{_sampler}
  {
    const double nl = scene.GetNumLights();
    const double ne = scene.GetNumEnvLights();
    const double na = scene.GetNumAreaLights();
    const double prob_pick_env_light = ne/(nl+ne+na);
    const double prob_pick_area_light = na/(nl+ne+na);
//     std::copy(
//       emitter_type_selection_probabilities.begin(),
//       emitter_type_selection_probabilities.end(),
//               {{ prob_pick_env_light, prob_pick_area_light, 1.-prob_pick_area_light-prob_pick_env_light }};
    emitter_type_selection_probabilities = { prob_pick_env_light, prob_pick_area_light, 1.-prob_pick_area_light-prob_pick_env_light };
  }
  
  using LightPick = std::tuple<const ROI::PointEmitter*, double>;
  using AreaLightPick = std::tuple<const Primitive*, const ROI::AreaEmitter*, double>;
  
  LightPick PickLight()
  {
    const int nl = scene.GetNumLights();
    assert(nl > 0);
    int idx = sampler.UniformInt(0, nl-1);
    const ROI::PointEmitter *light = &scene.GetLight(idx);
    double pmf_of_light = 1./nl;
    return LightPick{light, pmf_of_light};
  }
  
  double PmfOfLight(const ROI::PointEmitter &) const
  {
    return 1./scene.GetNumLights();
  }
  
  AreaLightPick PickAreaLight()
  {
    const int na = scene.GetNumAreaLights();
    assert (na > 0);
    const Primitive* prim;
    const ROI::AreaEmitter* emitter;
    std::tie(prim, emitter) = scene.GetAreaLight(sampler.UniformInt(0, na-1));
    double pmf_of_light = 1./na;
    return AreaLightPick{prim, emitter, pmf_of_light}; 
  }

  double PmfOfLight(const ROI::AreaEmitter &) const
  {
    return 1./scene.GetNumAreaLights();
  }
};


  



namespace RandomWalk
{
namespace RW = RandomWalk;
  
struct EnvLightPointSamplingBeyondScene
{
  const double diameter = NaN;
  const double sufficiently_long_distance_to_go_outside_the_scene_bounds = NaN;
  
  EnvLightPointSamplingBeyondScene(const Scene &_scene) 
    : diameter{UpperBoundToBoundingBoxDiameter(_scene)},
      sufficiently_long_distance_to_go_outside_the_scene_bounds{10.*diameter}
  {
  }
  
  Double3 Sample(const Double3 &exitant_direction, Sampler &sampler) const
  {
    Double3 disc_sample = SampleTrafo::ToUniformDisc(sampler.UniformUnitSquare());
    Eigen::Matrix3d frame = OrthogonalSystemZAligned(exitant_direction);
    Double3 org = 
      -sufficiently_long_distance_to_go_outside_the_scene_bounds*exitant_direction
      + frame * 0.5 * diameter * disc_sample;
    return org;
  }
  
  double Pdf(const Double3 &) const
  {
    return 1./(Pi*0.25*Sqr(diameter));
  }
};


// Be warned there be dragons and worse.

enum class NodeType : char 
{
  SCATTER,
  ENV,
  AREA_LIGHT,
  POINT_LIGHT,
  CAMERA,
  ZERO_CONTRIBUTION_ABORT_WALK
};

enum class GeometryType : char
{
  OTHER,
  SURFACE
};


struct PointEmitterInteraction : public InteractionPoint
{
  PointEmitterInteraction(const Double3 &pos, const ROI::PointEmitter &_e) : InteractionPoint{pos}, emitter(&_e) {}
  PointEmitterInteraction(const PointEmitterInteraction &) = default;
  PointEmitterInteraction& operator=(const PointEmitterInteraction &) = default;
  const ROI::PointEmitter *emitter;
};


struct PointEmitterArrayInteraction : public InteractionPoint
{
  PointEmitterArrayInteraction(const Double3 &pos, int _u, const ROI::PointEmitterArray &_e) : InteractionPoint{pos}, unit_index{_u}, emitter{&_e} {}
  PointEmitterArrayInteraction(const PointEmitterArrayInteraction &) = default;
  PointEmitterArrayInteraction& operator=(const PointEmitterArrayInteraction &) = default;
  int unit_index;
  const ROI::PointEmitterArray *emitter;
};


struct EnvironmentalRadianceFieldInteraction : public InteractionPoint
{
  EnvironmentalRadianceFieldInteraction(const Double3 &pos, const ROI::EnvironmentalRadianceField &_e) : InteractionPoint{pos}, emitter{&_e}, radiance{NaN} {}
  EnvironmentalRadianceFieldInteraction(const EnvironmentalRadianceFieldInteraction &) = default;
  EnvironmentalRadianceFieldInteraction& operator=(const EnvironmentalRadianceFieldInteraction &) = default;
  const ROI::EnvironmentalRadianceField *emitter;
  // What on earth is that? 
  // Well, the sequence of events is either a)
  // 1) Sample coodinate on light, which happens to be the direction for env lights.
  // 2) Evaluate outgoing radiance into direction towards the end of the other path.
  // or b)
  // 1) Generate coordinate on light from ray hit point.
  // 2) Evalute outgoing radiance into (reversed) direction of incident ray.
  // Now, my messy framework cannot decide in step 2) which kind of step 1) was taken. So
  // I store the radiance in step 1). This works because in 1) the direction is known.
  // The difference between a1) and b1) is that in a1), samples can be generated from delta-distributions
  // whereas in b) I can only return radiance from continuous densities.
  Spectral3 radiance;
};


// TODO: C++ wants to force default initialization/construction on me. Okay, so I should write constructors for the various node types.
struct PathNode
{ 
  NodeType node_type {NodeType::ZERO_CONTRIBUTION_ABORT_WALK};
  GeometryType geom_type { GeometryType::OTHER };
  Double3 incident_dir {0.};
  union Interactions
  {
    InteractionPoint point;
    RaySurfaceIntersection ray_surface;
    VolumeInteraction volume;
    PointEmitterInteraction emitter_point;
    PointEmitterArrayInteraction emitter_point_array;
    EnvironmentalRadianceFieldInteraction emitter_env;
    SurfaceInteraction emitter_area;
    Interactions() : point {Double3{0.}} {}
  };
  Interactions interaction;
  
  PathNode() = default;
  
  // Pretty sure all this stuff is trivially_copyable ...
  PathNode(const PathNode &other)
  {
    std::memcpy(this, &other, sizeof(decltype(*this)));
  }
  
  PathNode& operator=(const PathNode &other)
  {
    if (&other != this)
    {
      std::memcpy(this, &other, sizeof(decltype(*this)));
    }
    return *this;
  }
  
  Double3 geometry_normal() const
  {
    assert(geom_type == GeometryType::SURFACE);
    assert(node_type == NodeType::AREA_LIGHT ||
            node_type == NodeType::SCATTER);
    static_assert(std::is_base_of<SurfaceInteraction, RaySurfaceIntersection>::value, "Must share fields");
    return interaction.emitter_area.geometry_normal;
  }
  
  Double3 coordinates() const
  {
    static_assert(std::is_base_of<InteractionPoint, RaySurfaceIntersection>::value, "Must share fields");
    static_assert(std::is_base_of<InteractionPoint, VolumeInteraction>::value, "Must share fields");
    static_assert(std::is_base_of<InteractionPoint, PointEmitterInteraction>::value, "Must share fields");
    static_assert(std::is_base_of<InteractionPoint, PointEmitterArrayInteraction>::value, "Must share fields");
    static_assert(std::is_base_of<InteractionPoint, EnvironmentalRadianceFieldInteraction>::value, "Must share fields");
    static_assert(std::is_base_of<InteractionPoint, SurfaceInteraction>::value, "Must share fields");
    return interaction.point.pos;
  };
};




class RadianceEstimatorBase
{
protected:
  Sampler sampler;
  IntersectionCalculator intersector;
  const Scene &scene;
  double sufficiently_long_distance_to_go_outside_the_scene_bounds;
  int max_path_node_count;
private:
  ROI::TotalEnvironmentalRadianceField envlight;
protected:
  LightPicker light_picker;
  //ROI::PointEmitterArray::Response sensor_connection_response;
  int sensor_connection_unit = {};
public:
  RadianceEstimatorBase(const Scene &_scene, const AlgorithmParameters &algo_params = AlgorithmParameters{})
    : sampler{}, intersector{_scene.MakeIntersectionCalculator()}, scene{_scene},     
      sufficiently_long_distance_to_go_outside_the_scene_bounds{10.*UpperBoundToBoundingBoxDiameter(_scene)},
      max_path_node_count{algo_params.max_ray_depth+1},
      envlight{_scene}, light_picker{_scene, sampler}
  {

  }

  virtual ~RadianceEstimatorBase() {}

  

  struct StepResult
  {
    RW::PathNode node;
    RaySegment segment; // Leading to the node
    Pdf scatter_pdf; // Of existant coordinate. For envlight it is w.r.t. area. Else w.r.t solid angle.
    Spectral3 beta_factor {0.};
  };
  
  
  struct WeightedSegment
  {
    RaySegment segment {};
    Spectral3 weight { 0. };
  };
  

  StepResult TakeRandomWalkStep(const RW::PathNode &source_node, MediumTracker &medium_tracker, const PathContext &context)
  {
    struct TagRaySample {};
    using RaySample = Sample<Ray, Spectral3, TagRaySample>;

    auto SampleExitantRayDefault = [&]() -> RaySample
    {
      auto smpl = SampleScatterCoordinate(source_node, context);
      Ray ray{source_node.coordinates(), smpl.coordinates};
      smpl.value *= DFactorOf(source_node, ray.dir);
      AddAntiSelfIntersectionOffsetAt(ray.org, source_node, ray.dir);
      return RaySample{ray, smpl.value, smpl.pdf_or_pmf};
    };
       
    auto SampleExitantRayEnv = [&]() -> RaySample
    {
      auto smpl = SampleScatterCoordinateOfEmitter(source_node, context);
      return RaySample{{smpl.coordinates, source_node.coordinates()}, smpl.value, smpl.pdf_or_pmf};
    };
    
    RaySample ray_sample = (source_node.node_type != RW::NodeType::ENV) ?
      SampleExitantRayDefault() :
      SampleExitantRayEnv();
    const Ray &ray = ray_sample.coordinates;

    StepResult node_sample;
    node_sample.beta_factor = ray_sample.value / PmfOrPdfValue(ray_sample);

    if (node_sample.beta_factor.isZero()) // Because of odd situations where ray can be scattered below the surface due to smooth normal interpolation.
    {
      assert(node_sample.node.node_type == RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK); // TODO Write test to check that the node is default constructed as required here.
      return node_sample;
    }
    
    MaybeHandlePassageThroughSurface(source_node, ray.dir, medium_tracker);
    
    auto collision = CollisionData{ray};
    TrackToNextInteraction(collision, medium_tracker, context);
    
    node_sample.node = AllocateNode(collision, medium_tracker, context);
    node_sample.beta_factor *= collision.smpl.weight; 
    node_sample.scatter_pdf = ray_sample.pdf_or_pmf;
    node_sample.segment = collision.segment;
    assert(node_sample.scatter_pdf > 0.); // Since I rolled that sample it should have non-zero probability of being generated.
    return node_sample;
  }


  WeightedSegment Connection(const RW::PathNode &source_node, const MediumTracker &source_medium_tracker, const RW::PathNode &target_node, const PathContext &context, double *pdf_source, double *pdf_target)
  {
    WeightedSegment result{};
    if (source_node.node_type == RW::NodeType::ENV && target_node.node_type == RW::NodeType::ENV)
      return result;

    RaySegment &segment_to_target = result.segment = CalculateSegmentToTarget(source_node, target_node);

    Spectral3 scatter_factor = Evaluate(source_node, target_node, segment_to_target, context, pdf_source);
    Spectral3 target_scatter_factor = Evaluate(target_node, source_node, segment_to_target.Reversed(), context, pdf_target);

    if (scatter_factor.isZero() || target_scatter_factor.isZero())
      return result;

    MediumTracker medium_tracker = source_medium_tracker;
    MaybeHandlePassageThroughSurface(source_node, segment_to_target.ray.dir, medium_tracker);
    
    segment_to_target.ShortenBothEndsBy(RAY_EPSILON);
    
    auto transmittance = TransmittanceEstimate(segment_to_target, medium_tracker, context);

    bool is_wrt_solid_angle = source_node.node_type == RW::NodeType::ENV || target_node.node_type == RW::NodeType::ENV;
    double r2_factor = is_wrt_solid_angle ? 1. : 1./Sqr(segment_to_target.length);
    
    result.weight = r2_factor * transmittance * scatter_factor * target_scatter_factor;
    result.segment = segment_to_target;
    return result;
  }

  
  // Source node sends radiance/importance to target node. How much? TODO: remove target node parameter?!
  inline Spectral3 Evaluate(const RW::PathNode &source_node, const RW::PathNode &, const RaySegment &segment_to_target, const PathContext &context, double *source_scatter_pdf)
  {
    assert(source_node.node_type != RW::NodeType::ENV || (segment_to_target.ray.dir == source_node.coordinates()));
    Spectral3 scatter_value = EvaluateScatterCoordinate(source_node, segment_to_target.ray.dir, context, source_scatter_pdf);
    scatter_value *= DFactorOf(source_node, segment_to_target.ray.dir);
    return scatter_value;
  }

  // Convert the native pdf of sampling the scatter coordinate of  the source into the pdf to represent the location of the target node.
  // It is important in MIS weighting to express the pdf of various sampling 
  // strategies w.r.t. to the same integration domain. E.g. Solid angle or area.
  // However it should be okay to compose the joint pdfs of paths using different sub-domains, e.g.:
  // pdf(path) = pdf_w1(w1)*pdf_a2(x2)*pdf_a3(x3)*... This is cool as long as the product space is
  // consistently used.
  // Ref: Veach's Thesis and PBRT book.
  double PdfConversionFactorForTarget(const RW::PathNode &source, const RW::PathNode &target, const RaySegment &segment, bool is_parallel_beam) const
  {
    double factor = DFactorOf(target, segment.ray.dir);
    if (source.node_type != NodeType::ENV && target.node_type != NodeType::ENV && !is_parallel_beam)
    {
      // If source is env: I already generate points on a surface. Thus the conversion from angle to surface is only needed for other types.
      // If target is env: By definition, solid angle is used to describe the coordinate of the final node.
      factor /= (Sqr(segment.length)+Epsilon);
    }
    return factor;
  }
  
    
  void MaybeHandlePassageThroughSurface(const RW::PathNode &node, const Double3 &dir_of_travel, MediumTracker &medium_tracker) const
  {
    if (node.geom_type == RW::GeometryType::SURFACE && 
        node.node_type == RW::NodeType::SCATTER && 
        Dot(dir_of_travel, node.interaction.ray_surface.normal) < 0.)
    {
      // By definition, intersection.normal points to where the intersection ray is comming from.
      // Thus we can determine if the sampled direction goes through the surface by looking
      // if the direction goes in the opposite direction of the normal.
      medium_tracker.goingThroughSurface(dir_of_travel, node.interaction.ray_surface);
    }
  }

  
  void InitializeMediumTracker(const RW::PathNode &source_node, MediumTracker &medium_tracker) 
  {
    assert (source_node.node_type != RW::NodeType::SCATTER);
    medium_tracker.initializePosition(
      source_node.node_type != RW::NodeType::ENV ? source_node.coordinates() : Double3{Infinity},
      intersector);
  }
  
  
  struct CollisionData
  {
    CollisionData (const Ray &ray) :
      smpl{},
      segment{ray, LargeNumber},
      hit{}
    {}
    Medium::InteractionSample smpl;
    RaySegment segment;
    HitId hit;
  };
  
  
  RW::PathNode AllocateNode(const CollisionData &collision, MediumTracker &medium_tracker, const PathContext &context) const
  {
    RW::PathNode node;
    node.incident_dir = collision.segment.ray.dir;
    if (collision.smpl.t < collision.segment.length)
    {
      node.node_type = RW::NodeType::SCATTER;
      node.geom_type = RW::GeometryType::OTHER;
      node.interaction.volume = VolumeInteraction{
        Position(collision.smpl, collision.segment),
        medium_tracker.getCurrentMedium()
      };
    }
    else if (collision.hit)
    {
      node.geom_type   = RW::GeometryType::SURFACE;
      if (collision.hit.primitive->emitter)
      {
        node.node_type = RW::NodeType::AREA_LIGHT;
        node.interaction.emitter_area = SurfaceInteraction{collision.hit};
      }
      else 
      {
        node.node_type = RW::NodeType::SCATTER;
        node.interaction.ray_surface = RaySurfaceIntersection{collision.hit, collision.segment};
      }
    }
    else
    {
      const auto &emitter = this->GetEnvLight();
      node.node_type = RW::NodeType::ENV;
      node.geom_type = RW::GeometryType::OTHER;
      node.interaction.emitter_env = RW::EnvironmentalRadianceFieldInteraction{
        -collision.segment.ray.dir,
        emitter
      };
      node.interaction.emitter_env.radiance = emitter.Evaluate(-collision.segment.ray.dir, context);
    }
    return node;
  }
  
  
  RW::PathNode SampleEmissiveEnd(const PathContext &context, Pdf &pdf)
  {
    RW::PathNode node;
    int which_kind = TowerSampling<3>(light_picker.emitter_type_selection_probabilities.data(), sampler.Uniform01());
    if (which_kind == LightPicker::IDX_PROB_ENV)
    {
      const auto &emitter = this->GetEnvLight();
      auto smpl = emitter.TakeDirectionSample(sampler, context);
      node.node_type = RW::NodeType::ENV;
      node.geom_type = RW::GeometryType::OTHER;
      node.interaction.emitter_env = RW::EnvironmentalRadianceFieldInteraction{
        smpl.coordinates,
        emitter
      };
      node.interaction.emitter_env.radiance = smpl.value;
      pdf = smpl.pdf_or_pmf;
    }
    else if(which_kind == LightPicker::IDX_PROB_AREA)
    {
      const Primitive* primitive;
      const ROI::AreaEmitter* emitter;
      double pmf_of_light;
      std::tie(primitive, emitter, pmf_of_light) = light_picker.PickAreaLight();
      
      auto smpl = emitter->TakeAreaSample(*primitive, sampler, context);
      node.node_type   = RW::NodeType::AREA_LIGHT;
      node.geom_type   = RW::GeometryType::SURFACE;
      node.interaction.emitter_area = SurfaceInteraction{
        smpl.coordinates
      };
      pdf = pmf_of_light * smpl.pdf_or_pmf;
    }
    else
    {
      const ROI::PointEmitter* light; 
      double pmf_of_light;
      std::tie(light, pmf_of_light) = light_picker.PickLight();
      node.node_type = RW::NodeType::POINT_LIGHT;
      node.geom_type = RW::GeometryType::OTHER;
      node.interaction.emitter_point = RW::PointEmitterInteraction{
        light->Position(),
        *light
      };
      pdf = Pdf::MakeFromDelta(pmf_of_light);
    }
    pdf *= light_picker.emitter_type_selection_probabilities[which_kind];
    assert (std::isfinite(pdf));
    return node;
  }
  

  RW::PathNode SampleSensorEnd(int unit_index, const PathContext &context, Pdf &pdf)
  {
    RW::PathNode node;
    const ROI::PointEmitterArray& camera = scene.GetCamera();
    auto smpl = camera.TakePositionSample(unit_index, sampler, context);
    node.geom_type   = RW::GeometryType::OTHER;
    node.node_type   = RW::NodeType::CAMERA;
    node.interaction.emitter_point_array = RW::PointEmitterArrayInteraction{
      smpl.coordinates,
      unit_index,
      camera
    };
    pdf = smpl.pdf_or_pmf;
    return node;
  }
  
  
  ScatterSample SampleScatterCoordinate(const PathNode &node, const PathContext &context)
  {
    if (node.node_type == NodeType::SCATTER)
      return SampleScatterCoordinateOfScatterer(node, context);
    else
      return SampleScatterCoordinateOfEmitter(node, context);
  }
  
  // TODO: Well, that funnction name sounds a little retarded ...
  ScatterSample SampleScatterCoordinateOfScatterer(const PathNode &node, const PathContext &context)
  {
    const Double3 reverse_incident_dir = -node.incident_dir;
    if (node.geom_type == GeometryType::SURFACE)
    {
      const RaySurfaceIntersection &intersection = node.interaction.ray_surface;
      return intersection.shader().SampleBSDF(reverse_incident_dir, intersection, sampler, context);
    }
    else // is Volume
    {
      const VolumeInteraction &interaction = node.interaction.volume;
      auto smpl = interaction.medium().SamplePhaseFunction(reverse_incident_dir, interaction.pos, sampler, context);
      return smpl.as<ScatterSample>();
    }
  }
  
  ScatterSample SampleScatterCoordinateOfEmitter(const PathNode &node, const PathContext &context)
  {
    if (node.node_type == NodeType::CAMERA)
    {
      const PointEmitterArrayInteraction& interaction = node.interaction.emitter_point_array;
      auto smpl = interaction.emitter->TakeDirectionSampleFrom(
        interaction.unit_index, interaction.pos, sampler, context);
      return smpl.as<ScatterSample>();
    }
    else if (node.node_type == NodeType::ENV)
    {
      // For env light, the coordinate is a position!
      const EnvironmentalRadianceFieldInteraction &interaction = node.interaction.emitter_env;
      EnvLightPointSamplingBeyondScene gen{scene};
      double pdf = gen.Pdf(interaction.pos);
      Double3 org = gen.Sample(interaction.pos, sampler);
      return { org, interaction.radiance, pdf };
    }
    else if (node.node_type == NodeType::AREA_LIGHT)
    {
      const SurfaceInteraction &interaction = node.interaction.emitter_area;
      auto smpl = interaction.primitive().emitter->TakeDirectionSampleFrom(interaction.hitid, sampler, context);
      return smpl.as<ScatterSample>();
    }
    else if (node.node_type == NodeType::POINT_LIGHT)
    {
      const PointEmitterInteraction &interaction = node.interaction.emitter_point;
      auto smpl = interaction.emitter->TakeDirectionSampleFrom(interaction.pos, sampler, context);
      return smpl.as<ScatterSample>();
    }
    else
      return {};
  }

  
  Spectral3 EvaluateScatterCoordinate(const PathNode &node, const Double3 &out_direction, const PathContext &context, double *pdf)
  {
    if (node.node_type == NodeType::SCATTER)
      return EvaluateScatterCoordinateOfScatterer(node, out_direction, context, pdf);
    else
      return EvaluateScatterCoordinateOfEmitter(node, out_direction, context, pdf);
  }
  
  Spectral3 EvaluateScatterCoordinateOfScatterer(const PathNode &node, const Double3 &out_direction, const PathContext &context, double *pdf)
  {
    const Double3 reverse_incident_dir = -node.incident_dir;
    if (node.geom_type == GeometryType::SURFACE)
    {
      const RaySurfaceIntersection &intersection = node.interaction.ray_surface;
      return intersection.shader().EvaluateBSDF(reverse_incident_dir, intersection, out_direction, context, pdf);
    }
    else // is Volume
    {
      const VolumeInteraction &interaction = node.interaction.volume;
      return interaction.medium().EvaluatePhaseFunction(reverse_incident_dir, interaction.pos, out_direction, context, pdf);
    }
  }

  Spectral3 EvaluateScatterCoordinateOfEmitter(const PathNode &node, const Double3 &out_direction, const PathContext &context, double *pdf)
  {
    if (node.node_type == NodeType::CAMERA)
    {
      const PointEmitterArrayInteraction& interaction = node.interaction.emitter_point_array;
      auto response = interaction.emitter->Evaluate(interaction.pos, out_direction, context, pdf);
      this->sensor_connection_unit = response.unit_index;
      return response.weight;
    }
    else if (node.node_type == NodeType::ENV)
    {
      // For env light, the coordinate is a position!
      const EnvironmentalRadianceFieldInteraction &interaction = node.interaction.emitter_env;
      assert ((out_direction == interaction.pos));
      if (pdf)
      {
        EnvLightPointSamplingBeyondScene env_sampling{scene};
        *pdf = env_sampling.Pdf(interaction.pos);
      }
      return interaction.radiance;
    }
    else if (node.node_type == NodeType::AREA_LIGHT)
    {
      const SurfaceInteraction &interaction = node.interaction.emitter_area;
      return interaction.primitive().emitter->Evaluate(interaction.hitid, out_direction, context, pdf);
    }
    else if (node.node_type == NodeType::POINT_LIGHT)
    {
      const PointEmitterInteraction &interaction = node.interaction.emitter_point;
      return interaction.emitter->Evaluate(interaction.pos, out_direction, context, pdf);
    }
    else
      return Double3{0.};
  }


  double GetPdfOfGeneratingSampleOnEmitter(const RW::PathNode &node, const PathContext &context) const
  {
    if (node.node_type == RW::NodeType::AREA_LIGHT)
    {
      const SurfaceInteraction &interaction = node.interaction.emitter_area;
      assert (interaction.primitive().emitter);
      double pdf = interaction.primitive().emitter->EvaluatePdf(node.interaction.emitter_area.hitid, context);
      pdf *= light_picker.PmfOfLight(*interaction.primitive().emitter);
      pdf *= light_picker.emitter_type_selection_probabilities[LightPicker::IDX_PROB_AREA];
      return pdf;
    }
    else if(node.node_type == RW::NodeType::ENV)
    {
      const EnvironmentalRadianceFieldInteraction &interaction = node.interaction.emitter_env;
      double pdf = interaction.emitter->EvaluatePdf(interaction.pos, context);
      pdf *= light_picker.emitter_type_selection_probabilities[LightPicker::IDX_PROB_ENV];
      return pdf;
    }
    else // Other things are either not emitters or cannot be hit by definition.
    {
      return 0;
    }
  }
  

  RaySegment CalculateSegmentToTarget(const PathNode &source_node, const PathNode &target_node) const
  {
    assert(target_node.node_type != NodeType::ENV || source_node.node_type != NodeType::ENV);
    Double3 coord_source = source_node.coordinates();
    Double3 coord_target = target_node.coordinates();
    
    // This is a bit of an issue, because environemnt lights are represented only by the direction. The actual location of the node lies at infinity.
    // Capture by reference is important because the coordinates will change prior to calling this function a second time.
    using OrientedLength = std::pair<Double3, double>;
    auto FigureOutTheDirectionAndLenght = [&]() -> OrientedLength
    {
      if (target_node.node_type != NodeType::ENV)
      {
        if (source_node.node_type != NodeType::ENV)
        {
          Double3 d  = coord_target-coord_source;
          double l = Length(d);
          return OrientedLength{d/l, l};
        }
        else
        {
          ASSERT_NORMALIZED(coord_source);
          return OrientedLength{coord_source, sufficiently_long_distance_to_go_outside_the_scene_bounds};
        }
      }
      else
      {
        if (source_node.node_type != NodeType::ENV)
        {
          ASSERT_NORMALIZED(coord_target);
          return OrientedLength{-coord_target, sufficiently_long_distance_to_go_outside_the_scene_bounds};
        }
        else
        {
          assert (!"Should not get here.");
          return OrientedLength{Double3{NaN}, NaN};
        }
      }
    };
    Double3 direction;
    double length;
    std::tie(direction, length) = FigureOutTheDirectionAndLenght();
    
    AddAntiSelfIntersectionOffsetAt(coord_source, source_node, direction);
    AddAntiSelfIntersectionOffsetAt(coord_target, target_node, -direction);
    
    // Now as the coordinates likely changed, I have to figure out the direction again.
    std::tie(direction, length) = FigureOutTheDirectionAndLenght();
    
    assert(length <= LargeNumber);
    ASSERT_NORMALIZED(direction);
    assert(coord_source.allFinite());
    assert(coord_target.allFinite());
    
    if (source_node.node_type != NodeType::ENV)
      return RaySegment{{coord_source, direction}, length};
    else 
      return RaySegment{{coord_target-length*direction, direction}, length};
  }

  bool RouletteTerminationAllowedAtLength(int n)
  {
    static constexpr int MIN_NODE_COUNT = 3;
    return n > MIN_NODE_COUNT;
  }
  
  bool RouletteSurvival(Spectral3 &coeff)
  {
    assert (coeff.maxCoeff() >= 0.);
    // Clip at 0.9 to make the path terminate in high-albedo scenes.
    double p_survive = std::min(0.9, coeff.maxCoeff());
    if (sampler.Uniform01() > p_survive)
      return false;
    coeff *= 1./p_survive;
    return true;
  }

  bool SurvivalAtNthScatterNode(Spectral3 &scatter_coeff, int node_count)
  {
    if (node_count < max_path_node_count)
    {
      return !RouletteTerminationAllowedAtLength(node_count) || RouletteSurvival(scatter_coeff);
    }
    else
      return false;
  }
  
  
  double DFactorOf(const RW::PathNode &node, const Double3 &exitant_dir) const
  {
    // By definition the factor is 1 for Volumes.
    if (node.geom_type == RW::GeometryType::SURFACE)
    {
      ASSERT_NORMALIZED(node.geometry_normal());
      return std::abs(Dot(node.geometry_normal(), exitant_dir));
    }
    return 1.;
  } 
  
  
  void AddAntiSelfIntersectionOffsetAt(Double3 &position, const RW::PathNode &node, const Double3 &exitant_dir) const
  {
    if (node.geom_type == RW::GeometryType::SURFACE)
    {
      position += AntiSelfIntersectionOffset(node.geometry_normal(), RAY_EPSILON, exitant_dir);
    }
  }
  
  bool IsHitableEmissiveNode(const PathNode &node)
  {
    return node.node_type == NodeType::AREA_LIGHT || node.node_type == NodeType::ENV;
  }
  
  
  const ROI::EnvironmentalRadianceField& GetEnvLight() const
  {
    return envlight;
  }

  
  Spectral3 TransmittanceEstimate(RaySegment seg, MediumTracker &medium_tracker, const PathContext &context)
  {
    // TODO: Russian roulette.
    Spectral3 result{1.};
    auto SegmentContribution = [&seg, &medium_tracker, &context, this](double t1, double t2) -> Spectral3
    {
      RaySegment subseg{{seg.ray.PointAt(t1), seg.ray.dir}, t2-t1};
      return medium_tracker.getCurrentMedium().EvaluateTransmission(subseg, sampler, context);
    };

    intersector.All(seg.ray, seg.length);
    double t = 0;
    for (const auto &hit : intersector.Hits())
    {
      RaySurfaceIntersection intersection{hit, seg};
      result *= intersection.shader().EvaluateBSDF(-seg.ray.dir, intersection, seg.ray.dir, context, nullptr);
      if (result.isZero())
        return result;
      result *= SegmentContribution(t, hit.t);
      medium_tracker.goingThroughSurface(seg.ray.dir, intersection);
      t = hit.t;
    }
    result *= SegmentContribution(t, seg.length);
    return result;
  }


  void TrackToNextInteraction(
    CollisionData &collision,
    MediumTracker &medium_tracker,
    const PathContext &context)
  {
    Spectral3 total_weight{1.};
    auto &segment = collision.segment;
    while (true)
    {
      const Medium& medium = medium_tracker.getCurrentMedium();
      
      const auto &hit = collision.hit = intersector.First(segment.ray, segment.length);
      const auto &medium_smpl = collision.smpl = medium.SampleInteractionPoint(segment, sampler, context);
      total_weight *= medium_smpl.weight;
      
      if (medium_smpl.t >= segment.length && hit && hit.primitive->shader->IsPassthrough())
      {
        RaySurfaceIntersection intersection{hit, segment};
        medium_tracker.goingThroughSurface(segment.ray.dir, intersection);
        segment.ray.org = intersection.pos;
        segment.ray.org += AntiSelfIntersectionOffset(intersection, RAY_EPSILON, segment.ray.dir);
        segment.length = LargeNumber;
      }
      else
      {
        collision.smpl.weight = total_weight;
        return;
      }
    }
  }
};


} // namespace RandomWalk


class IRenderingAlgo
{
public:
  struct SensorResponse
  {
    int unit_index { -1 };
    RGB weight {};
    operator bool() const { return unit_index>=0; }
  };
  virtual ~IRenderingAlgo() {}
  virtual RGB MakePrettyPixel(int pixel_index) = 0;
  virtual ToyVector<SensorResponse> &GetSensorResponses() = 0;
  virtual void NotifyPassesFinished(int) {}
};



class NormalVisualizer : public IRenderingAlgo, public RandomWalk::RadianceEstimatorBase
{
  ToyVector<IRenderingAlgo::SensorResponse> rgb_responses;
public:
  NormalVisualizer(const Scene &_scene, const AlgorithmParameters &_params) : RandomWalk::RadianceEstimatorBase(_scene, _params) 
  {
  }
  
  RGB MakePrettyPixel(int pixel_index) override
  {
    auto context = PathContext{Color::LambdaIdxClosestToRGBPrimaries()};
    const auto &camera = scene.GetCamera();
    auto smpl_pos = camera.TakePositionSample(pixel_index, sampler, context);
    auto smpl_dir = camera.TakeDirectionSampleFrom(pixel_index, smpl_pos.coordinates, sampler, context);
    smpl_dir.pdf_or_pmf *= smpl_pos.pdf_or_pmf;
    smpl_dir.value *= smpl_pos.value;
    
    RaySegment seg{{smpl_pos.coordinates, smpl_dir.coordinates}, LargeNumber};
    HitId hit = intersector.First(seg.ray, seg.length);
    if (hit)
    {
      RaySurfaceIntersection intersection{hit, seg};
      RGB col = (intersection.normal.array() * 0.5 + 0.5).cast<Color::RGBScalar>();
      return col;
    }
    else
    {
      return RGB::Zero();
    }
  };
  
  ToyVector<IRenderingAlgo::SensorResponse>& GetSensorResponses() override
  {
    return rgb_responses;
  }
};


#pragma GCC diagnostic pop // Restore command line options
