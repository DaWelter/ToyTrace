#pragma once

#include <boost/pool/simple_segregated_storage.hpp>

#include "renderingalgorithms.hxx"

namespace RandomWalk
{

class Vertex
{
public:
  virtual ~Vertex() {}
public:
  virtual ScatterSample Sample(Sampler& sampler, const PathContext &context) const = 0;
  virtual Spectral3 Evaluate(const Double3 &out_direction, const PathContext &context, double *pdf) const = 0;
//   virtual void UpdateThingsAndMakeExitantRay(Ray &ray, ScatterSample &smpl, MediumTracker &medium_tracker, PathContext &context) const = 0;
  virtual void ApplyAntiSelfIntersectionTransform(Ray &ray) const {}
  
  virtual Double3 Position() const { return Double3{}; }
  
  virtual void InitRayAndMaybeHandlePassageThroughSurface(Ray &ray, const Double3 &exitant_dir, MediumTracker &medium_tracker) const
  {
    ray.org = Position();
    ray.dir = exitant_dir;
  }
  
//   virtual double Dfactor(const Double3 &exitant_dir) const { return 1.; }
#if 0 // TODO Anti self intersection transform
    
    if (intersection) 
      seg.ray.org += AntiSelfIntersectionOffset(*intersection, RAY_EPSILON, seg.ray.dir);

#endif
};


class Surface : public Vertex
{
  RaySurfaceIntersection intersection;
  Double3 reverse_incident_dir;
public:
  Surface(const HitId &_hitid, const RaySegment &_incident_segment)
    : intersection(_hitid, _incident_segment), reverse_incident_dir{-_incident_segment.ray.dir}
  {
  }

  ScatterSample Sample(Sampler& sampler, const PathContext &context) const override
  {
    const auto &shader = intersection.shader();
    auto smpl = shader.SampleBSDF(reverse_incident_dir, intersection, sampler, context);
    double d_factor = std::max(0., Dot(intersection.shading_normal, smpl.dir));
    smpl.value *= d_factor;
    return smpl;
  }
  
  Spectral3 Evaluate(const Double3 &out_direction, const PathContext &context, double *pdf) const override
  {
    double d_factor = std::max(0., Dot(intersection.shading_normal, out_direction));
    const auto &shader = intersection.shader();
    Spectral3 scatter_factor = shader.EvaluateBSDF(reverse_incident_dir, intersection, out_direction, context, pdf);
    return d_factor*scatter_factor;
  }
  
  virtual void InitRayAndMaybeHandlePassageThroughSurface(Ray &ray, const Double3 &exitant_dir, MediumTracker &medium_tracker) const
  {
    if (Dot(exitant_dir, intersection.normal) < 0.)
    {
      // By definition, intersection.normal points to where the intersection ray is comming from.
      // Thus we can determine if the sampled direction goes through the surface by looking
      // if the direction goes in the opposite direction of the normal.
      medium_tracker.goingThroughSurface(exitant_dir, intersection);
    }
    ray.org = intersection.pos;
    ray.dir = exitant_dir;
    ray.org += AntiSelfIntersectionOffset(intersection, RAY_EPSILON, ray.dir);
  }
  
  void ApplyAntiSelfIntersectionTransform(Ray &ray) const 
  {
    ray.org += AntiSelfIntersectionOffset(intersection, RAY_EPSILON, ray.dir);
  }
  
  Double3 Position() const { return intersection.pos; }
};


class Volume : public Vertex
{
  Double3 pos;
  Double3 reverse_incident_dir;
  const Medium& medium;
  // Sigma_s factor is accounted for in the medium interaction sample weight.
public:
  Volume(const Medium::InteractionSample &_medium_smpl, const Medium &_medium, const RaySegment &_incident_segment)
    : pos{_incident_segment.ray.PointAt(_medium_smpl.t)},
      reverse_incident_dir{-_incident_segment.ray.dir},
      medium{_medium}
  {
  }

  ScatterSample Sample(Sampler& sampler, const PathContext &context) const override
  {
    return medium.SamplePhaseFunction(reverse_incident_dir, pos, sampler, context);
  }
  
  Spectral3 Evaluate(const Double3 &out_direction, const PathContext &context, double *pdf) const override
  {
    return medium.EvaluatePhaseFunction(reverse_incident_dir, pos, out_direction, context, pdf);
  }
  
  Double3 Position() const { return pos; }
};



// class Distant : public Vertex 
// {
//   const Scene* scene;
// public:
//   Distant(const Scene* _scene, const Double3 &_incident_dir)
//     : scene(_scene)
//   {
//   }
// };
// 
// 
// class LightOrSensor : public Vertex
// {
//   RadianceOrImportance::EmitterSensorArray* emitter;
// public:
//   Sensor(RadianceOrImportance::EmitterSensorArray* _emitter)
//     : emitter(_emitter)
//   {
//   }
// };




class VertexStorage
{
  static constexpr std::size_t chunk_size = std::max({
    sizeof(void*), // Ensure sufficient space for the next-free-chunk pointer.
    sizeof(Vertex),
    sizeof(Surface),
    sizeof(Volume)
  });
  
  std::vector<char> memory;
  using MemoryManager = boost::simple_segregated_storage<std::size_t>;
  MemoryManager manager;

  void _init_manager()
  {
    manager.add_block(&memory.front(), memory.size(), chunk_size);
  }
  
public:
  VertexStorage(std::size_t max_num_items)
  {
    std::size_t some_reserve_for_inter_block_links = 128; // Which is chosen completely arbitrary in good faith.
    memory.resize(
      chunk_size * max_num_items + 
      some_reserve_for_inter_block_links);
    _init_manager();
  }
  
  template<class T, typename... Args>
  T* allocate(Args&&... args)
  {
    static_assert(sizeof(T) <= chunk_size, "Must fit in the chunks");
    void *p = manager.malloc();
    assert(p != nullptr);
    return new(p) T(std::forward<Args>(args)...);
  }
  
  // Clear the memory. WARNING: currently no d'tors are called!!!
  void clear()
  {
    manager.~MemoryManager();
#ifndef NDEBUG
    std::fill(memory.begin(), memory.end(), std::numeric_limits<char>::max());
#endif
    new (&manager) MemoryManager();
    _init_manager();
  }
  
  template<class T>
  void free(T *p)
  {
    p->~T();
    manager.free(p);
  }
};


} // RandomWalk


namespace RW = RandomWalk;

class PathTracing : public BaseAlgo
{ 
  int max_number_of_interactions;
  double sufficiently_long_distance_to_go_outside_the_scene_bounds;
  
  RW::VertexStorage vertex_storage;
  LambdaSelectionFactory lambda_selection_factory;
  
public:
  PathTracing(const Scene &_scene, const AlgorithmParameters &algo_params) 
  : BaseAlgo(_scene), 
    max_number_of_interactions(algo_params.max_ray_depth),
    vertex_storage(algo_params.max_ray_depth)
  {
    Box bb = _scene.GetBoundingBox();
    sufficiently_long_distance_to_go_outside_the_scene_bounds = 10.*(bb.max - bb.min).maxCoeff();
  }
  
  
  bool RouletteSurvival(Spectral3 &beta, int number_of_interactions)
  {
    static constexpr int MIN_LEVEL = 3;
    static constexpr double LOW_CONTRIBUTION = 0.5;
    if (number_of_interactions >= max_number_of_interactions || beta.isZero())
      return false;
    if (number_of_interactions < MIN_LEVEL)
      return true;
    double p_survive = std::min(0.9, beta.maxCoeff() / LOW_CONTRIBUTION);
    if (sampler.Uniform01() > p_survive)
      return false;
    beta *= 1./p_survive;
    return true;
  }
  
  
  RaySegment MakeSegmentToLight(const Double3 &pos_to_be_lit, const RadianceOrImportance::Sample &light_sample, const RW::Vertex &vertex)
  {
    RaySegment seg;
    if (!light_sample.is_direction)
    {
      seg = RaySegment::FromTo(pos_to_be_lit, light_sample.pos);
    }
    else
    {
      // TODO: Make the distance calculation work for the case where the entire space is filled with a scattering medium.
      // In this case an interaction point could be generated far outside the bounds of the scene geometry.
      // I have to derive an appropriate distance. Or just set the distance to infinite because I cannot know
      // the density distribution of the medium.
      seg = RaySegment{{pos_to_be_lit, -light_sample.pos},
        sufficiently_long_distance_to_go_outside_the_scene_bounds};
    }
    seg.length -= 2.*RAY_EPSILON;
    return seg;
  }
  

  Spectral3 LightConnection(const RW::Vertex &vertex, const MediumTracker &_medium_tracker_parent, const PathContext &context)
  {
    if (scene.GetNumLights() <= 0)
      return Spectral3{0.};

    const Light* light; double pmf_of_light; std::tie(light, pmf_of_light) = PickLightUniform();

    RadianceOrImportance::Sample light_sample = light->TakePositionSample(
      sampler, 
      RadianceOrImportance::LightPathContext(context.lambda_idx));
    
    RaySegment segment_to_light = MakeSegmentToLight(vertex.Position(), light_sample, vertex);

    Spectral3 scatter_factor = vertex.Evaluate(segment_to_light.ray.dir, context, nullptr);

    if (scatter_factor.isZero())
      return Spectral3{0.};

    MediumTracker medium_tracker = _medium_tracker_parent;
    vertex.InitRayAndMaybeHandlePassageThroughSurface(segment_to_light.ray, segment_to_light.ray.dir, medium_tracker);
    
    auto transmittance = TransmittanceEstimate(segment_to_light, medium_tracker, context);

    auto sample_value = (transmittance * light_sample.measurement_contribution * scatter_factor)
                         / (light_sample.pdf * (light_sample.is_direction ? 1. : Sqr(segment_to_light.length)) * pmf_of_light);

    PATH_LOGGING(
    if (!sample_value.isZero())
      path_logger.AddSegment(pos, segment_to_light.EndPoint(), sample_value, PathLogger::EYE_LIGHT_CONNECTION);
    )

    return sample_value;
  }
    

  RW::Vertex* TrackAndAllocateNextInteractionPoint(
    const Ray &ray, 
    MediumTracker &medium_tracker, 
    PathContext &context)
  {
    CollisionData collision{ray};
    
    TrackToNextInteraction(collision, medium_tracker, context);
    
    context.beta *= collision.smpl.weight;
    
    if (collision.smpl.t < collision.segment.length)
    {
      return vertex_storage.allocate<RW::Volume>(collision.smpl, medium_tracker.getCurrentMedium(), collision.segment);
    }
    else if (collision.hit)
    {
      return vertex_storage.allocate<RW::Surface>(collision.hit, collision.segment);
    }
    else
    {
      return nullptr; // vertex_storage.allocate<Distant>(ray.dir);
    }
  }
  
  
  void InitializeWalkFromCamera(int pixel_index, Ray &ray, PathContext &context)
  {
    auto cam_sample = RadianceOrImportance::TakeRaySample(
      scene.GetCamera(), pixel_index, sampler, 
      RadianceOrImportance::LightPathContext{context.lambda_idx});
    
    context.beta *= cam_sample.measurement_contribution / cam_sample.pdf;
    
    ray = cam_sample.ray_out;
  }
  
  
  RGB MakePrettyPixel(int pixel_index) override
  {
    auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    PathContext context{lambda_selection.first};
    
    Ray ray;
    Spectral3 path_sample_value{0.};
    
    InitializeWalkFromCamera(pixel_index, ray, context);

    if (context.beta.isZero())
      return RGB::Zero();
    
    MediumTracker medium_tracker{scene};
    medium_tracker.initializePosition(ray.org, hits);
    
    RW::Vertex* vertex = nullptr;
    int number_of_interactions = 0;
    
    while (true)
    {
      vertex = TrackAndAllocateNextInteractionPoint(ray, medium_tracker, context);
      
      if (!vertex)
        break;
      
      ++number_of_interactions;
      
      // TODO: MIS?
      path_sample_value += context.beta *
        LightConnection(*vertex, medium_tracker, context);
      
      bool survive = RouletteSurvival(context.beta, number_of_interactions);
      if (!survive)
        break;

      auto scatter_smpl = vertex->Sample(sampler, context);
      context.beta *= scatter_smpl.value / scatter_smpl.pdf;
      
      vertex->InitRayAndMaybeHandlePassageThroughSurface(ray, scatter_smpl.dir, medium_tracker);
      
      if (context.beta.isZero())
        break;

      assert(context.beta.allFinite());
      vertex_storage.free<RW::Vertex>(vertex);
    }

    if (vertex == nullptr && !context.beta.isZero() && number_of_interactions == 0) // Escaped!
    {
      // TODO: MIS?
      path_sample_value += context.beta * EvaluateEnvironmentalRadianceField(
        ray.dir,
        RadianceOrImportance::LightPathContext(lambda_selection.first));
    }
    
    vertex_storage.clear();
    return Color::SpectralSelectionToRGB(lambda_selection.second*path_sample_value, lambda_selection.first);
  }
};
