#pragma once

#include <boost/pool/simple_segregated_storage.hpp>

#include "renderingalgorithms.hxx"

namespace RandomWalk
{

class Vertex
{
public:
  virtual ~Vertex() {}
  enum PathEndTag
  {
    END_VERTEX,
    SCATTER_VERTEX
  };
  const PathEndTag path_end_tag;
public:
  Vertex(PathEndTag _path_end_tag = SCATTER_VERTEX) : path_end_tag{_path_end_tag} {}
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

  virtual double Dfactor(const Double3 &exitant_dir) const { return 1.; }

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
    double d_factor = Dfactor(smpl.coordinates);
    smpl.value *= d_factor;
    return smpl;
  }
  
  Spectral3 Evaluate(const Double3 &out_direction, const PathContext &context, double *pdf) const override
  {
    double d_factor = Dfactor(out_direction);
    const auto &shader = intersection.shader();
    Spectral3 scatter_factor = shader.EvaluateBSDF(reverse_incident_dir, intersection, out_direction, context, pdf);
    return d_factor*scatter_factor;
  }
  
  virtual void InitRayAndMaybeHandlePassageThroughSurface(Ray &ray, const Double3 &exitant_dir, MediumTracker &medium_tracker) const override
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
  
  void ApplyAntiSelfIntersectionTransform(Ray &ray) const override
  {
    ray.org += AntiSelfIntersectionOffset(intersection, RAY_EPSILON, ray.dir);
  }
  
  Double3 Position() const override { return intersection.pos; }
  
  double Dfactor(const Double3 &exitant_dir) const override 
  { 
    return std::max(0., Dot(intersection.shading_normal, exitant_dir)); 
  }
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
  
  Double3 Position() const override { return pos; }
};


class Camera : public Vertex
{
  int unit_index;
  const ROI::PointEmitterArray &emitter;
  Double3 pos;
  IntersectionCalculator &intersector;
public:
  // TODO: Get rid of the intersector!
  Camera(const ROI::PointEmitterArray &_emitter, int _unit_index, IntersectionCalculator &_intersector)
    : Vertex(END_VERTEX), unit_index{_unit_index}, emitter{_emitter}, intersector{_intersector}
  {}
  
  ROI::PositionSample PositionSample(Sampler& sampler, const PathContext &context)
  {
    ROI::LightPathContext light_context{context.lambda_idx};
    auto smpl = emitter.TakePositionSample(unit_index, sampler, light_context);
    pos = smpl.coordinates;
    return smpl;
  }
  
  ScatterSample Sample(Sampler& sampler, const PathContext &context) const override
  {
    ROI::LightPathContext light_context{context.lambda_idx};
    auto smpl_dir = emitter.TakeDirectionSampleFrom(unit_index, pos, sampler, light_context);
    auto scatter_smpl = ScatterSample{smpl_dir.coordinates, smpl_dir.value, smpl_dir.pdf_or_pmf};
    SetPmfFlag(scatter_smpl); // TODO: See below.
    return scatter_smpl;
  }
  
  Spectral3 Evaluate(const Double3 &out_direction, const PathContext &context, double *pdf) const override
  {   
    // TODO: Fix me! As if there was only one discrete direction in which the ray goes through the pixel.
    // Of course this is not a bad approximation for a pin hole camera. But it is not true in general.
    // Need to support camera types with large aperture.
    if (pdf)
      *pdf = 0.;
    return Spectral3{0.};
  }
  
  void InitRayAndMaybeHandlePassageThroughSurface(Ray &ray, const Double3 &exitant_dir, MediumTracker &medium_tracker) const
  {
    medium_tracker.initializePosition(pos, intersector);
    Vertex::InitRayAndMaybeHandlePassageThroughSurface(ray, exitant_dir, medium_tracker);
  }

  Double3 Position() const override { return pos; }
};



// class EmissiveSurface : public Vertex
// {
//   ROI::AreaSample area_sample;
//   static ROI::AreaSample MakeAreaSampleFromHitPoint(const HitId &_hit, const RaySegment &_incident_segment)
//   {
//     ROI::AreaSample area;
//     area.hit = _hit;
//     Double3 dummy1, dummy2;
//     _hit.primitive->GetLocalGeometry(_hit, area.coordinates, area.normal, dummy1);
//     area.pdf_or_pmf = NaN;
//     area.value = Spectral3{NaN};
//     return area;
//   }
//   inline const ROI::AreaEmitter& GetEmitter() const { return *area_sample.hit.primitive->emitter; }
// public:
//   EmissiveSurface(const HitId &_hit, const RaySegment &_incident_segment)
//     : Vertex(END_VERTEX), area_sample{MakeAreaSampleFromHitPoint(_hit, _incident_segment)}
//   {
//     assert(_hit.primitive->emitter);
//   }
//   
//   ScatterSample Sample(Sampler& sampler, const PathContext &context) const override
//   {
//     assert (!"Not Implemented");
//     return ScatterSample{};
//   }
//   
//   Spectral3 Evaluate(const Double3 &out_direction, const PathContext &context, double *pdf_pos) const override
//   {
//     auto radiance = GetEmitter().Evaluate(area_sample, out_direction, ROI::LightPathContext{context.lambda_idx}, pdf_pos, nullptr);
//     return radiance;
//   }
//   
//   Double3 Position() const override { return area_sample.coordinates; }
//   
//   double Dfactor(const Double3 &out_direction) const override
//   {
//     return std::max(0., Dot(out_direction, area_sample.normal));
//   }
// };


// class EmissiveEnvironment : public Vertex
// {
//   const ROI::EnvironmentalRadianceField &envlight;
// public:
//   EmissiveEnvironment(const ROI::EnvironmentalRadianceField &_envlight) 
//     : Vertex(END_VERTEX), envlight(_envlight) 
//   {
//   }
//   
//   ScatterSample Sample(Sampler& sampler, const PathContext &context) const override
//   {
//     assert (!"Not Implemented");
//     return ScatterSample{};
//   }
//   
//   Spectral3 Evaluate(const Double3 &out_direction, const PathContext &context, double *pdf) const override
//   {
//     auto radiance = envlight.Evaluate(out_direction, ROI::LightPathContext{context.lambda_idx}, pdf);
//     return radiance;
//   }
// };


class VertexStorage
{
  static constexpr std::size_t chunk_size = std::max({
    sizeof(void*), // Ensure sufficient space for the next-free-chunk pointer.
    sizeof(Vertex),
    sizeof(Surface),
    sizeof(Volume),
    sizeof(Camera)
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
  bool do_sample_brdf;
  bool do_sample_lights;
public:
  PathTracing(const Scene &_scene, const AlgorithmParameters &algo_params) 
  : BaseAlgo(_scene), 
    max_number_of_interactions(algo_params.max_ray_depth),
    vertex_storage(algo_params.max_ray_depth),
    lambda_selection_factory{},
    do_sample_brdf{true},
    do_sample_lights{true}
  {
    Box bb = _scene.GetBoundingBox();
    sufficiently_long_distance_to_go_outside_the_scene_bounds = 10.*(bb.max - bb.min).maxCoeff();
  }
  
  template<class SampleType>
  double MaybeMisWeight(const SampleType &sample, double pdf_wrt_solid_angle)
  {
    double mis_weight = 1.;
    if (IsFromPdf(sample) && this->do_sample_brdf && this->do_sample_lights)
    {
      mis_weight = PowerHeuristic(PdfValue(sample), {pdf_wrt_solid_angle});
    }
    return mis_weight;
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
  
  struct TagEndNodeSample {};
  using EndNodeSample = Sample<Double3, Spectral3, TagEndNodeSample>;
  
  struct ConnectionEndNodeData
  {
    EndNodeSample sample;
    RaySegment segment_to_target;
    bool is_wrt_solid_angle;
  };
  
  
  ConnectionEndNodeData ComputeConnectionSample(const ROI::EnvironmentalRadianceField &target, const RW::Vertex &vertex, const ROI::LightPathContext &light_context)
  {
    ConnectionEndNodeData nd;
    auto dir_sample = target.TakeDirectionSample(sampler, light_context);
    nd.sample = dir_sample.as<TagEndNodeSample>();
    nd.segment_to_target = RaySegment{{vertex.Position(), -dir_sample.coordinates}, sufficiently_long_distance_to_go_outside_the_scene_bounds};
    nd.is_wrt_solid_angle = true;
    return nd;
  }
  
  ConnectionEndNodeData ComputeConnectionSample(const ROI::PointEmitter& target, const RW::Vertex &vertex, const ROI::LightPathContext &light_context)
  {
    ConnectionEndNodeData nd;
    auto pos_sample = target.TakePositionSample(sampler, light_context);
    nd.segment_to_target = RaySegment::FromTo(vertex.Position(), pos_sample.coordinates);
    // Direction component of Le(x-x') is zero for lights with delta function in Le.
    auto direction_factor = target.EvaluateDirectionComponent(pos_sample.coordinates, -nd.segment_to_target.ray.dir, light_context, nullptr);
    nd.sample = pos_sample.as<TagEndNodeSample>();
    nd.sample.value *= direction_factor;
    nd.is_wrt_solid_angle = false;
    return nd;
  }
  
  ConnectionEndNodeData ComputeConnectionSample(const ROI::AreaEmitter& target, const Primitive &primitive, const RW::Vertex &vertex, const ROI::LightPathContext &light_context)
  {
    ConnectionEndNodeData nd;
    auto area_sample = target.TakeAreaSample(primitive, sampler, light_context);
    nd.segment_to_target = RaySegment::FromTo(vertex.Position(), area_sample.coordinates.pos);
    auto direction_factor = target.Evaluate(area_sample.coordinates, -nd.segment_to_target.ray.dir, light_context, nullptr, nullptr);
    nd.sample.coordinates = area_sample.coordinates.pos;
    nd.sample.value       = area_sample.value;
    nd.sample.pdf_or_pmf  = area_sample.pdf_or_pmf;
    nd.sample.value *= direction_factor;
    nd.sample.pdf_or_pmf = TransformPdfFromAreaToSolidAngle(nd.sample.pdf_or_pmf, nd.segment_to_target.length, nd.segment_to_target.ray.dir, area_sample.coordinates.normal);
    nd.is_wrt_solid_angle = true;
    return nd;
  }

  Spectral3 CalculateLightConnectionSubPathWeight(const RW::Vertex &vertex, const MediumTracker &_medium_tracker_parent, const PathContext &context)
  {    
    ROI::LightPathContext light_context{context.lambda_idx};

    //bool can_be_hit_by_scatter_function_sampling = true;
    //double pdf_of_light_vertex_wrt_solid_angle = NaN;
    ConnectionEndNodeData nd;
    double pmf_of_light = 1.;
    int which_kind = TowerSampling<3>(emitter_type_selection_probabilities.data(), sampler.Uniform01());
    if (which_kind == IDX_PROB_ENV)
    {
      const auto &envlight = this->GetEnvLight();
      nd = ComputeConnectionSample(envlight, vertex, light_context);
    }
    else if(which_kind == IDX_PROB_AREA)
    {
      const Primitive* primitive;
      const ROI::AreaEmitter* emitter;
      std::tie(primitive, emitter, pmf_of_light) = PickAreaLight();
      nd = ComputeConnectionSample(*emitter, *primitive, vertex, light_context);
    }
    else
    {
      const ROI::PointEmitter* light; 
      std::tie(light, pmf_of_light) = PickLight();
      nd = ComputeConnectionSample(*light, vertex, light_context);
    }
    // TODO: Unfuck this. Maybe try to use an offset from the surface. Otherwise there are self-interseciton issues!
    nd.segment_to_target.length -= 10.*RAY_EPSILON; // For safety against intersections with the target.
    nd.sample.pdf_or_pmf *= pmf_of_light*emitter_type_selection_probabilities[which_kind];
    
    double scatter_pdf_wrt_solid_angle = NaN;
    Spectral3 scatter_factor = vertex.Evaluate(nd.segment_to_target.ray.dir, context, &scatter_pdf_wrt_solid_angle);

    if (scatter_factor.isZero())
      return Spectral3{0.};

    MediumTracker medium_tracker = _medium_tracker_parent;
    vertex.InitRayAndMaybeHandlePassageThroughSurface(nd.segment_to_target.ray, nd.segment_to_target.ray.dir, medium_tracker);
    
    auto transmittance = TransmittanceEstimate(nd.segment_to_target, medium_tracker, context);
    double mis_weight = MaybeMisWeight(nd.sample, scatter_pdf_wrt_solid_angle);    
    double r2_factor = nd.is_wrt_solid_angle ? 1. : 1./Sqr(nd.segment_to_target.length);
    
    Spectral3 sub_path_weight = transmittance * scatter_factor * nd.sample.value;
    sub_path_weight *= r2_factor*mis_weight / PmfOrPdfValue(nd.sample);
    return sub_path_weight;
  }
  

  enum EntityHitFlag
  {
    NOTHING,
    SCATTERER,
    AREA_EMITTER,
    ENV_LIGHT
  };
  
  Spectral3 EmitterHit(const ROI::EnvironmentalRadianceField& emitter, const RaySegment &incident_segment, const PathContext &context, double &pdf_of_emitter_wrt_solid_angle)
  {
    auto radiance = envlight.Evaluate(-incident_segment.ray.dir, ROI::LightPathContext{context.lambda_idx}, &pdf_of_emitter_wrt_solid_angle);
    assert(pdf_of_emitter_wrt_solid_angle >= 0 && std::isfinite(pdf_of_emitter_wrt_solid_angle));
    return radiance;
  }
  
  Spectral3 EmitterHit(const ROI::AreaEmitter &emitter, const HitId &hit, const RaySegment &incident_segment, const PathContext &context, double &pdf_of_emitter_wrt_solid_angle)
  {
    ROI::AreaSampleCoordinates area = ROI::MakeAreaSampleCoordinatesFrom(hit);
    double pdf_of_pos;
    auto radiance = emitter.Evaluate(area, -incident_segment.ray.dir, ROI::LightPathContext{context.lambda_idx}, &pdf_of_pos, nullptr);
    assert(pdf_of_pos >= 0 && std::isfinite(pdf_of_pos));
    pdf_of_emitter_wrt_solid_angle = TransformPdfFromAreaToSolidAngle(pdf_of_pos, incident_segment.length, incident_segment.ray.dir, area.normal);
    return radiance;
  }
  
  
  Spectral3 CalculateEmitterHitSubPathWeight(EntityHitFlag entity_hit_flag, const CollisionData &collision, const ScatterSample &last_scatter_sample, const PathContext &context)
  {
    Spectral3 end_weight{0.};
    double pdf_of_emitter_wrt_solid_angle{0.};
    if (entity_hit_flag == ENV_LIGHT)
    {
      assert(collision.hit.primitive == nullptr);
      end_weight = EmitterHit(
        GetEnvLight(), collision.segment, context, pdf_of_emitter_wrt_solid_angle);
      pdf_of_emitter_wrt_solid_angle *= emitter_type_selection_probabilities[IDX_PROB_ENV];
    }
    else if (entity_hit_flag == AREA_EMITTER) 
    {
      assert(collision.hit.primitive && collision.hit.primitive->emitter);
      const auto &emitter = *collision.hit.primitive->emitter;
      end_weight = EmitterHit(
        emitter, collision.hit, collision.segment, context, pdf_of_emitter_wrt_solid_angle);
      pdf_of_emitter_wrt_solid_angle *= emitter_type_selection_probabilities[IDX_PROB_AREA] * PmfOfLight(emitter);
    }
    double mis_weight = MaybeMisWeight(last_scatter_sample, pdf_of_emitter_wrt_solid_angle);
    end_weight *= mis_weight;
    return end_weight;
  }

  
  EntityHitFlag DetermineNextInteractionType(const CollisionData &collision, MediumTracker &medium_tracker, RW::Vertex* &vertex)
  {
    vertex = nullptr;
    if (collision.smpl.t < collision.segment.length)
    {
      vertex = vertex_storage.allocate<RW::Volume>(collision.smpl, medium_tracker.getCurrentMedium(), collision.segment);
      return SCATTERER;
    }
    else if (collision.hit)
    {
      if (collision.hit.primitive->emitter)
      {
        return AREA_EMITTER;
      }
      else 
      {
        vertex = vertex_storage.allocate<RW::Surface>(collision.hit, collision.segment);
        return SCATTERER;
      }
    }
    else
    {
      return ENV_LIGHT;
    }
  }
  
  RGB MakePrettyPixel(int pixel_index) override
  {
    auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    PathContext context{lambda_selection.first};
    MediumTracker medium_tracker{scene};
    
    Spectral3 path_sample_value{0.};
    
    RW::Camera start_vertex(scene.GetCamera(), pixel_index, intersector);
    {
      auto pos_smpl = start_vertex.PositionSample(sampler, context);
      context.beta *= (1./PmfOrPdfValue(pos_smpl));
    }
  
    RW::Vertex* vertex = &start_vertex;
    ScatterSample scatter_smpl{};
    CollisionData collision{Ray{}};
    EntityHitFlag entity_hit_flag = NOTHING;
    
    Ray ray;
    {
      scatter_smpl = vertex->Sample(sampler, context);
      context.beta *= scatter_smpl.value / PmfOrPdfValue(scatter_smpl);
     
      if (context.beta.isZero())
        return RGB::Zero();
      
      vertex->InitRayAndMaybeHandlePassageThroughSurface(ray, scatter_smpl.coordinates, medium_tracker);
    }

    int number_of_interactions = 0;
    
    while (true)
    {
      collision = CollisionData{ray};
      TrackToNextInteraction(collision, medium_tracker, context);
      context.beta *= collision.smpl.weight;
      
      entity_hit_flag = DetermineNextInteractionType(collision, medium_tracker, vertex);
      
      if (entity_hit_flag != SCATTERER)
        break;
      
      ++number_of_interactions;

      bool survive = RouletteSurvival(context.beta, number_of_interactions);
      if (!survive)
        break;
      
      if (this->do_sample_lights)
      {
        path_sample_value += context.beta *
          CalculateLightConnectionSubPathWeight(*vertex, medium_tracker, context);
      }

      scatter_smpl = vertex->Sample(sampler, context);
      context.beta *= scatter_smpl.value / PmfOrPdfValue(scatter_smpl);
      
      if (context.beta.isZero())
        break;
      
      vertex->InitRayAndMaybeHandlePassageThroughSurface(ray, scatter_smpl.coordinates, medium_tracker);

      assert(context.beta.allFinite());
      vertex_storage.free<RW::Vertex>(vertex);
    }

    if (this->do_sample_brdf)
    {
      path_sample_value += context.beta * CalculateEmitterHitSubPathWeight(
        entity_hit_flag, collision, scatter_smpl, context
      );
    }
    
    vertex_storage.clear();
    return Color::SpectralSelectionToRGB(lambda_selection.second*path_sample_value, lambda_selection.first);
  }
};
