#pragma once

#include <boost/pool/simple_segregated_storage.hpp>

#include "renderingalgorithms.hxx"


class Vertex
{
public:
  virtual ~Vertex() {}
  
};


class Scattering
{
public:
  virtual ScatterSample Sample(const RaySurfaceIntersection &surface_hit, Sampler& sampler, const PathContext &context) const = 0;
  virtual Spectral3 Evaluate(const RaySurfaceIntersection &surface_hit, const Double3 &out_direction, const PathContext &context, double *pdf) const = 0;
};


class Surface : public Vertex
{
  RaySurfaceIntersection intersection;
  Double3 reverse_incident_dir;
};


class Volume : public Vertex
{
  Double3 pos;
  Double3 reverse_incident_dir;
  Medium& medium;
};


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
};


class PathTracing : public BaseAlgo
{
  int max_ray_depth;
  double sufficiently_long_distance_to_go_outside_the_scene_bounds;
  
  LambdaSelectionFactory lambda_selection_factory;
  
public:
  PathTracing(const Scene &_scene, const AlgorithmParameters &algo_params) 
  : BaseAlgo(_scene), max_ray_depth(algo_params.max_ray_depth) 
  {
    Box bb = _scene.GetBoundingBox();
    sufficiently_long_distance_to_go_outside_the_scene_bounds = 10.*(bb.max - bb.min).maxCoeff();

  }
  
  
  bool RouletteSurvival(Spectral3 &beta, int level)
  {
    static constexpr int MIN_LEVEL = 3;
    static constexpr double LOW_CONTRIBUTION = 0.5;
    if (level >= max_ray_depth || beta.isZero())
      return false;
    if (level < MIN_LEVEL)
      return true;
    double p_survive = std::min(0.9, beta.maxCoeff() / LOW_CONTRIBUTION);
    if (sampler.Uniform01() > p_survive)
      return false;
    beta *= 1./p_survive;
    return true;
  }
  
  
  RaySegment MakeSegmentToLight(const Double3 &pos_to_be_lit, const RadianceOrImportance::Sample &light_sample, const RaySurfaceIntersection *intersection)
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
    if (intersection) seg.ray.org += AntiSelfIntersectionOffset(*intersection, RAY_EPSILON, seg.ray.dir);
    seg.length -= 2.*RAY_EPSILON;
    return seg;
  }
  
  
  Spectral3 TransmittanceEstimate(RaySegment seg, MediumTracker medium_tracker, const PathContext &context)
  {
    // TODO: Russian roulette.
    Spectral3 result{1.};
    auto SegmentContribution = [&seg, &medium_tracker, &context, this](double t1, double t2) -> Spectral3
    {
      RaySegment subseg{{seg.ray.PointAt(t1), seg.ray.dir}, t2-t1};
      return medium_tracker.getCurrentMedium().EvaluateTransmission(subseg, sampler, context);
    };

    hits.clear();
    scene.IntersectAll(seg.ray, seg.length, hits);
    double t = 0;
    for (const auto &hit : hits)
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
  
  
  Spectral3 LightConnection(const Double3 &pos, const Double3 &incident_dir, const RaySurfaceIntersection *intersection, const MediumTracker &medium_tracker_parent, const PathContext &context)
  {
    if (scene.GetNumLights() <= 0)
      return Spectral3{0.};

    const Light* light; double pmf_of_light; std::tie(light, pmf_of_light) = PickLightUniform();

    RadianceOrImportance::Sample light_sample = light->TakePositionSample(
      sampler, 
      RadianceOrImportance::LightPathContext(context.lambda_idx));
    RaySegment segment_to_light = MakeSegmentToLight(pos, light_sample, intersection);

    double d_factor = 1.;
    Spectral3 scatter_factor;
    if (intersection)
    {
      d_factor = std::max(0., Dot(intersection->shading_normal, segment_to_light.ray.dir));
      const auto &shader = intersection->shader();
      scatter_factor = shader.EvaluateBSDF(-incident_dir, *intersection, segment_to_light.ray.dir, context, nullptr);
    }
    else
    {
      const auto &medium = medium_tracker_parent.getCurrentMedium();
      scatter_factor = medium.EvaluatePhaseFunction(-incident_dir, pos, segment_to_light.ray.dir, context, nullptr);
    }

    if (d_factor <= 0.)
      return Spectral3{0.};

    auto transmittance = TransmittanceEstimate(segment_to_light, medium_tracker_parent, context);

    auto sample_value = (transmittance * light_sample.measurement_contribution * scatter_factor)
                        * d_factor / (light_sample.pdf * (light_sample.is_direction ? 1. : Sqr(segment_to_light.length)) * pmf_of_light);

    PATH_LOGGING(
    if (!sample_value.isZero())
      path_logger.AddSegment(pos, segment_to_light.EndPoint(), sample_value, PathLogger::EYE_LIGHT_CONNECTION);
    )

    return sample_value;
  }
  
  Spectral3 EvaluateEnvironmentalRadianceField(const Double3 &viewing_dir, const RadianceOrImportance::LightPathContext &context)
  {
    Spectral3 environmental_radiance{0.};
    for (int i=0; i<scene.GetNumLights(); ++i)
    {
      const auto &light = scene.GetLight(i);
      if (light.is_environmental_radiance_distribution) // What an aweful hack. Should there not be an env map class or similar?
      {
        environmental_radiance += light.EvaluatePositionComponent(
          -viewing_dir,
          context,
          nullptr);
      }
    }
    return environmental_radiance;
  }
  
  RGB MakePrettyPixel(int pixel_index) override
  {
    auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    PathContext context{lambda_selection.first};
    auto cam_sample = TakeRaySample(
      scene.GetCamera(), pixel_index, sampler, 
      RadianceOrImportance::LightPathContext{lambda_selection.first});

    if (cam_sample.measurement_contribution.isZero())
      return RGB::Zero();
    
    MediumTracker medium_tracker = this->medium_tracker_root;
    medium_tracker.initializePosition(cam_sample.ray_out.org, hits);
    
    context.beta *= cam_sample.measurement_contribution / cam_sample.pdf;
    PATH_LOGGING(
      path_logger.NewTrace(context.beta);)

    Spectral3 path_sample_value{0.};
    RaySegment segment{cam_sample.ray_out, LargeNumber};
    int number_of_interactions = 0;

    bool gogogo = true;
    while (gogogo)
    {
      auto hit = scene.Intersect(segment.ray, segment.length);

      const Medium& medium = medium_tracker.getCurrentMedium();
      auto medium_smpl = medium.SampleInteractionPoint(segment, sampler, context);

      context.beta *= medium_smpl.weight;

      if (medium_smpl.t < segment.length)
      {
        Double3 interaction_location = segment.ray.PointAt(medium_smpl.t);
        PATH_LOGGING(path_logger.AddSegment(segment.ray.org, interaction_location, context.beta, PathLogger::EYE_SEGMENT);)

        ++number_of_interactions;
        
        path_sample_value += context.beta *
          LightConnection(interaction_location, segment.ray.dir, nullptr, medium_tracker, context);

        gogogo = RouletteSurvival(context.beta, number_of_interactions);
        if (gogogo)
        {
          auto scatter_smpl = medium.SamplePhaseFunction(-segment.ray.dir, interaction_location, sampler, context);
          context.beta *= scatter_smpl.value / scatter_smpl.pdf;

          PATH_LOGGING(path_logger.AddScatterEvent(segment.ray.org, scatter_smpl.dir, context.beta, PathLogger::SCATTER_VOLUME);)

          segment.ray.org = interaction_location;
          segment.ray.dir = scatter_smpl.dir;
          segment.length  = LargeNumber;
        }
      }
      else if (hit)
      {
        RaySurfaceIntersection intersection{hit, segment};
        PATH_LOGGING(path_logger.AddSegment(segment.ray.org, intersection.pos, context.beta, PathLogger::EYE_SEGMENT);)

        number_of_interactions = intersection.shader().IsPassthrough() ? number_of_interactions : number_of_interactions+1;
        
        if (!intersection.shader().IsReflectionSpecular())
        {
          auto lc = LightConnection(intersection.pos, segment.ray.dir, &intersection, medium_tracker, context);
          path_sample_value += context.beta * lc;
        }

        gogogo = RouletteSurvival(context.beta, number_of_interactions);
        if (gogogo)
        {
          auto surface_sample  = intersection.shader().SampleBSDF(-segment.ray.dir, intersection, sampler, context);
          gogogo = !surface_sample.value.isZero();
          if (gogogo)
          {
            // By definition, intersection.normal points to where the intersection ray is comming from.
            // Thus we can determine if the sampled direction goes through the surface by looking
            // if the direction goes in the opposite direction of the normal.
            double d_factor = 1.;
            if (Dot(surface_sample.dir, intersection.normal) < 0.)
            {
              medium_tracker.goingThroughSurface(surface_sample.dir, intersection);
            }
            else
            {
              d_factor = std::max(0., Dot(surface_sample.dir, intersection.shading_normal));
            }
            context.beta *= d_factor / surface_sample.pdf * surface_sample.value;
            PATH_LOGGING(path_logger.AddScatterEvent(intersection.pos, surface_sample.dir, context.beta, PathLogger::SCATTER_SURFACE);)

            segment.ray.org = intersection.pos+AntiSelfIntersectionOffset(intersection, RAY_EPSILON, surface_sample.dir);
            segment.ray.dir = surface_sample.dir;
            segment.length  = LargeNumber;
          }
        }
      }
      else
      {
        gogogo = false;
      }
      assert(context.beta.allFinite());
    }

    if (number_of_interactions == 0)
    {
      // I should actually do this whenever there was no deterministic light connection.
      // And that would be of course here, if the primary ray hit nothing, or if the last interaction
      // was perfectly specular.
      path_sample_value += 1./cam_sample.pdf * cam_sample.measurement_contribution * EvaluateEnvironmentalRadianceField(
        segment.ray.dir,
        RadianceOrImportance::LightPathContext(lambda_selection.first));
    }
    
    return Color::SpectralSelectionToRGB(lambda_selection.second*path_sample_value, lambda_selection.first);
  }
};
