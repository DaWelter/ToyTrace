#pragma once

#include <fstream>
#include <cstring> // memcopy, yes memcopy
#include <type_traits>

#include "scene.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "camera.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "renderbuffer.hxx"
#include "lightpicker_trivial.hxx"
#include "rendering_util.hxx"

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wunused-parameter" 
#pragma GCC diagnostic warning "-Wunused-variable"

using  AlgorithmParameters = RenderingParameters;
namespace ROI = RadianceOrImportance;



double UpperBoundToBoundingBoxDiameter(const Scene &scene);





namespace RandomWalk
{
namespace RW = RandomWalk;


// Be warned there be dragons and worse.

enum class NodeType : char 
{
  SURFACE_SCATTER,
  VOLUME_SCATTER,
  SURFACE_EMITTER,
  VOLUME_EMITTER,
  ENV,
  POINT_LIGHT,
  CAMERA,
  ZERO_CONTRIBUTION_ABORT_WALK
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


static_assert(std::is_base_of<InteractionPoint, VolumeInteraction>::value, "Must share fields");
static_assert(std::is_base_of<InteractionPoint, PointEmitterInteraction>::value, "Must share fields");
static_assert(std::is_base_of<InteractionPoint, PointEmitterArrayInteraction>::value, "Must share fields");
static_assert(std::is_base_of<InteractionPoint, EnvironmentalRadianceFieldInteraction>::value, "Must share fields");
static_assert(std::is_base_of<InteractionPoint, SurfaceInteraction>::value, "Must share fields");


// TODO: C++ wants to force default initialization/construction on me. Okay, so I should write constructors for the various node types.
struct PathNode
{ 
  NodeType node_type {NodeType::ZERO_CONTRIBUTION_ABORT_WALK};
  Double3 incident_dir {0.};
  union Interactions
  {
    InteractionPoint point;
    SurfaceInteraction surface;
    VolumeInteraction volume;
    PointEmitterInteraction emitter_point;
    PointEmitterArrayInteraction emitter_point_array;
    EnvironmentalRadianceFieldInteraction emitter_env;
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
  
  bool IsSurfaceInteraction() const
  {
    return node_type == NodeType::SURFACE_EMITTER || node_type == NodeType::SURFACE_SCATTER;
  }

  bool IsVolumeInteraction() const
  {
    return node_type == NodeType::VOLUME_SCATTER || node_type == NodeType::VOLUME_EMITTER;
  }
  
  bool IsScatterNode() const {
    return node_type == NodeType::VOLUME_SCATTER || node_type == NodeType::SURFACE_SCATTER;
  }
  
  bool TryEnableOperationsOnEmitter(const Scene &scene)
  {
    switch(node_type)
    {
      case NodeType::SURFACE_SCATTER:
        if (scene.GetMaterialOf(interaction.surface.hitid).emitter)
        {
          node_type = NodeType::SURFACE_EMITTER;
          return true;
        }
        return false;
      case NodeType::VOLUME_SCATTER:
        if (interaction.volume.medium().is_emissive)
        {
          node_type = NodeType::VOLUME_EMITTER;
          return true;
        }
        return false;
      case NodeType::SURFACE_EMITTER:
      case NodeType::VOLUME_EMITTER:
      case NodeType::ENV:
      case NodeType::POINT_LIGHT:
      case NodeType::CAMERA:
        return true;
      case NodeType::ZERO_CONTRIBUTION_ABORT_WALK:
        return false;
    }
    
    return true;
  }
  
  
  void EnableOperationOnScattererIfFeasible()
  {
    switch (node_type)
    {
      case NodeType::SURFACE_EMITTER:
        node_type = NodeType::SURFACE_SCATTER;
        return;
      case NodeType::VOLUME_EMITTER:
        node_type = NodeType::VOLUME_SCATTER;
        return;
      default:
        return;
    }
  }
  
  
  Double3 geometry_normal() const
  {
    assert(IsSurfaceInteraction());
    return interaction.surface.geometry_normal;
  }
  
  Double3 coordinates() const
  {

    return interaction.point.pos;
  };
};





// A set of callbacks to be invoked when a light is sampled.
// Lights are pretty distinct types. So I use the visitor
// pattern to invoke the appropriate action for the selected type.
struct EmitterSampleVisitor
{
  const PathContext &context;
  const Scene &scene;
  Sampler &sampler;
  Pdf &pdf;           // Return value.
  RW::PathNode &node; // Return value.
  
  void operator()(const ROI::EnvironmentalRadianceField &env, double prob)
  {
    auto smpl = env.TakeDirectionSample(sampler, context);
    node.node_type = RW::NodeType::ENV;
    node.interaction.emitter_env = RW::EnvironmentalRadianceFieldInteraction{
      smpl.coordinates,
      env
    };
    node.interaction.emitter_env.radiance = smpl.value;
    pdf = prob*smpl.pdf_or_pmf;
  }
  void operator()(const ROI::PointEmitter &light, double prob)
  {
    node.node_type = RW::NodeType::POINT_LIGHT;
    node.interaction.emitter_point = RW::PointEmitterInteraction{
      light.Position(),
      light
    };
    pdf = Pdf::MakeFromDelta(prob);
  }
  void operator()(const PrimRef& prim_ref, double prob)
  {
    const auto &mat = scene.GetMaterialOf(prim_ref);
    assert(mat.emitter); // Otherwise would would not have been selected.
    auto smpl = mat.emitter->TakeAreaSample(prim_ref, sampler, context);
    node.node_type = RW::NodeType::SURFACE_EMITTER;
    node.interaction.surface = SurfaceInteraction{
      smpl.coordinates };
    pdf = prob*smpl.pdf_or_pmf;
  }
  void operator()(const Medium &medium, double prob)
  {
    Medium::VolumeSample smpl = medium.SampleEmissionPosition(sampler, context);
    double pdf_which_cannot_be_delta = 0;
    auto radiance =  medium.EvaluateEmission(smpl.pos, context, &pdf_which_cannot_be_delta);
    node.node_type = RW::NodeType::VOLUME_EMITTER;
    node.interaction.volume = VolumeInteraction(smpl.pos, medium, radiance, Spectral3{0.});
    // The reason sigma_s is set to 0 in the line above, is that the initial light node will never be used for scattering.
    pdf = prob*pdf_which_cannot_be_delta;
  }
};



class RadianceEstimatorBase
{
protected:
  Sampler sampler;
  const Scene &scene;
  double sufficiently_long_distance_to_go_outside_the_scene_bounds;
  int max_path_node_count;

protected:
  TrivialLightPicker light_picker;
  int sensor_connection_unit = {};
public:
  RadianceEstimatorBase(const Scene &_scene, const AlgorithmParameters &algo_params = AlgorithmParameters{})
    : sampler{}, scene{_scene},     
      sufficiently_long_distance_to_go_outside_the_scene_bounds{10.*UpperBoundToBoundingBoxDiameter(_scene)},
      max_path_node_count{algo_params.max_ray_depth+1},
      light_picker{_scene}
  {

  }

  virtual ~RadianceEstimatorBase() {}

  struct StepResult
  {
    RW::PathNode node;
    RaySegment segment; // Leading to the node
    Pdf scatter_pdf; // Of existant coordinate. For envlight it is w.r.t. area. Else w.r.t solid angle. TODO: Maybe put elsewhere.
    Spectral3 beta_factor {0.};
  };
  
  
  StepResult TakeRandomWalkStep(const RW::PathNode &source_node, MediumTracker &medium_tracker, const PathContext &context, VolumePdfCoefficients *volume_pdf_coeff = nullptr)
  {
    struct TagRaySample {};
    using RaySample = Sample<Ray, Spectral3, TagRaySample>;

    auto SampleExitantRay = [&]() -> RaySample
    {
      if (source_node.node_type == RW::NodeType::ENV)
      {
        auto smpl = SampleScatterCoordinateOfEmitter(source_node, context);
        // Note that source_node.coordinates() is the light direction, in contrast to how all other nodes function.
        return RaySample{{smpl.coordinates, source_node.coordinates()}, smpl.value, smpl.pdf_or_pmf};
      }
      else
      {
        auto smpl = source_node.IsScatterNode() ?
          SampleScatterCoordinateOfScatterer(source_node, context) :
          SampleScatterCoordinateOfEmitter(source_node, context);
        Ray ray{source_node.coordinates(), smpl.coordinates};
        smpl.value *= DFactorOf(source_node, ray.dir);
        AddAntiSelfIntersectionOffsetAt(ray.org, source_node, ray.dir);
        return RaySample{ray, smpl.value, smpl.pdf_or_pmf};
      }
    };
    
    RaySample ray_sample = SampleExitantRay();
    const Ray &ray = ray_sample.coordinates;

    StepResult node_sample;
    node_sample.beta_factor = ray_sample.value / PmfOrPdfValue(ray_sample);

    if (node_sample.beta_factor.isZero()) // Because of odd situations where ray can be scattered below the surface due to smooth normal interpolation.
    {
      assert(node_sample.node.node_type == RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK); // TODO Write test to check that the node is default constructed as required here.
      return node_sample;
    }
       
    MaybeHandlePassageThroughSurface(source_node, ray.dir, medium_tracker);

    TrackToNextInteraction(ray, context, medium_tracker, volume_pdf_coeff, 
      /*surface_visitor=*/[&](const SurfaceInteraction &intersection, double distance, const Spectral3 &weight)
      {
        node_sample.beta_factor *= weight;
        node_sample.segment = RaySegment{ray, distance};
        node_sample.node.node_type = RW::NodeType::SURFACE_SCATTER;
        node_sample.node.interaction.surface = intersection;
      },
      /*volume visitor=*/[&](const VolumeInteraction &interaction, double distance, const Spectral3 &weight)
      {
        node_sample.beta_factor *= weight;
        node_sample.segment = RaySegment{ray, distance};
        const auto &m = medium_tracker.getCurrentMedium();
        const Spectral3 radiance = m.is_emissive ? m.EvaluateEmission(interaction.pos, context, nullptr) : Spectral3::Zero();
        node_sample.node.interaction.volume = interaction;
        node_sample.node.interaction.volume.radiance = radiance;
        node_sample.node.node_type = RW::NodeType::VOLUME_SCATTER;
      },
      /* escape_visitor=*/[&](const Spectral3 &weight)
      {
        node_sample.beta_factor *= weight;
        node_sample.segment = RaySegment{ray, LargeNumber};
        const auto &emitter = scene.GetTotalEnvLight();
        node_sample.node.node_type = RW::NodeType::ENV;
        node_sample.node.interaction.emitter_env = RW::EnvironmentalRadianceFieldInteraction{
          -ray.dir,
          emitter
        };
        node_sample.node.interaction.emitter_env.radiance = emitter.Evaluate(-ray.dir, context);
      }
    );
    
    node_sample.node.incident_dir = ray.dir;
    node_sample.scatter_pdf = ray_sample.pdf_or_pmf;    
    assert(node_sample.scatter_pdf > 0.); // Since I rolled that sample it should have non-zero probability of being generated.
    return node_sample;
  }

  
  struct ConnectionSegment
  {
    RaySegment segment {};
    Spectral3 weight { 0. };
  };
  

  ConnectionSegment CalculateConnection(
    const RW::PathNode &eye_node, 
    const MediumTracker &eye_medium_tracker, 
    const RW::PathNode &light_node, 
    const PathContext &eye_context, 
    const PathContext &light_context, 
    double *pdf_source, 
    double *pdf_target, 
    VolumePdfCoefficients *volume_pdf_coeff = nullptr)
  {
    ConnectionSegment result{};
    if (eye_node.node_type == RW::NodeType::ENV && light_node.node_type == RW::NodeType::ENV)
      return result;

    RaySegment &segment_to_light = result.segment = CalculateSegmentToTarget(eye_node, light_node);
      
    Spectral3 scatter_factor = Evaluate(eye_node, segment_to_light.ray.dir, eye_context, pdf_source);
    Spectral3 target_scatter_factor = Evaluate(light_node, -segment_to_light.ray.dir, light_context, pdf_target);
    
    if (scatter_factor.isZero() || target_scatter_factor.isZero())
      return result;

    scatter_factor *= DFactorOf(eye_node, segment_to_light.ray.dir);
    target_scatter_factor *= DFactorOf(light_node, -segment_to_light.ray.dir);
    
    MediumTracker medium_tracker = eye_medium_tracker;
    MaybeHandlePassageThroughSurface(eye_node, segment_to_light.ray.dir, medium_tracker);
    
    auto transmittance = TransmittanceEstimate(segment_to_light, medium_tracker, eye_context, volume_pdf_coeff);

    bool is_wrt_solid_angle = eye_node.node_type == RW::NodeType::ENV || light_node.node_type == RW::NodeType::ENV;
    double r2_factor = is_wrt_solid_angle ? 1. : 1./Sqr(segment_to_light.length);
    
    result.weight = r2_factor * transmittance * scatter_factor * target_scatter_factor;
    result.segment = segment_to_light;
    return result;
  }

  
  Spectral3 Evaluate(const PathNode &node, const Double3 &out_direction, const PathContext &context, double *pdf)
  {
    return node.IsScatterNode() ?
      EvaluateScatterCoordinateOfScatterer(node, out_direction, context, pdf) :
      EvaluateScatterCoordinateOfEmitter(node, out_direction, context, pdf);
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
    else if (node.node_type == NodeType::SURFACE_EMITTER)
    {
      const SurfaceInteraction &interaction = node.interaction.surface;
      return ASSERT_NOT_NULL(GetMaterialOf(interaction, scene).emitter)->Evaluate(interaction.hitid, out_direction, context, pdf);
    }
    else if (node.node_type == NodeType::POINT_LIGHT)
    {
      const PointEmitterInteraction &interaction = node.interaction.emitter_point;
      return interaction.emitter->Evaluate(interaction.pos, out_direction, context, pdf);
    }
    else if (node.node_type == NodeType::VOLUME_EMITTER)
    {
      const VolumeInteraction &interaction = node.interaction.volume;
      if (pdf)
      {
        // Emission from volumes is always isotropic.
        *pdf = 1./(4.*Pi);
      }
      return interaction.radiance;
    }
    else 
      return Double3{0.};
  }


  Spectral3 EvaluateScatterCoordinateOfScatterer(const PathNode &node, const Double3 &out_direction, const PathContext &context, double *pdf)
  {
    
    const Double3 reverse_incident_dir = -node.incident_dir;
    if (node.node_type == NodeType::SURFACE_SCATTER)
    {
      const SurfaceInteraction &intersection = node.interaction.surface;
      auto  fs = GetShaderOf(intersection, scene).EvaluateBSDF(reverse_incident_dir, intersection, out_direction, context, pdf);
      fs *= BsdfCorrectionFactor(reverse_incident_dir, intersection, out_direction, context.transport);
      return fs;
    }
    else // is Volume
    {
      assert(node.node_type == RW::NodeType::VOLUME_SCATTER);
      const VolumeInteraction &interaction = node.interaction.volume;
      Spectral3 kernel = interaction.medium().EvaluatePhaseFunction(reverse_incident_dir, interaction.pos, out_direction, context, pdf);
      kernel *= interaction.sigma_s;
      return kernel;
    }
  }


  // Convert the native pdf of sampling the scatter coordinate of  the source into the pdf to represent the location of the target node.
  // It is important in MIS weighting to express the pdf of various sampling 
  // strategies w.r.t. to the same integration domain. E.g. Solid angle or area.
  // However it should be okay to compose the joint pdfs of paths using different sub-domains, e.g.:
  // pdf(path) = pdf_w1(w1)*pdf_a2(x2)*pdf_a3(x3)*... This is cool as long as the product space is
  // consistently used.
  // Ref: Veach's Thesis and PBRT book.
  double PdfConversionFactor(
    const RW::PathNode &source, 
    const RW::PathNode &target, 
    const RaySegment &segment,
    const std::tuple<double, double> interaction_density_and_transmittance,
    bool is_parallel_beam
  ) const
  {
    double factor = DFactorOf(target, segment.ray.dir);
    if (source.node_type != NodeType::ENV && target.node_type != NodeType::ENV && !is_parallel_beam)
    {
      // If source is env: I already generate points on a surface. Thus the conversion from angle to surface is only needed for other types.
      // If target is env: By definition, solid angle is used to describe the coordinate of the final node.
      factor /= (Sqr(segment.length)+Epsilon);
    }
    factor *= target.IsVolumeInteraction() ? 
      std::get<0>(interaction_density_and_transmittance) : 
      std::get<1>(interaction_density_and_transmittance);
    return factor;
  }
  
    
  void MaybeHandlePassageThroughSurface(const RW::PathNode &node, const Double3 &dir_of_travel, MediumTracker &medium_tracker) const
  {
    if (node.IsSurfaceInteraction() && 
        Dot(dir_of_travel, node.interaction.surface.normal) < 0.)
    {
      // By definition, intersection.normal points to where the intersection ray is coming from.
      // Thus we can determine if the sampled direction goes through the surface by looking
      // if the direction goes in the opposite direction of the normal.
      medium_tracker.goingThroughSurface(dir_of_travel, node.interaction.surface);
    }
  }

  
  void InitializeMediumTracker(const RW::PathNode &source_node, MediumTracker &medium_tracker) 
  {
    medium_tracker.initializePosition(
      source_node.node_type != RW::NodeType::ENV ? source_node.coordinates() : Double3{Infinity});
  }
  
  
  RW::PathNode SampleEmissiveEnd(const PathContext &context, Pdf &pdf)
  {
    RW::PathNode node;
    light_picker.PickLight(sampler, EmitterSampleVisitor{context, scene, sampler, pdf, node});
    return node;
  }
  

  RW::PathNode SampleSensorEnd(int unit_index, const PathContext &context, Pdf &pdf)
  {
    RW::PathNode node;
    const ROI::PointEmitterArray& camera = scene.GetCamera();
    auto smpl = camera.TakePositionSample(unit_index, sampler, context);
    node.node_type   = RW::NodeType::CAMERA;
    node.interaction.emitter_point_array = RW::PointEmitterArrayInteraction{
      smpl.coordinates,
      unit_index,
      camera
    };
    pdf = smpl.pdf_or_pmf;
    return node;
  }
  
  
  // Generate y_i+1 given y_i and an incident direction, stored in node.
  ScatterSample SampleScatterCoordinateOfScatterer(const PathNode &node, const PathContext &context)
  {
    const Double3 reverse_incident_dir = -node.incident_dir;
    if (node.IsSurfaceInteraction())
    {
      const SurfaceInteraction &intersection = node.interaction.surface;
      auto smpl = GetShaderOf(intersection,scene).SampleBSDF(reverse_incident_dir, intersection, sampler, context);
      smpl.value *= BsdfCorrectionFactor(reverse_incident_dir, intersection, smpl.coordinates, context.transport);
      return smpl;
    }
    else // is Volume
    {
      const VolumeInteraction &interaction = node.interaction.volume;
      auto smpl = interaction.medium().SamplePhaseFunction(reverse_incident_dir, interaction.pos, sampler, context);
      smpl.value *= interaction.sigma_s;
      return smpl;
    }
  }
  
  
  // 
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
    else if (node.node_type == NodeType::SURFACE_EMITTER)
    {
      const SurfaceInteraction &interaction = node.interaction.surface;
      const auto* emitter = ASSERT_NOT_NULL(GetMaterialOf(interaction, scene).emitter);
      auto smpl = emitter->TakeDirectionSampleFrom(interaction.hitid, sampler, context);
      return smpl.as<ScatterSample>();
    }
    else if (node.node_type == NodeType::POINT_LIGHT)
    {
      const PointEmitterInteraction &interaction = node.interaction.emitter_point;
      auto smpl = interaction.emitter->TakeDirectionSampleFrom(interaction.pos, sampler, context);
      return smpl.as<ScatterSample>();
    }
    else if (node.node_type == NodeType::VOLUME_EMITTER)
    {
      const VolumeInteraction &interaction = node.interaction.volume;
      return ScatterSample{
        SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare()),
        interaction.radiance,
        1./(4.*Pi),
      };
    }
    else
      return {};
  }
  
 
  /* Straight forwardly following Cpt 5.3 Veach's thesis.
   * Basically when multiplied, it turns fs(wi -> wo, Nshd) into the corrected BSDF fs(wi->wo, Nsh)*Dot(Nshd,wi)/Dot(N,wi),
   * where wi refers to the incident direction of light as per Eq. 5.17.
   * 
   * Except that my conventions are different. In/out directions refer to the random walk. */
  static double BsdfCorrectionFactor(const Double3 &reverse_incident_dir, const SurfaceInteraction &intersection, const Double3 &exitant_dir, TransportType transport)
  {
    const Double3 &light_incident = transport==RADIANCE ? exitant_dir : reverse_incident_dir;
    double correction = std::abs(Dot(intersection.shading_normal, light_incident))/
                        (std::abs(Dot(intersection.normal, light_incident))+Epsilon);
    return correction;
  }
   
  double GetPdfOfGeneratingSampleOnEmitter(const RW::PathNode &node, const PathContext &context) const
  {
    if (node.node_type == RW::NodeType::SURFACE_EMITTER)
    {
      const SurfaceInteraction &interaction = node.interaction.surface;
      const auto &emitter = *ASSERT_NOT_NULL(GetMaterialOf(interaction, scene).emitter);
      double pdf = emitter.EvaluatePdf(node.interaction.surface.hitid, context);
      pdf *= light_picker.PmfOfLight(emitter);
      return pdf;
    }
    else if(node.node_type == RW::NodeType::VOLUME_EMITTER)
    {
      const VolumeInteraction &interaction = node.interaction.volume;
      double pdf = 0;
      interaction.medium().EvaluateEmission(interaction.pos, context, &pdf);
      pdf *= light_picker.PmfOfLight(interaction.medium());
      return pdf;
    }
    else if(node.node_type == RW::NodeType::ENV)
    {
      const EnvironmentalRadianceFieldInteraction &interaction = node.interaction.emitter_env;
      double pdf = interaction.emitter->EvaluatePdf(interaction.pos, context);
      pdf *= light_picker.PmfOfLight(*interaction.emitter);
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
    static constexpr int MIN_NODE_COUNT = 5;
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
    if (node.IsSurfaceInteraction())
    {
      // TODO: Refactor code so that this formula goes together with BSDF corrections
      // into a common "kernel" following Veach Fig. 5.8.
      ASSERT_NORMALIZED(node.geometry_normal());
      return std::abs(Dot(node.geometry_normal(), exitant_dir));
    }
    return 1.;
  } 
  
  
  void AddAntiSelfIntersectionOffsetAt(Double3 &position, const RW::PathNode &node, const Double3 &exitant_dir) const
  {
    if (node.IsSurfaceInteraction())
    {
      position += AntiSelfIntersectionOffset(node.interaction.surface, exitant_dir);
    }
  }

  
  Spectral3 TransmittanceEstimate(RaySegment seg, MediumTracker &medium_tracker, const PathContext &context, VolumePdfCoefficients *volume_pdf_coeff = nullptr)
  {
    static constexpr int MAX_ITERATIONS = 100; // For safety, in case somethign goes wrong with the intersections ...
    Spectral3 result{1.};
    SurfaceInteraction intersection;
    seg.length *= 0.99999; // Dammit!
    IterateIntersectionsBetween iter{seg, scene};
    for (int n = 0; n < MAX_ITERATIONS; ++n)
    {
      bool hit = iter.Next(seg, intersection);
      if (hit)
      {
        result *= GetShaderOf(intersection, scene).EvaluateBSDF(-seg.ray.dir, intersection, seg.ray.dir, context, nullptr);
        if (result.isZero())
          break;
      }
      if (volume_pdf_coeff)
      {
        VolumePdfCoefficients local_coeff = medium_tracker.getCurrentMedium().ComputeVolumePdfCoefficients(seg, context);
        Accumulate(*volume_pdf_coeff, local_coeff, n == 0, !hit);
      }
      result *= medium_tracker.getCurrentMedium().EvaluateTransmission(seg, sampler, context);
      if (hit)
      {
        medium_tracker.goingThroughSurface(seg.ray.dir, intersection);
      }
      if (!hit)
        break;
      assert (n < MAX_ITERATIONS-1);
    }
    return result;
  }


  /* Ray is the ray to shoot. It must already include the anti-self-intersection offset.
   */
  template<class VolumeHitVisitor, class SurfaceHitVisitor, class EscapeVisitor>
  bool TrackToNextInteraction(
    const Ray &ray,
    const PathContext &context,
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
          volume_visitor(interaction, prev_t+medium_smpl.t, total_weight);
          return true;
        }
        else if (hit)
        {
          surface_visitor(intersection, iter.GetT(), total_weight);
          return true;
        }
        break; // Escaped the scene.
      }
      assert (interfaces_crossed < MAX_ITERATIONS-1);
    }
    
    escape_visitor(total_weight);
    return false;
  }
};


} // namespace RandomWalk




#pragma GCC diagnostic pop // Restore command line options
