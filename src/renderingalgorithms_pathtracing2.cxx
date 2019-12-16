#include <functional>

#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/flow_graph.h>

#include "scene.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "camera.hxx"
#include "renderbuffer.hxx"
#include "util_thread.hxx"
#include "rendering_util.hxx"
#include "pathlogger.hxx"
#include "spectral.hxx"

#include "renderingalgorithms_interface.hxx"
#include "renderingalgorithms_simplebase.hxx"
#include "lightpicker_ucb.hxx"

namespace pathtracing2
{
using SimplePixelByPixelRenderingDetails::SamplesPerPixelSchedule;
using Lights::LightRef;
using Lightpickers::UcbLightPicker;
class PathTracingAlgo2;

class LightPickerUcbBufferedQueue
{
private:
  template<class T>
  using Buffer = Span<T>;

  template<class T>
  Span<T> CopyToSpan(const ToyVector<T> &v)
  {
    auto mem = std::make_unique<T[]>(v.size());
    memcpy(mem.get(), v.data(), sizeof(T)*v.size());
    return Span<T>(mem.release(), v.size());
  }

public:
  struct ThreadLocal
  {
    ToyVector<std::pair<Lights::LightRef, float>> nee_returns;
  };

private:
  static constexpr size_t BUFFER_SIZE = 10240;
  tbb::flow::graph graph;
  tbb::flow::function_node<Buffer<std::pair<LightRef, float>>> node_nee;
  ToyVector<ThreadLocal, tbb::cache_aligned_allocator<ThreadLocal>> local;
  UcbLightPicker picker_nee;
  
public:
  LightPickerUcbBufferedQueue(const Scene &scene, int num_workers)
    : local(num_workers),
    picker_nee(scene),
    graph(),
    node_nee(this->graph, 1, [&](Buffer<std::pair<LightRef, float>> x) {  picker_nee.ObserveReturns(x); delete x.begin();  })
  {

  }

  void ObserveReturnNee(ThreadLocal &l, LightRef lr, const Spectral3 &value)
  {
    l.nee_returns.push_back(std::make_pair(lr, Lightpickers::RadianceToObservationValue(value)));
    if (l.nee_returns.size() >= BUFFER_SIZE)
    {
      node_nee.try_put(CopyToSpan(l.nee_returns));
      l.nee_returns.clear();
    }
  }

  void ComputeDistribution()
  {
    graph.wait_for_all();

    picker_nee.ComputeDistribution();
    std::cout << "NEE LP: ";
    picker_nee.Distribution().Print(std::cout);
  }

  const Lightpickers::LightSelectionProbabilityMap& GetDistributionNee() const { return picker_nee.Distribution(); }
};




struct PathState
{
  PathState(const Scene &scene, const LambdaSelection &lambda_selection)
    : medium_tracker{ scene },
    context{ lambda_selection }
  {}

  MediumTracker medium_tracker;
  PathContext context;
  Ray ray;
  Spectral3 weight;
  boost::optional<Pdf> last_scatter_pdf_value; // For MIS.
  int pixel_index;
  int current_node_count;
  bool monochromatic;
};


class alignas(128) CameraRenderWorker
{
  const PathTracingAlgo2 * const master;
  LightPickerUcbBufferedQueue* const pickers;
  mutable LightPickerUcbBufferedQueue::ThreadLocal picker_local;
  mutable Sampler sampler;
  mutable Span<RGB> framebuffer;
  RayTermination ray_termination;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  static constexpr int num_lambda_sweeps = decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED;
  const int worker_index;

private:
  void InitializePathState(PathState &p, Int2 pixel, const LambdaSelection &lambda_selection) const;
  bool TrackToNextInteractionAndRecordPixel(PathState &ps) const;

  bool MaybeScatter(const SurfaceInteraction &interaction, PathState &ps) const;
  bool MaybeScatter(const VolumeInteraction &interaction, PathState &ps) const;

  void MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps) const;
  void AddEnvEmission(const PathState &ps) const;
  
  void AddDirectLighting(const SurfaceInteraction &interaction, const PathState &ps) const;
  void AddDirectLighting(const VolumeInteraction &interaction, const PathState &ps) const;
  
  void RecordMeasurementToCurrentPixel(const Spectral3 &measurement, const PathState &ps) const;

public:
  CameraRenderWorker(PathTracingAlgo2 *master, int worker_index);
  void Render(const ImageTileSet::Tile & tile);

};



class PathTracingAlgo2 : public RenderingAlgo
{
  friend class PhotonmappingWorker;
  friend class CameraRenderWorker;
private:
  ToyVector<RGB> framebuffer;
  ImageTileSet tileset;
  ToyVector<std::uint64_t> samplesPerTile; // False sharing!

  int num_pixels = 0;
  SamplesPerPixelSchedule spp_schedule;
  tbb::task_group the_task_group;
  tbb::task_arena the_task_arena;
  tbb::atomic<bool> stop_flag = false;
  ToyVector<CameraRenderWorker> camerarender_workers;
  std::unique_ptr<LightPickerUcbBufferedQueue> pickers;

public:
  const RenderingParameters &render_params;
  const Scene &scene;

public:
  PathTracingAlgo2(const Scene &scene_, const RenderingParameters &render_params_);

  void Run() override;

  // Will potentially be called before Run is invoked!
  void RequestFullStop() override
  {
    stop_flag.store(true);
    the_task_group.cancel();
  }

  // Will potentially be called before Run is invoked!
  void RequestInterrupt() override
  {
    the_task_group.cancel();
  }

  std::unique_ptr<Image> GenerateImage() override;

protected:
  inline int GetNumPixels() const { return num_pixels; }
  inline int GetSamplesPerPixel() const { return spp_schedule.GetPerIteration(); }
  inline int NumThreads() const {
    return the_task_arena.max_concurrency();
  }
};



PathTracingAlgo2::PathTracingAlgo2(
  const Scene &scene_, const RenderingParameters &render_params_)
  :
  RenderingAlgo{},
  spp_schedule{ render_params_ }, render_params{ render_params_ }, scene{ scene_ },
  tileset({ render_params_.width, render_params_.height })
{
  the_task_arena.initialize(std::max(1, this->render_params.num_threads));
  num_pixels = render_params.width * render_params.height;

  framebuffer.resize(num_pixels, RGB{});
  samplesPerTile.resize(tileset.size(), 0);

  pickers = std::make_unique<LightPickerUcbBufferedQueue>(scene, NumThreads());

  for (int i = 0; i < the_task_arena.max_concurrency(); ++i)
  {
    camerarender_workers.emplace_back(this, i);
  }
}





inline void PathTracingAlgo2::Run()
{
  Sampler sampler;

  while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
  {
    pickers->ComputeDistribution();

    the_task_arena.execute([this] {
      parallel_for_interruptible(0, this->tileset.size(), 1, [this](int i)
      {
        const int worker_num = tbb::this_task_arena::current_thread_index();
        camerarender_workers[worker_num].Render(this->tileset[i]);
        this->samplesPerTile[i] += spp_schedule.GetPerIteration();
      },
        /*irq_handler=*/[this]() -> bool
      {
        if (this->stop_flag.load())
          return false;
        this->CallInterruptCb(false);
        return true;
      }, the_task_group);
    });
    
    std::cout << "Sweep " << spp_schedule.GetTotal() << " finished" << std::endl;
    CallInterruptCb(true);
    spp_schedule.UpdateForNextPass();
  } // Pass iteration
}


inline std::unique_ptr<Image> PathTracingAlgo2::GenerateImage()
{
  auto bm = std::make_unique<Image>(render_params.width, render_params.height);
  tbb::parallel_for(0, tileset.size(), [this, bm{ bm.get() }](int i) {
    const auto tile = tileset[i];
    if (this->samplesPerTile[i] > 0)
    {
      framebuffer::ToImage(*bm, tile, AsSpan(this->framebuffer), this->samplesPerTile[i]);
    }
  });
  return bm;
}




CameraRenderWorker::CameraRenderWorker(PathTracingAlgo2* master, int worker_index)
  : master{ master },
  pickers{ master->pickers.get() },
  worker_index{ worker_index },
  ray_termination{ master->render_params }
{
  framebuffer = AsSpan(master->framebuffer);
}


void CameraRenderWorker::Render(const ImageTileSet::Tile &tile)
{
  const Int2 end = tile.corner + tile.shape;
  const int image_width = master->render_params.width;
  const int samples_per_pixel = master->GetSamplesPerPixel();

  for (int lambda_sweep = 0; lambda_sweep < num_lambda_sweeps; ++lambda_sweep)
  {
    const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    PathState state{ master->scene,  lambda_selection};
    for (int iy = tile.corner[1]; iy < end[1]; ++iy)
    {
      for (int ix = tile.corner[0]; ix < end[0]; ++ix)
      {
        for (int i = 0; i < samples_per_pixel; ++i)
        {
          InitializePathState(state, { ix, iy }, lambda_selection);
          bool keepgoing = true;
          do
          {
            keepgoing = TrackToNextInteractionAndRecordPixel(state);
          } while (keepgoing);
        }
      }
    }
  }
}




void CameraRenderWorker::InitializePathState(PathState &p, Int2 pixel, const LambdaSelection &lambda_selection) const
{
  p.context.pixel_x = pixel[0];
  p.context.pixel_y = pixel[1];
  p.current_node_count = 2; // First node on camera. We start with the node code of the next interaction.
  p.monochromatic = false;
  p.weight = lambda_selection.weights / num_lambda_sweeps;
  p.last_scatter_pdf_value = boost::none;


  const auto& camera = master->scene.GetCamera();
  p.pixel_index = camera.PixelToUnit({ pixel[0], pixel[1] });
  auto pos = camera.TakePositionSample(p.pixel_index, sampler, p.context);
  p.weight *= pos.value / pos.pdf_or_pmf;
  auto dir = camera.TakeDirectionSampleFrom(p.pixel_index, pos.coordinates, sampler, p.context);
  p.weight *= dir.value / dir.pdf_or_pmf;
  p.ray = { pos.coordinates, dir.coordinates };

  p.medium_tracker.initializePosition(pos.coordinates);
}


bool CameraRenderWorker::TrackToNextInteractionAndRecordPixel(PathState &ps) const
{
  const bool keepgoing = TrackToNextInteraction(master->scene, ps.ray, ps.context, ps.weight, sampler, ps.medium_tracker, nullptr,
    /*surface_visitor=*/[&ps, this](const SurfaceInteraction &interaction, double tfar, const Spectral3 &track_weight) -> bool
  {
    ps.weight *= track_weight;
    MaybeAddEmission(interaction, ps);
    AddDirectLighting(interaction, ps);
    ps.current_node_count++;
    return MaybeScatter(interaction, ps);
  },
    /*volume visitor=*/[&ps, this](const VolumeInteraction &vi, double tfar, const Spectral3 &weight) -> bool
  {
    AddDirectLighting(vi, ps);
    return MaybeScatter(vi, ps);
  },
    /* escape_visitor=*/[&ps, this](const Spectral3 &track_weight) -> bool
  {
    ps.weight *= track_weight;
    AddEnvEmission(ps);
    return false;
  }
  );
  return keepgoing;
}


static double MisWeight(Pdf pdf_or_pmf_taken, double pdf_other)
{
  double mis_weight = 1.;
  if (!pdf_or_pmf_taken.IsFromDelta())
  {
    mis_weight = PowerHeuristic(pdf_or_pmf_taken, { pdf_other });
  }
  return mis_weight;
}


void CameraRenderWorker::AddDirectLighting(const SurfaceInteraction &interaction, const PathState &ps) const
{
  // To consider for direct lighting NEE with MIS.
  // Segment, DFactors, AntiselfIntersectionOffset, Medium Passage, Transmittance Estimate, BSDF/Le factor, inv square factor.
  // Mis weight. P_light, P_bsdf

  const auto &shader = GetShaderOf(interaction, master->scene);

  RaySegment segment_to_light;
  Pdf pdf;
  Spectral3 light_weight;
  LightRef light_ref;

  pickers->GetDistributionNee().Sample(sampler, [&](auto &&light, double prob, const LightRef &light_ref_)
  {
    std::tie(segment_to_light, pdf, light_weight) = light.SampleConnection(SomeInteraction{ interaction }, master->scene, sampler, ps.context);
    light_weight /= ((double)(pdf)*prob);
    if constexpr (!std::remove_reference<decltype(light)>::type::IsAngularDistribution())
    {
      light_weight /= Sqr(segment_to_light.length);
      // For MIS combination with BSDF sampling, the pdf is converted to solid angle at the point of incidence.
      pdf =
        PdfConversion::AreaToSolidAngle(
          segment_to_light.length,
          segment_to_light.ray.dir,
          light.SurfaceNormal()) * pdf;
    }
    light_ref = light_ref_;
  });
  auto[ray, length] = segment_to_light;

  Spectral3 path_weight = ps.weight;
  path_weight *= DFactorPBRT(interaction, ray.dir);

  double bsdf_pdf = 0.;
  Spectral3 bsdf_weight = shader.EvaluateBSDF(-ps.ray.dir, interaction, ray.dir, ps.context, &bsdf_pdf);
  path_weight *= bsdf_weight;


  MediumTracker medium_tracker{ ps.medium_tracker }; // Copy because well don't want to keep modifications.
  MaybeGoingThroughSurface(medium_tracker, ray.dir, interaction);

  Spectral3 transmittance = TransmittanceEstimate(master->scene, segment_to_light, medium_tracker, ps.context, sampler);
  path_weight *= transmittance;

  double mis_weight = MisWeight(pdf, bsdf_pdf);

  Spectral3 measurement_estimator = mis_weight * path_weight*light_weight;

  pickers->ObserveReturnNee(this->picker_local, light_ref, measurement_estimator);
  RecordMeasurementToCurrentPixel(measurement_estimator, ps);
}


void pathtracing2::CameraRenderWorker::AddDirectLighting(const VolumeInteraction & interaction, const PathState & ps) const
{
  RaySegment segment_to_light;
  Pdf pdf;
  Spectral3 light_weight;
  LightRef light_ref;

  pickers->GetDistributionNee().Sample(sampler, [&](auto &&light, double prob, const LightRef &light_ref_)
  {
    std::tie(segment_to_light, pdf, light_weight) = light.SampleConnection(SomeInteraction{ interaction }, master->scene, sampler, ps.context);
    light_weight /= ((double)(pdf)*prob);
    if constexpr (!std::remove_reference<decltype(light)>::type::IsAngularDistribution())
    {
      light_weight /= Sqr(segment_to_light.length);
      // For MIS combination with BSDF sampling, the pdf is converted to solid angle at the point of incidence.
      pdf =
        PdfConversion::AreaToSolidAngle(
          segment_to_light.length,
          segment_to_light.ray.dir,
          light.SurfaceNormal()) * pdf;
    }
    light_ref = light_ref_;
  });
  auto[ray, length] = segment_to_light;

  Spectral3 path_weight = ps.weight;

  double bsdf_pdf = 0.;
  Spectral3 bsdf_weight = interaction.medium().EvaluatePhaseFunction(-ps.ray.dir, interaction.pos, ray.dir, ps.context, &bsdf_pdf);
  path_weight *= bsdf_weight;

  MediumTracker medium_tracker{ ps.medium_tracker }; // Copy because well don't want to keep modifications.

  Spectral3 transmittance = TransmittanceEstimate(master->scene, segment_to_light, medium_tracker, ps.context, sampler);
  path_weight *= transmittance;

  double mis_weight = MisWeight(pdf, bsdf_pdf);

  Spectral3 measurement_estimator = mis_weight * path_weight*light_weight;

  pickers->ObserveReturnNee(this->picker_local, light_ref, measurement_estimator);
  RecordMeasurementToCurrentPixel(measurement_estimator, ps);
}



bool CameraRenderWorker::MaybeScatter(const SurfaceInteraction &interaction, PathState &ps) const
{
  const auto &shader = GetShaderOf(interaction, master->scene);
  auto smpl = shader.SampleBSDF(-ps.ray.dir, interaction, sampler, ps.context);

  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{ 1. }, ps.current_node_count, sampler))
  {
    ps.monochromatic |= shader.require_monochromatic;
    ps.weight *= smpl.value * DFactorPBRT(interaction, smpl.coordinates) / smpl.pdf_or_pmf;
    ps.ray.dir = smpl.coordinates;
    ps.ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ps.ray.dir);
    MaybeGoingThroughSurface(ps.medium_tracker, ps.ray.dir, interaction);
    ps.last_scatter_pdf_value = smpl.pdf_or_pmf;
    return true;
  }
  else
    return false;
}


bool pathtracing2::CameraRenderWorker::MaybeScatter(const VolumeInteraction & interaction, PathState & ps) const
{
  auto smpl = interaction.medium().SamplePhaseFunction(-ps.ray.dir, interaction.pos, sampler, ps.context);

  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{ 1. }, ps.current_node_count, sampler))
  {
    ps.weight *= smpl.value / smpl.pdf_or_pmf;
    ps.ray.dir = smpl.coordinates;
    ps.ray.org = interaction.pos;
    ps.last_scatter_pdf_value = smpl.pdf_or_pmf;
    return true;
  }
  else
    return false;
}




void CameraRenderWorker::MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps) const
{
  const auto emitter = GetMaterialOf(interaction, master->scene).emitter;
  if (!emitter)
    return;

  Spectral3 radiance = emitter->Evaluate(interaction.hitid, -ps.ray.dir, ps.context, nullptr);

  double mis_weight = 1.0;
  if (ps.last_scatter_pdf_value) // Should be set if this is secondary ray.
  {
    const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, interaction.hitid));
    const double area_pdf = emitter->EvaluatePdf(interaction.hitid, ps.context);
    const double pdf_cvt = PdfConversion::AreaToSolidAngle(Length(ps.ray.org - interaction.pos), ps.ray.dir, interaction.normal);
    mis_weight = MisWeight(*ps.last_scatter_pdf_value, prob_select*area_pdf*pdf_cvt);
  }

#ifdef DEBUG_BUFFERS 
  AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*weight_accum);
#endif
  RecordMeasurementToCurrentPixel(mis_weight*radiance*ps.weight, ps);
}


void CameraRenderWorker::AddEnvEmission(const PathState &ps) const
{
  if (!master->scene.HasEnvLight())
    return;

  const auto &emitter = master->scene.GetTotalEnvLight();
  const auto radiance = emitter.Evaluate(-ps.ray.dir, ps.context);

  double mis_weight = 1.0;
  if (ps.last_scatter_pdf_value) // Should be set if this is secondary ray.
  {
    const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, emitter));
    const double pdf_env = emitter.EvaluatePdf(-ps.ray.dir, ps.context);
    mis_weight = MisWeight(*ps.last_scatter_pdf_value, pdf_env*prob_select);
  }

  RecordMeasurementToCurrentPixel(mis_weight*radiance*ps.weight, ps);
}




inline Spectral3 MaybeReWeightToMonochromatic(const Spectral3 &w, bool monochromatic)
{
  return monochromatic ? Spectral3{ w[0] * 3._sp,0,0 } : w;
}


void CameraRenderWorker::RecordMeasurementToCurrentPixel(const Spectral3 &measurement, const PathState &ps) const
{
  auto color = Color::SpectralSelectionToRGB(measurement, ps.context.lambda_idx);
  framebuffer[ps.pixel_index] += color;
}



} // namespace pathtracing2



std::unique_ptr<RenderingAlgo> AllocatePathtracing2RenderingAlgo(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<pathtracing2::PathTracingAlgo2>(scene, params);
}
