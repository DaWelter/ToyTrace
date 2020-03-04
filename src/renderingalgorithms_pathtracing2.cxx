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

#if 1 // If I want to use the UCB light picker ...
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
  //ToyVector<ThreadLocal, tbb::cache_aligned_allocator<ThreadLocal>> local;
  UcbLightPicker picker_nee;
  
public:
  LightPickerUcbBufferedQueue(const Scene &scene, int num_workers)
    :
    graph(),
    node_nee(this->graph, 1, [&](Buffer<std::pair<LightRef, float>> x) {  picker_nee.ObserveReturns(x); delete x.begin();  }),
    //local(num_workers),
    picker_nee(scene)
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
#else
class LightPickerUcbBufferedQueue
{
public:
  struct ThreadLocal
  {
  };

private:
  Lightpickers::LightSelectionProbabilityMap distribution;

public:
  LightPickerUcbBufferedQueue(const Scene &scene, int num_workers)
    : distribution{ scene }
  {

  }

  void ObserveReturnNee(ThreadLocal &l, LightRef lr, const Spectral3 &value)
  {
  }

  void ComputeDistribution()
  {
  }

  const Lightpickers::LightSelectionProbabilityMap& GetDistributionNee() const { return distribution; }
};
#endif



struct PathState
{
  PathState(const Scene &scene)
    : medium_tracker{ scene },
    context{}
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

private:
  void InitializePathState(PathState &p, Int2 pixel, const LambdaSelection &lambda_selection) const;
  bool TrackToNextInteractionAndRecordPixel(PathState &ps) const;

  bool MaybeScatter(const SomeInteraction &interaction, PathState &ps) const;

  void MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps) const;
  void AddEnvEmission(const PathState &ps) const;
  
  void AddDirectLighting(const SomeInteraction &interaction, const PathState &ps) const;
  
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
  tileset({ render_params_.width, render_params_.height }),
  spp_schedule{ render_params_ }, 
  render_params{ render_params_ },
   scene{ scene_ }
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
  ray_termination{ master->render_params }
{
  framebuffer = AsSpan(master->framebuffer);
}


void CameraRenderWorker::Render(const ImageTileSet::Tile &tile)
{
  const Int2 end = tile.corner + tile.shape;
  const int image_width = master->render_params.width;
  const int samples_per_pixel = master->GetSamplesPerPixel();
  
  PathState state{ master->scene };
  for (int iy = tile.corner[1]; iy < end[1]; ++iy)
  {
    for (int ix = tile.corner[0]; ix < end[0]; ++ix)
    {
      for (int i = 0; i < samples_per_pixel; ++i)
      {
        const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
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




void CameraRenderWorker::InitializePathState(PathState &p, Int2 pixel, const LambdaSelection &lambda_selection) const
{
  p.context = PathContext(lambda_selection);
  p.context.pixel_x = pixel[0];
  p.context.pixel_y = pixel[1];
  p.current_node_count = 2; // First node on camera. We start with the node code of the next interaction.
  p.monochromatic = false;
  p.weight = lambda_selection.weights;
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
  // TODO: fix the nonsensical use of initial weight in Atmosphere Material!!
  auto[interaction, tfar, track_weight] = TrackToNextInteraction(master->scene, ps.ray, ps.context, Spectral3::Ones(), sampler, ps.medium_tracker, nullptr);
  ps.weight *= track_weight;

  if (auto si = std::get_if<SurfaceInteraction>(&interaction))
  {
      MaybeAddEmission(*si, ps);
      AddDirectLighting(*si, ps);
      ps.current_node_count++;
      return MaybeScatter(*si, ps);
  }
  else if (auto vi = std::get_if<VolumeInteraction>(&interaction))
  {
    AddDirectLighting(*vi, ps);
    ps.current_node_count++;
    return MaybeScatter(*vi, ps);
  }
  else
  {
    AddEnvEmission(ps);
    return false;
  }
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


void CameraRenderWorker::AddDirectLighting(const SomeInteraction & interaction, const PathState & ps) const
{
  RaySegment segment_to_light;
  Pdf pdf;
  Spectral3 light_weight;
  LightRef light_ref;

  pickers->GetDistributionNee().Sample(sampler, [&](auto &&light, double prob, const LightRef &light_ref_)
  {
    std::tie(segment_to_light, pdf, light_weight) = light.SampleConnection(interaction, master->scene, sampler, ps.context);
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

  MediumTracker medium_tracker{ ps.medium_tracker }; // Copy because well don't want to keep modifications.
  
  double bsdf_pdf = 0.;

  if (auto si = std::get_if<SurfaceInteraction>(&interaction))
  {
    // Surface specific
    path_weight *= DFactorPBRT(*si, ray.dir);
    Spectral3 bsdf_weight = GetShaderOf(*si, master->scene).EvaluateBSDF(-ps.ray.dir, *si, ray.dir, ps.context, &bsdf_pdf);
    path_weight *= bsdf_weight;
    MaybeGoingThroughSurface(medium_tracker, ray.dir, *si);
  }
  else // must be volume interaction
  {
    auto vi = std::get_if<VolumeInteraction>(&interaction);
    Spectral3 bsdf_weight = vi->medium().EvaluatePhaseFunction(-ps.ray.dir, vi->pos, ray.dir, ps.context, &bsdf_pdf);
    path_weight *= bsdf_weight;
    path_weight *= vi->sigma_s;
  }

  Spectral3 transmittance = TransmittanceEstimate(master->scene, segment_to_light, medium_tracker, ps.context, sampler);
  path_weight *= transmittance;

  double mis_weight = MisWeight(pdf, bsdf_pdf);

  Spectral3 measurement_estimator = mis_weight*path_weight*light_weight;

  pickers->ObserveReturnNee(this->picker_local, light_ref, measurement_estimator);
  RecordMeasurementToCurrentPixel(measurement_estimator, ps);
}

namespace
{


ScatterSample SampleScatterer(const SurfaceInteraction &interaction, const Double3 &reverse_incident_dir, const Scene &scene, Sampler &sampler, const PathContext &context)
{
  return GetShaderOf(interaction, scene).SampleBSDF(reverse_incident_dir, interaction, sampler, context);
}

ScatterSample SampleScatterer(const VolumeInteraction &interaction, const Double3 &reverse_incident_dir, const Scene &scene, Sampler &sampler, const PathContext &context)
{
  ScatterSample s = interaction.medium().SamplePhaseFunction(reverse_incident_dir, interaction.pos, sampler, context);
  s.value *= interaction.sigma_s;
  return s;
}

void PrepareStateForAfterScattering(PathState &ps, const SurfaceInteraction &interaction, const Scene &scene, const ScatterSample &scatter_sample)
{
  ps.monochromatic |= GetShaderOf(interaction, scene).require_monochromatic;
  ps.weight *= scatter_sample.value * DFactorPBRT(interaction, scatter_sample.coordinates) / scatter_sample.pdf_or_pmf;
  ps.ray.dir = scatter_sample.coordinates;
  ps.ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ps.ray.dir);
  MaybeGoingThroughSurface(ps.medium_tracker, ps.ray.dir, interaction);
  ps.last_scatter_pdf_value = scatter_sample.pdf_or_pmf;
}

void PrepareStateForAfterScattering(PathState &ps, const VolumeInteraction &interaction, const Scene &scene, const ScatterSample &scatter_sample)
{
  ps.weight *= scatter_sample.value / scatter_sample.pdf_or_pmf;
  ps.ray.dir = scatter_sample.coordinates;
  ps.ray.org = interaction.pos;
  ps.last_scatter_pdf_value = scatter_sample.pdf_or_pmf;
}


} // anonymous namespace


bool CameraRenderWorker::MaybeScatter(const SomeInteraction &interaction, PathState &ps) const
{
  auto smpl = std::visit([this, &ps, &scene{ master->scene }](auto &&ia) {
    return SampleScatterer(ia, -ps.ray.dir, scene, sampler, ps.context);
  }, interaction);

  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{ 1. }, ps.current_node_count, sampler))
  {
    std::visit([&ps, &scene{ master->scene }, &smpl](auto &&ia) {
      return PrepareStateForAfterScattering(ps, ia, scene, smpl);
    }, interaction);
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
  assert(measurement.isFinite().all());
  auto color = Color::SpectralSelectionToRGB(measurement, ps.context.lambda_idx);
  framebuffer[ps.pixel_index] += color;
}



} // namespace pathtracing2



std::unique_ptr<RenderingAlgo> AllocatePathtracing2RenderingAlgo(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<pathtracing2::PathTracingAlgo2>(scene, params);
}
