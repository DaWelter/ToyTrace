#include <functional>
#include <type_traits>

#include <tbb/atomic.h>
#include <tbb/mutex.h>
//#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/cache_aligned_allocator.h>
//#include <tbb/flow_graph.h>

#include <range/v3/view/transform.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/range/conversion.hpp>

#include <boost/filesystem/path.hpp>

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
#include "path_guiding.hxx"

namespace fs = boost::filesystem;

namespace {
Spectral3 ClampPathWeight(const Spectral3 &w) {
  //return w.min(10.);
  return w;
}
}


namespace pathtracing_guided
{
using SimplePixelByPixelRenderingDetails::SamplesPerPixelSchedule;
using Lights::LightRef;
using Lightpickers::UcbLightPicker;
class PathTracingAlgo2;

#if 0 // If I want to use the UCB light picker ...
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
    std::cout << "Sampling lights from ";
    distribution.Print(std::cout);
    std::cout << std::endl;
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

// Requried to compute the indirect incident radiance at every vertex
/*              Li(x)_nee (light source here)
                |
         T      |
   y ---------- x -------- .... >  tracing direction
              Le(x)  (surface at x may emit)

*/
struct VertexCoefficients
{
  Double3 pos;
  Double3 dir; // outgoing ray
  Double3 dir_nee;
  Double3 normal;
  double scatter_pdf = 0.;
  double nee_pdf_conversion_factor = 1.;
  // T(y, x)/p_T(y,x) where y is the previously visited vertex
  Spectral3 segment_transmission_from_prev{Eigen::zero};
  // rho(x) / p_rho(x). BSDF + Dfactors evaluated for the traced path (not NEE sampling)
  Spectral3 this_scatter_path{Eigen::zero};
  Spectral3 this_scatter_nee{Eigen::zero};
  // Emission Le(x)
  Spectral3 emission{Eigen::zero};
  // NEE quantities pertaining to the path segment to the sampled light and the light itself.
  Spectral3 nee_emission_times_transmission{Eigen::zero};
  bool specular = false;
  bool specular_nee = false;
  bool surface = false;
};

using PathCoefficients = ToyVector<VertexCoefficients>;


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
  guiding::PathGuiding* radiance_recorder_surface;
  guiding::PathGuiding* radiance_recorder_volume;
  mutable LightPickerUcbBufferedQueue::ThreadLocal picker_local;
  guiding::PathGuiding::ThreadLocal radrec_local_surface;
  guiding::PathGuiding::ThreadLocal radrec_local_volume;
  mutable Sampler sampler;
  mutable Span<RGB> framebuffer;
  RayTermination ray_termination;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  static constexpr int num_lambda_sweeps = decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED;
  bool enable_nee;

private:
  void InitializePathState(PathState &p, PathCoefficients &coeffs, Int2 pixel, const LambdaSelection &lambda_selection) const;
  bool TrackToNextInteractionAndRecordPixel(PathState &ps, PathCoefficients &coeffs) const;

  ScatterSample SampleScatterer(const SurfaceInteraction &interaction, const Double3 &reverse_incident_dir, const PathContext &context) const;
  ScatterSample SampleScatterer(const VolumeInteraction &interaction, const Double3 &reverse_incident_dir, const PathContext &context) const;
  bool MaybeScatter(const SomeInteraction &interaction, PathState &ps, PathCoefficients &coeffs) const;

  void MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps, PathCoefficients &coeffs) const;
  void AddEnvEmission(const PathState &ps, PathCoefficients &coeffs) const;
  
  void AddDirectLighting(const SomeInteraction &interaction, const PathState &ps, PathCoefficients &coeffs) const;
  
  void RecordMeasurementToCurrentPixel(const Spectral3 &measurement, const PathState &ps) const;

  void ProcessGuidingData(const PathCoefficients &coeffs, const PathState &ps);

public:
  CameraRenderWorker(PathTracingAlgo2 *master, int worker_index);
  void Render(const ImageTileSet::Tile & tile, const int samples_per_pixel);
  void PrepassRender(long sample_count);
  
  auto* GetGuidingLocalDataSurface() { return &radrec_local_surface;  }
  auto* GetGuidingLocalDataVolume() { return &radrec_local_volume; }
};



class PathTracingAlgo2 : public RenderingAlgo
{
  friend class CameraRenderWorker;
  friend class DebugRenderWorker;
private:
  ToyVector<RGB> framebuffer;
  ImageTileSet tileset;
  ToyVector<std::uint64_t> samplesPerTile; // False sharing!

  int num_pixels = 0;
  SamplesPerPixelSchedule spp_schedule;
  //SimplePixelByPixelRenderingDetails::SamplesPerPixelScheduleConstant spp_schedule;
  tbb::task_group the_task_group;
  tbb::task_arena the_task_arena;
  tbb::atomic<bool> stop_flag = false;
  ToyVector<CameraRenderWorker> camerarender_workers;
  std::unique_ptr<LightPickerUcbBufferedQueue> pickers;
  std::unique_ptr<guiding::PathGuiding> radiance_recorder_surface;
  std::unique_ptr<guiding::PathGuiding> radiance_recorder_volume;
  bool record_samples_for_guiding = true;

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

  void RenderRadianceEstimates(fs::path filename);

protected:
  inline int GetNumPixels() const { return num_pixels; }
  inline int NumThreads() const {
    return the_task_arena.max_concurrency();
  }
};


////////////////////////////////////////////////////////////////////////////
///  Debug Renderer
////////////////////////////////////////////////////////////////////////////


class alignas(128) DebugRenderWorker
{
  PathTracingAlgo2 const * master;
  guiding::PathGuiding const * radiance_recorder_surface;
  guiding::PathGuiding const * radiance_recorder_volume;
  LightPickerUcbBufferedQueue* const pickers;
  Sampler sampler;
  Span<RGB> framebuffer;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  static constexpr int num_lambda_sweeps = decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED;

public:
  DebugRenderWorker(PathTracingAlgo2* master, Span<RGB> framebuffer)
    : master{ master },
    radiance_recorder_surface{master->radiance_recorder_surface.get()},
    radiance_recorder_volume{master->radiance_recorder_volume.get()},
    pickers{master->pickers.get()},
    framebuffer{framebuffer}
  {
  }

  void Render(const ImageTileSet::Tile &tile)
  {
    //const auto lambda_selection = LambdaSelection{
      // Color::LambdaIdxClosestToRGBPrimaries(), 
      // Spectral3::Constant(3.), 
      // LambdaSelectionStrategy::SampleWavelengthStrata(Color::LambdaIdxClosestToRGBPrimaries(), sampler)};
    MediumTracker medium_tracker{master->scene};

    const Int2 end = tile.corner + tile.shape;
    for (int iy = tile.corner[1]; iy < end[1]; ++iy) 
    for (int ix = tile.corner[0]; ix < end[0]; ++ix)
    for (int lambda_sweep = 0; lambda_sweep < lambda_selection_factory.NUM_SAMPLES_REQUIRED; ++lambda_sweep)
    {
      const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
      PathContext context{lambda_selection};
      context.pixel_x = ix;
      context.pixel_y = iy;

      auto& camera = master->scene.GetCamera();
      auto pixel = camera.PixelToUnit({ix, iy});
      auto pos = camera.TakePositionSample(pixel, sampler, context);
      auto dir = camera.TakeDirectionSampleFrom(pixel, pos.coordinates, sampler, context);
      medium_tracker.initializePosition(pos.coordinates);

      Spectral3 measurement_estimator{Eigen::zero};

      Spectral3 weight = pos.value * dir.value * lambda_selection.weights / (Value(pos.pdf_or_pmf) * Value(dir.pdf_or_pmf));
      auto ray = Ray{ pos.coordinates, dir.coordinates };
      
      bool keep_going = true;
      for (int ray_depth = 0; ray_depth < master->render_params.max_ray_depth && keep_going; ++ray_depth)
      {
        auto[interaction, tfar, track_weight] = TrackToNextInteraction(master->scene, ray, context, Spectral3::Ones(), sampler, medium_tracker, nullptr);
        
        if (auto *si = std::get_if<SurfaceInteraction>(&interaction))
        {
          if (ray_depth == 0) // If camera ray hits an emissive surface ...
          {
            if (const auto *emitter = GetMaterialOf(*si, master->scene).emitter; emitter)
            {
              Spectral3 radiance = emitter->Evaluate(si->hitid, -ray.dir, context, nullptr);
              measurement_estimator += weight * track_weight * radiance;
            }
          }

          auto bsdf_sample = GetShaderOf(*si, master->scene).SampleBSDF(-ray.dir, *si, sampler, context);
          if (bsdf_sample.pdf_or_pmf.IsFromDelta())
          {
            // Sample next direction
            weight *= track_weight * bsdf_sample.value * DFactorPBRT(*si, bsdf_sample.coordinates) / bsdf_sample.pdf_or_pmf;
            ray.dir = bsdf_sample.coordinates;
            ray.org = si->pos + AntiSelfIntersectionOffset(*si, ray.dir);
            MaybeGoingThroughSurface(medium_tracker, ray.dir, *si);
          }
          else
          {
            // Stop on a diffuse surface
            auto [incident_radiance_estimator, segment_to_light] = ComputeDirectLighting(*si, medium_tracker, sampler, context);

            const double dfactor = DFactorPBRT(*si, segment_to_light.ray.dir);
            const Spectral3 bsdf_weight = GetShaderOf(*si, master->scene).EvaluateBSDF(-ray.dir, *si, segment_to_light.ray.dir, context, nullptr);
            measurement_estimator += weight * track_weight * bsdf_weight * dfactor * incident_radiance_estimator;

            const Spectral3 indirect_lighting = ComputeIndirectLighting(*si, ray.dir, sampler, context);
            measurement_estimator += weight * track_weight * indirect_lighting;

            keep_going = false;
          }
        }
        else if (auto *vi = std::get_if<VolumeInteraction>(&interaction))
        {
          // track weight ~= 1/sigma_t * 1/p_lambda = 100 * 12
          const Spectral3 radiance_estimate = ComputeInscatteredRadiance(*vi, ray.dir, sampler, context);
          // if (ix == 180 && iy == 180)
          //   std::cout << radiance_estimate << " " << vi->sigma_s << " " << (weight * track_weight) << std::endl;
          measurement_estimator += weight * track_weight * vi->sigma_s * radiance_estimate;
          keep_going = false;
        }
        else
          keep_going = false;
      }
      RecordMeasurementToCurrentPixel(measurement_estimator /= lambda_selection_factory.NUM_SAMPLES_REQUIRED, pixel, lambda_selection);
    }
  }


  Spectral3 ComputeIndirectLighting(const SurfaceInteraction &interaction, const Double3 &rev_incident_dir, Sampler &sampler, const PathContext &context)
  {
    const auto& estimate = radiance_recorder_surface->FindRadianceEstimate(interaction.pos);
    const auto& distribution = estimate.radiance_distribution;
    const auto& shader = GetShaderOf(interaction, master->scene);

    // Numerical convolution using Monte Carlo
    Spectral3 result{Eigen::zero};
    const int sample_count = 2;
    for (int i = 0; i<sample_count; ++i)
    {
      const Eigen::Vector3f dir = vmf_fitting::Sample(distribution, { sampler.Uniform01(), sampler.Uniform01(), sampler.Uniform01() });

      if (dir.dot(interaction.normal.cast<float>()) <= 0.)
        continue;

      // Drops out because Li = pdf * estimate.incident_flux_density. Dividing by pdf because Monte Carlo,
      // makes pdf drop out of the equation.
      //const float pdf = vmf_fitting::Pdf(distribution, dir);
      const Spectral3 bsdf_weight = shader.EvaluateBSDF(-rev_incident_dir, interaction, dir.cast<double>(), context, nullptr);
      result += bsdf_weight /*/ (double)pdf*/ * DFactorPBRT(interaction, dir.cast<double>());
    }
    // Bad approximation because the estimate incident flux is averaged over wavelengths!
    result *= estimate.incident_flux_density / sample_count;
    return result;
  }


  Spectral3 ComputeInscatteredRadiance(const VolumeInteraction &interaction, const Double3& /*rev_incident_dir*/, Sampler& /*sampler*/, const PathContext& /*context*/)
  {
    const auto& estimate = radiance_recorder_volume->FindRadianceEstimate(interaction.pos);
    Spectral3 result = Spectral3::Constant(estimate.incident_flux_density / UnitSphereSurfaceArea);
    return result;
  }


  void RecordMeasurementToCurrentPixel(const Spectral3 &measurement, int pixel_index, const LambdaSelection &lambda_selection)
  {
    assert(measurement.isFinite().all());
    auto color = Color::SpectralSelectionToRGB(measurement, lambda_selection.indices);
    framebuffer[pixel_index] += color;
  }

  std::tuple<Spectral3, RaySegment> ComputeDirectLighting(const SurfaceInteraction & interaction, const MediumTracker &medium_tracker, Sampler &sampler, const PathContext &context) const
  {
    RaySegment segment_to_light;
    Pdf pdf;
    Spectral3 light_radiance{};
    Spectral3 path_weight = Spectral3::Ones();

    pickers->GetDistributionNee().Sample(sampler, [&](auto &&light, double prob, const LightRef &light_ref_)
    {
      std::tie(segment_to_light, pdf, light_radiance) = light.SampleConnection(interaction, master->scene, sampler, context);
      path_weight /= ((double)(pdf)*prob);
      if constexpr (!std::remove_reference<decltype(light)>::type::IsAngularDistribution())
      {
        path_weight *= 1./Sqr(segment_to_light.length);
      }
    });
    auto[ray, length] = segment_to_light;

    MediumTracker medium_tracker_copy{ medium_tracker }; // Copy because well don't want to keep modifications.
    
    // Surface specific
    MaybeGoingThroughSurface(medium_tracker_copy, ray.dir, interaction);

    Spectral3 transmittance = TransmittanceEstimate(master->scene, segment_to_light, medium_tracker_copy, context, sampler);
    path_weight *= transmittance;

    Spectral3 incident_radiance_estimator = path_weight*light_radiance;
    return std::make_tuple(incident_radiance_estimator, segment_to_light);
  }

}; // class DebugRenderWorker


void PathTracingAlgo2::RenderRadianceEstimates(fs::path filename)
{
  the_task_arena.execute([this, &filename] ()
  {
    std::cout << "Rendering Radiance Estimates " << filename << std::endl;

    auto num_pixels = render_params.width * render_params.height;
    ToyVector<RGB> framebuffer(num_pixels);

    ToyVector<DebugRenderWorker> workers; workers.reserve(the_task_arena.max_concurrency());
    for (int i = 0; i<the_task_arena.max_concurrency(); ++i)
      workers.emplace_back(this, AsSpan(framebuffer));

    tbb::parallel_for(0, this->tileset.size(), 1, [this, &workers](int i)
    {
      const int worker_num = tbb::this_task_arena::current_thread_index();
      workers[worker_num].Render(this->tileset[i]);
    });

    auto bm = Image(render_params.width, render_params.height);
    tbb::parallel_for(0, tileset.size(), [this, &bm, fb=AsSpan(framebuffer)](int i) {
      const auto tile = tileset[i];
      framebuffer::ToImage(bm, tile, fb, 1, false);
    });

    bm.write(filename.string());
    std::cout << "Done" << std::endl;
  });
}


////////////////////////////////////////////////////////////////////////////
///  The alog
////////////////////////////////////////////////////////////////////////////


PathTracingAlgo2::PathTracingAlgo2(
  const Scene &scene_, const RenderingParameters &render_params_)
  :
  RenderingAlgo{}, 
  tileset({ render_params_.width, render_params_.height }),
  spp_schedule{ render_params_ },
  render_params{ render_params_ }, scene{ scene_ }
{
  the_task_arena.initialize(std::max(1, this->render_params.num_threads));
  num_pixels = render_params.width * render_params.height;

  framebuffer.resize(num_pixels, RGB{});
  samplesPerTile.resize(tileset.size(), 0);

  pickers = std::make_unique<LightPickerUcbBufferedQueue>(scene, NumThreads());

  radiance_recorder_surface = std::make_unique<guiding::PathGuiding>(scene.GetBoundingBox(), 0.1, render_params_, the_task_arena);
  radiance_recorder_volume = std::make_unique<guiding::PathGuiding>(scene.GetBoundingBox(), 0.1, render_params_, the_task_arena);

  for (int i = 0; i < the_task_arena.max_concurrency(); ++i)
  {
    camerarender_workers.emplace_back(this, i);
  }
}


inline void PathTracingAlgo2::Run()
{
  #ifdef WRITE_DEBUG_OUT
  scene.WriteObj(guiding::GetDebugFilePrefix() / boost::filesystem::path("scene.obj"));
  #endif

  RenderRadianceEstimates(guiding::GetDebugFilePrefix() / fs::path{"guiderad_initial.png"});

  std::vector<guiding::PathGuiding::ThreadLocal*> radrec_local_surface = 
    camerarender_workers | ranges::views::transform([](auto &w) { return w.GetGuidingLocalDataSurface();  }) | ranges::to_vector;
  std::vector<guiding::PathGuiding::ThreadLocal*> radrec_local_volume = 
    camerarender_workers | ranges::views::transform([](auto &w) { return w.GetGuidingLocalDataVolume();  }) | ranges::to_vector;

  {
    const long num_samples_stop = this->num_pixels;
    long num_samples = 32 * 32;
    while (!stop_flag.load() && (num_samples < num_samples_stop))
    {
      std::cout << strconcat("Prepass: ", num_samples, " / ", num_samples_stop, "\n");
      
      radiance_recorder_surface->BeginRound(AsSpan(radrec_local_surface));
      radiance_recorder_volume->BeginRound(AsSpan(radrec_local_volume));

      the_task_arena.execute([this, num_samples] {
        tbb::parallel_for(tbb::blocked_range<long>(0l, num_samples, 32), [this](const tbb::blocked_range<long> &r)
        {
          const int worker_num = tbb::this_task_arena::current_thread_index();
          camerarender_workers[worker_num].PrepassRender(r.end()-r.begin());
        });
      });

      radiance_recorder_surface->FinalizeRound(AsSpan(radrec_local_surface));
      radiance_recorder_surface->WriteDebugData("surface_");
      radiance_recorder_surface->PrepareAdaptedStructures();
      radiance_recorder_volume->FinalizeRound(AsSpan(radrec_local_volume));
      radiance_recorder_volume->WriteDebugData("volume_");
      radiance_recorder_volume->PrepareAdaptedStructures();

      num_samples *= 2;
    }
  }

  RenderRadianceEstimates(guiding::GetDebugFilePrefix() / fs::path{"guiderad_prepass.png"});

  {
    long num_samples = 1;
    while (!stop_flag.load() && num_samples <= render_params.guiding_max_spp)
    {
      // Clear the frame buffer to get rid of samples from previous iterations
      // which are assumed to be worse than current samples.
      std::fill(framebuffer.begin(), framebuffer.end(), RGB::Zero());
      std::fill(samplesPerTile.begin(), samplesPerTile.end(), 0ul);

      radiance_recorder_surface->BeginRound(AsSpan(radrec_local_surface));
      radiance_recorder_volume->BeginRound(AsSpan(radrec_local_volume));

      pickers->ComputeDistribution();

      the_task_arena.execute([this, num_samples] {
        parallel_for_interruptible(0, this->tileset.size(), 1, [this, num_samples](int i)
        {
          const int worker_num = tbb::this_task_arena::current_thread_index();
          camerarender_workers[worker_num].Render(this->tileset[i], num_samples);
          this->samplesPerTile[i] = num_samples;
        },
          /*irq_handler=*/[this]() -> bool
        {
          if (this->stop_flag.load())
            return false;
          this->CallInterruptCb(false);
          return true;
        }, the_task_group);
      });
      
      radiance_recorder_surface->FinalizeRound(AsSpan(radrec_local_surface));
      radiance_recorder_surface->WriteDebugData("surface_");
      radiance_recorder_surface->PrepareAdaptedStructures();
      radiance_recorder_volume->FinalizeRound(AsSpan(radrec_local_volume));
      radiance_recorder_volume->WriteDebugData("volume_");
      radiance_recorder_volume->PrepareAdaptedStructures();

      std::cout << "Guiding sweep " << num_samples << " finished" << std::endl;
      CallInterruptCb(true);

      RenderRadianceEstimates(guiding::GetDebugFilePrefix() / fs::path{strconcat("guiderad_",num_samples,".png")});

      num_samples *= 2;
    } // Pass iteration
  }

  this->record_samples_for_guiding = false;

  while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
  {
    the_task_arena.execute([this] {
      parallel_for_interruptible(0, this->tileset.size(), 1, [this](int i)
      {
        const int worker_num = tbb::this_task_arena::current_thread_index();
        camerarender_workers[worker_num].Render(this->tileset[i], spp_schedule.GetPerIteration());
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
    the_task_arena.execute([this, &bm] {
    
    tbb::parallel_for(0, tileset.size(), [this, bm{ bm.get() }](int i) {
      const auto tile = tileset[i];
      if (this->samplesPerTile[i] > 0)
      {
        framebuffer::ToImage(*bm, tile, AsSpan(this->framebuffer), this->samplesPerTile[i], false);
      }
    });
  });
  return bm;
}


////////////////////////////////////////////////////////////////////////////
///  Camera worker
////////////////////////////////////////////////////////////////////////////


CameraRenderWorker::CameraRenderWorker(PathTracingAlgo2* master, int worker_index)
  : master{ master },
  pickers{ master->pickers.get() },
  radiance_recorder_surface{master->radiance_recorder_surface.get()},
  radiance_recorder_volume{master->radiance_recorder_volume.get()},
  ray_termination{ master->render_params },
  enable_nee{ true }
{
  framebuffer = AsSpan(master->framebuffer);
  if (master->render_params.pt_sample_mode == "bsdf")
  {
    enable_nee = false;
  }
}


void CameraRenderWorker::Render(const ImageTileSet::Tile &tile, const int samples_per_pixel)
{
  const Int2 end = tile.corner + tile.shape;
  const int image_width = master->render_params.width;
  PathCoefficients path_coeffs; path_coeffs.reserve(128);
  PathState state{ master->scene };
  for (int iy = tile.corner[1]; iy < end[1]; ++iy)
  {
    for (int ix = tile.corner[0]; ix < end[0]; ++ix)
    {
      for (int i = 0; i < samples_per_pixel; ++i)
      {
        const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
        InitializePathState(state, path_coeffs, { ix, iy }, lambda_selection);
        bool keepgoing = true;
        do
        {
          keepgoing = TrackToNextInteractionAndRecordPixel(state, path_coeffs);
        } while (keepgoing);
        if (master->record_samples_for_guiding)
          ProcessGuidingData(path_coeffs, state);
      }
    }
  }
}


void CameraRenderWorker::PrepassRender(long sample_count)
{
  PathCoefficients path_coeffs; path_coeffs.reserve(128);
  PathState state{ master->scene };
  while (sample_count-- > 0)
  {
    const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    int ix = sampler.UniformInt(0, master->render_params.width-1);
    int iy = sampler.UniformInt(0, master->render_params.height-1);
    InitializePathState(state, path_coeffs, { ix, iy }, lambda_selection);
    bool keepgoing = true;
    do
    {
      keepgoing = TrackToNextInteractionAndRecordPixel(state, path_coeffs);
    } while (keepgoing);
    if (master->record_samples_for_guiding)
      ProcessGuidingData(path_coeffs, state);
  }
}


void CameraRenderWorker::InitializePathState(PathState &p, PathCoefficients &coeffs, Int2 pixel, const LambdaSelection &lambda_selection) const
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

  coeffs.clear();
  coeffs.emplace_back();
  coeffs.back().this_scatter_path = p.weight;

  p.medium_tracker.initializePosition(pos.coordinates);
}


bool CameraRenderWorker::TrackToNextInteractionAndRecordPixel(PathState &ps, PathCoefficients &coeffs) const
{
  // TODO: fix the nonsensical use of initial weight in Atmosphere Material!!
  auto[interaction, tfar, track_weight] = TrackToNextInteraction(master->scene, ps.ray, ps.context, Spectral3::Ones(), sampler, ps.medium_tracker, nullptr);
  ps.weight *= track_weight;

  coeffs.emplace_back();
  coeffs.back().segment_transmission_from_prev = track_weight;

  if (auto si = std::get_if<SurfaceInteraction>(&interaction))
  {
      MaybeAddEmission(*si, ps, coeffs);
      if (enable_nee)
        AddDirectLighting(*si, ps, coeffs);
      ps.current_node_count++;
      return MaybeScatter(*si, ps, coeffs);
  }
  else if (auto vi = std::get_if<VolumeInteraction>(&interaction))
  {
    if (enable_nee)
      AddDirectLighting(*vi, ps, coeffs);
    ps.current_node_count++;
    return MaybeScatter(*vi, ps, coeffs);
  }
  else
  {
    AddEnvEmission(ps, coeffs);
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


void CameraRenderWorker::AddDirectLighting(const SomeInteraction & interaction, const PathState & ps, PathCoefficients &coeffs) const
{
  RaySegment segment_to_light;
  Pdf pdf;
  Spectral3 light_radiance{Eigen::zero};
  LightRef light_ref;
  double pdf_conversion_factor = 1.;

  pickers->GetDistributionNee().Sample(sampler, [&](auto &&light, double prob, const LightRef &light_ref_)
  {
    std::tie(segment_to_light, pdf, light_radiance) = light.SampleConnection(interaction, master->scene, sampler, ps.context);
    
    if constexpr (!std::remove_reference<decltype(light)>::type::IsAngularDistribution())
    {
      light_radiance /= prob * (double)pdf * Sqr(segment_to_light.length);
      // For MIS combination with BSDF sampling, the pdf is converted to solid angle at the point of incidence.
      if (!pdf.IsFromDelta())
      {
        pdf_conversion_factor = PdfConversion::AreaToSolidAngle(
            segment_to_light.length,
            segment_to_light.ray.dir,
            light.SurfaceNormal()) * pdf;
      }
    }
    light_ref = light_ref_;
  });
  auto[ray, length] = segment_to_light;

  MediumTracker medium_tracker{ ps.medium_tracker }; // Copy because well don't want to keep modifications.
  
  double bsdf_pdf = 0.;
  Spectral3 scatter_kernel;
  if (auto si = std::get_if<SurfaceInteraction>(&interaction))
  {
    // Surface specific
    scatter_kernel = GetShaderOf(*si, master->scene).EvaluateBSDF(-ps.ray.dir, *si, ray.dir, ps.context, &bsdf_pdf);
    scatter_kernel *= DFactorPBRT(*si, ray.dir);
    MaybeGoingThroughSurface(medium_tracker, ray.dir, *si);
  }
  else // must be volume interaction
  {
    auto vi = std::get_if<VolumeInteraction>(&interaction);
    scatter_kernel = vi->medium().EvaluatePhaseFunction(-ps.ray.dir, vi->pos, ray.dir, ps.context, &bsdf_pdf);
    scatter_kernel *= vi->sigma_s;
  }

  const Spectral3 transmittance = TransmittanceEstimate(master->scene, segment_to_light, medium_tracker, ps.context, sampler);

  const double mis_weight = MisWeight(pdf_conversion_factor*pdf, bsdf_pdf);

  Spectral3 measurement_estimator = mis_weight*ps.weight*scatter_kernel*transmittance*light_radiance;

  {
    auto& coeff = coeffs.back();
    coeff.nee_pdf_conversion_factor = pdf_conversion_factor;
    coeff.specular_nee = pdf.IsFromDelta();
    coeff.nee_emission_times_transmission = light_radiance*transmittance*mis_weight;
    coeff.this_scatter_nee = scatter_kernel;
    coeff.dir_nee = segment_to_light.ray.dir;
  }

  pickers->ObserveReturnNee(this->picker_local, light_ref, measurement_estimator);
  //RecordMeasurementToCurrentPixel(measurement_estimator, ps);
}

namespace
{


class GmmRefDistribution
{
  const vmf_fitting::VonMisesFischerMixture *mixture;
public:
  GmmRefDistribution(const vmf_fitting::VonMisesFischerMixture& mixture, const Double3 &) :
    mixture{ &mixture }
  {
  }

  Double3 Sample(Sampler &sampler)
  {
    auto r1 = sampler.Uniform01();
    auto r2 = sampler.Uniform01();
    auto r3 = sampler.Uniform01();
    dbg = { r1, r2, r3 };
    return vmf_fitting::Sample(*mixture, { r1,r2,r3 }).cast<double>();
  }

  double Pdf(const Double3 &dir) const
  {
    return vmf_fitting::Pdf(*mixture, dir.cast<float>());
  }

  std::array<double,3> dbg{};
};

#if 0
void Check(const vmf_fitting::VonMisesFischerMixture &mixture, const Double3 x, double pdf, double* rndvars, bool sample_came_from_this)
{
  static tbb::mutex m;
  bool ok = true;
  ok &= std::abs(x.norm() - 1.f) < 1.e-3;
  ok &= std::isfinite(pdf);
  if (sample_came_from_this)
    ok &= pdf > 0.f;
  if (!ok)
  {
    tbb::mutex::scoped_lock l{m};
    std::cerr << "Bad sample: " << (sample_came_from_this ? "from this " : " from elsewhere") << "\n";
    std::cerr << " x=" << (x);
    std::cerr << " pdf=" << (pdf);
    std::cerr << "\n";
    std::cerr << "RND: " << rndvars[0] << ", " << rndvars[1] << ", " << rndvars[2] << "\n";
    std::cerr << "Means:\n";
    std::cerr << mixture.means << '\n';
    std::cerr << "Conc:\n";
    std::cerr << mixture.concentrations << '\n';
    std::cerr << "Weights:\n";
    std::cerr << mixture.weights << '\n';
    std::cerr << std::endl;
    assert(ok);
  }
}
#endif


ScatterSample SampleWithBinaryMixtureDensity(
  const Shader &shader, const vmf_fitting::VonMisesFischerMixture& mixture, const SurfaceInteraction &interaction, const Double3 &reverse_incident_dir, 
  const Scene &scene, Sampler &sampler, const PathContext &context, double prob_bsdf)
{
  auto otherDistribution = GmmRefDistribution{mixture, interaction.normal };
  //auto otherDistribution = Rotated<UniformHemisphericalDistribution>(frame, UniformHemisphericalDistribution{});

  // This terrible hack won't work with a mixture of delta and continuous distribution.
  // I.e. clear coated plastic, etc ...
  ScatterSample smpl = shader.SampleBSDF(reverse_incident_dir, interaction, sampler, context);
  assert(reverse_incident_dir.array().isFinite().all() && interaction.normal.array().isFinite().all());

  if (smpl.pdf_or_pmf.IsFromDelta())
    return smpl;

  if (sampler.Uniform01() < prob_bsdf)
  {
    ScatterSample smpl = shader.SampleBSDF(reverse_incident_dir, interaction, sampler, context);
    if (smpl.pdf_or_pmf.IsFromDelta())
    {
      smpl.pdf_or_pmf *= prob_bsdf;
    }
    else
    {
      double mix_pdf = otherDistribution.Pdf(smpl.coordinates);
      smpl.pdf_or_pmf = Lerp(
        mix_pdf,
        (double)(smpl.pdf_or_pmf),
        prob_bsdf);
      //Check(mixture, smpl.coordinates, mix_pdf, otherDistribution.dbg.data(), false);
    }
    return smpl;
  }
  else
  {
    Double3 dir = otherDistribution.Sample(sampler);
    double mix_pdf = otherDistribution.Pdf(dir);
    //Check(mixture, dir, mix_pdf, otherDistribution.dbg.data(), true);
    double bsdf_pdf = 0.;
    Spectral3 bsdf_val = shader.EvaluateBSDF(reverse_incident_dir, interaction, dir, context, &bsdf_pdf);
    double pdf = Lerp(
      mix_pdf,
      bsdf_pdf,
      prob_bsdf);
    return {
      dir,
      bsdf_val,
      pdf
    };
  } 
}

void PrepareStateForAfterScattering(PathState &ps, PathCoefficients &coeff, const SurfaceInteraction &interaction, const Scene &scene, const ScatterSample &scatter_sample)
{
  ps.monochromatic |= GetShaderOf(interaction, scene).require_monochromatic;
  const Spectral3 scatter_factor = scatter_sample.value * DFactorPBRT(interaction, scatter_sample.coordinates) / scatter_sample.pdf_or_pmf;
  ps.weight *= scatter_factor;
  ps.ray.dir = scatter_sample.coordinates;
  ps.ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ps.ray.dir);
  MaybeGoingThroughSurface(ps.medium_tracker, ps.ray.dir, interaction);
  ps.last_scatter_pdf_value = scatter_sample.pdf_or_pmf;

  coeff.back().this_scatter_path = scatter_factor;
  coeff.back().normal = interaction.normal;
  coeff.back().pos = interaction.pos;
  coeff.back().scatter_pdf = (double)scatter_sample.pdf_or_pmf;
  coeff.back().specular = scatter_sample.pdf_or_pmf.IsFromDelta();
  coeff.back().surface = true;
  coeff.back().dir = ps.ray.dir;
}

void PrepareStateForAfterScattering(PathState &ps, PathCoefficients &coeff, const VolumeInteraction &interaction, const Scene &scene, const ScatterSample &scatter_sample)
{
  const Spectral3 scatter_factor = scatter_sample.value / scatter_sample.pdf_or_pmf;
  ps.weight *= scatter_factor;
  ps.ray.dir = scatter_sample.coordinates;
  ps.ray.org = interaction.pos;
  ps.last_scatter_pdf_value = scatter_sample.pdf_or_pmf;

  coeff.back().this_scatter_path = scatter_factor;
  coeff.back().normal = Double3::Zero();
  coeff.back().pos = interaction.pos;
  coeff.back().scatter_pdf = (double)scatter_sample.pdf_or_pmf;
  coeff.back().specular = false;
  coeff.back().surface = false;
  coeff.back().dir = ps.ray.dir;
}


} // anonymous namespace




ScatterSample CameraRenderWorker::SampleScatterer(
  const SurfaceInteraction &interaction, const Double3 &reverse_incident_dir,
  const PathContext &context) const
{
  const auto& mixture = this->radiance_recorder_surface->FindRadianceEstimate(interaction.pos).radiance_distribution;
  const double prob_bsdf = 0.5;
  return SampleWithBinaryMixtureDensity(
    GetShaderOf(interaction, master->scene), mixture,
    interaction, reverse_incident_dir, master->scene, sampler, context, prob_bsdf);
}


ScatterSample CameraRenderWorker::SampleScatterer(
  const VolumeInteraction &interaction, const Double3 &reverse_incident_dir, 
  const PathContext &context) const
{
  ScatterSample s = interaction.medium().SamplePhaseFunction(reverse_incident_dir, interaction.pos, sampler, context);
  s.value *= interaction.sigma_s;
  return s;
}


bool CameraRenderWorker::MaybeScatter(const SomeInteraction &interaction, PathState &ps, PathCoefficients &coeffs) const
{
  auto smpl = std::visit([this, &ps](auto &&ia) {
    return SampleScatterer(ia, -ps.ray.dir, ps.context);
  }, interaction);

  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{ 1. }, ps.current_node_count, sampler))
  {
    std::visit([&ps, &coeffs, &scene{ master->scene }, &smpl](auto &&ia) {
      PrepareStateForAfterScattering(ps, coeffs, ia, scene, smpl);
    }, interaction);
    return true;
  }
  else
    return false;
}




void CameraRenderWorker::MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps, PathCoefficients &coeffs) const
{
  const auto emitter = GetMaterialOf(interaction, master->scene).emitter;
  if (!emitter)
    return;

  Spectral3 radiance = emitter->Evaluate(interaction.hitid, -ps.ray.dir, ps.context, nullptr);

  double mis_weight = 1.0;
  if (ps.last_scatter_pdf_value && enable_nee) // Should be set if this is secondary ray.
  {
    const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, interaction.hitid));
    const double area_pdf = emitter->EvaluatePdf(interaction.hitid, ps.context);
    const double pdf_cvt = PdfConversion::AreaToSolidAngle(Length(ps.ray.org - interaction.pos), ps.ray.dir, interaction.normal);
    mis_weight = MisWeight(*ps.last_scatter_pdf_value, prob_select*area_pdf*pdf_cvt);
  }

  //coeffs[coeffs.size()-2].this_scatter_path *= mis_weight;
  coeffs.back().emission = radiance * mis_weight;

#ifdef DEBUG_BUFFERS 
  AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*weight_accum);
#endif
  //RecordMeasurementToCurrentPixel(mis_weight*radiance*ps.weight, ps);
}


void CameraRenderWorker::AddEnvEmission(const PathState &ps, PathCoefficients &coeffs) const
{
  if (!master->scene.HasEnvLight())
    return;

  const auto &emitter = master->scene.GetTotalEnvLight();
  const auto radiance = emitter.Evaluate(-ps.ray.dir, ps.context);

  double mis_weight = 1.0;
  if (ps.last_scatter_pdf_value && enable_nee) // Should be set if this is secondary ray.
  {
    const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, emitter));
    const double pdf_env = emitter.EvaluatePdf(-ps.ray.dir, ps.context);
    mis_weight = MisWeight(*ps.last_scatter_pdf_value, pdf_env*prob_select);
  }

  coeffs[coeffs.size()-2].this_scatter_path *= mis_weight;
  coeffs.back().emission = radiance;

  //RecordMeasurementToCurrentPixel(mis_weight*radiance*ps.weight, ps);
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

namespace {

inline Spectral3 AccumulateIndirectRadiance(
  PathCoefficients::const_iterator lit_vertex, 
  PathCoefficients::const_iterator end)
{
    using I = PathCoefficients::const_iterator;

    Spectral3 accum_incident{Eigen::zero};
    Spectral3 path_weight{Eigen::ones};

    int i = 0;
    // Search forward for light sources
    for (I node = lit_vertex+1; node != end; ++node, ++i)
    {
      path_weight *= node->segment_transmission_from_prev;
      const Spectral3 clamped_weight = ClampPathWeight(path_weight);

      accum_incident += clamped_weight*node->nee_emission_times_transmission*node->this_scatter_nee;
      
      if (i > 0)
      {
        accum_incident += clamped_weight*node->emission;
      }
      path_weight *= node->this_scatter_path;
    }
    assert(accum_incident.isFinite().all());

    return accum_incident;
}

}


void CameraRenderWorker::ProcessGuidingData(const PathCoefficients &coeffs, const PathState &ps)
{
  if (coeffs.size() < 3)
    return;

  {
    const Spectral3 accum_incident = AccumulateIndirectRadiance(
          coeffs.begin() + 1,
          coeffs.end());
    
    const Spectral3 radiance_in = coeffs[0].this_scatter_path*coeffs[1].segment_transmission_from_prev*
      (coeffs[1].this_scatter_path*accum_incident + coeffs[1].nee_emission_times_transmission*coeffs[1].this_scatter_nee);

    RecordMeasurementToCurrentPixel(radiance_in, ps);
  }

  for (int i=1; i<isize(coeffs)-1; ++i)
  {
    auto &lit = coeffs[i];
    if (lit.specular)
      continue;

    // Don't consider light that comes from below the surface.
    // If volume, this will be zero, and thus, execution will skip over the continue.
    if (lit.normal.dot(lit.dir) < 0.)
      continue;

    const auto accum_incident = AccumulateIndirectRadiance(
      coeffs.begin() + i,
      coeffs.end()
    );

    assert((float)lit.scatter_pdf > 0.f);

    if (/*accum_incident.nonZeros() &*/ accum_incident.isFinite().all())
    {
      const Spectral3 sample = accum_incident / lit.scatter_pdf;
      if (lit.surface)
      {
        radiance_recorder_surface->AddSample(
          this->radrec_local_surface,
          lit.pos,
          sampler,
          lit.dir,
          sample);
      }
      else
      {
        radiance_recorder_volume->AddSample(
          this->radrec_local_volume,
          lit.pos,
          sampler,
          lit.dir,
          sample);
      }
    }
    
    if (!lit.surface && i<isize(coeffs)-2) // Add direct light
    {
      const Spectral3 sample = 2.*coeffs[i+1].segment_transmission_from_prev*coeffs[i+1].emission / lit.scatter_pdf;
      radiance_recorder_volume->AddSample(
        this->radrec_local_volume,
        lit.pos,
        sampler,
        lit.dir,
        sample);
    }
    if (!lit.surface)
    {
      const Spectral3 sample = 2.*lit.nee_emission_times_transmission / lit.nee_pdf_conversion_factor;
      radiance_recorder_volume->AddSample(
        this->radrec_local_volume,
        lit.pos,
        sampler,
        lit.dir_nee,
        sample);
    }
  }

#if 0
  for (int i=isize(coeffs)-1; i>=1; --i)
  {
    // Propagate emission "backward".
    // Use += because of existing nee contribution.
    auto &prev = coeffs[i-1];

    prev.outscattered_radiance_direct +=
      prev.this_scatter_path * coeffs[i].segment_transmission_prev * coeffs[i].emission;
    
    // Into the previous vertex!
    const Spectral3 incident_indirect = coeffs[i].segment_transmission_prev * 
      (coeffs[i].outscattered_radiance_indirect + coeffs[i].outscattered_radiance_direct);

    // Direct lighting on previous vertex!
    const Spectral3 incident_direct = coeffs[i].segment_transmission_prev * coeffs[i].emission;

    prev.outscattered_radiance_indirect = 
      prev.this_scatter_path * incident_indirect;

    if (prev.on_non_specular_surface)
    {
      assert(prev.scatter_pdf > 0);
      Spectral3 sample_weight = [&]() -> Spectral3 {
        if (prev.surface->shading_normal.dot(coeffs[i].ray_to_this.dir) > 0.)
          if (enable_nee)
            return incident_indirect / prev.scatter_pdf;
          else
            return (incident_indirect + incident_direct) / prev.scatter_pdf;
        else
          return Spectral3::Zero();
      }();
      assert (prev.surface);
      if (sample_weight.nonZeros())
      {
        radiance_recorder_surface->AddSample(
          this->radrec_local_surface,
          *prev.surface,
          sampler,
          coeffs[i].ray_to_this.dir,
          sample_weight);
      }
    }
  }
#endif
}



} // namespace pathtracing_guided



std::unique_ptr<RenderingAlgo> AllocatePathtracingGuidedRenderingAlgo(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<pathtracing_guided::PathTracingAlgo2>(scene, params);
}
