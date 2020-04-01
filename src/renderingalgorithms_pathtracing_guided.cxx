#include <functional>
#include <type_traits>

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
#include "path_guiding.hxx"


namespace {
Spectral3 ClampPathWeight(const Spectral3 &w) {
  return w.min(10.);
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
  Double3 normal;
  double scatter_pdf = 0.;
  // T(y, x)/p_T(y,x) where y is the previously visited vertex
  Spectral3 segment_transmission_from_prev = Spectral3::Zero();
  // rho(x) / p_rho(x). BSDF + Dfactors evaluated for the traced path (not NEE sampling)
  Spectral3 this_scatter_path = Spectral3::Zero();
  // Emission Le(x)
  Spectral3 emission = Spectral3::Zero();
  // NEE quantities pertaining to the path segment to the sampled light and the light itself.
  Spectral3 nee_emission = Spectral3::Zero();
  Spectral3 nee_scatter_times_transmission_weight = Spectral3::Zero();
  bool specular = false;
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
  guiding::SurfacePathGuiding* radiance_recorder;
  mutable LightPickerUcbBufferedQueue::ThreadLocal picker_local;
  guiding::SurfacePathGuiding::ThreadLocal guiding_local;
  mutable Sampler sampler;
  mutable Span<RGB> framebuffer;
  RayTermination ray_termination;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  static constexpr int num_lambda_sweeps = decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED;
  const int worker_index;
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

  void ProcessGuidingData(const PathCoefficients &coeffs);

public:
  CameraRenderWorker(PathTracingAlgo2 *master, int worker_index);
  void Render(const ImageTileSet::Tile & tile);
  void PrepassRender(long sample_count);
  guiding::SurfacePathGuiding::ThreadLocal* GetGuidingLocalData() { return &guiding_local;  }
};



class PathTracingAlgo2 : public RenderingAlgo
{
  friend class CameraRenderWorker;
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
  std::unique_ptr<guiding::SurfacePathGuiding> radiance_recorder;
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
  render_params{ render_params_ }, scene{ scene_ }
{
  the_task_arena.initialize(std::max(1, this->render_params.num_threads));
  num_pixels = render_params.width * render_params.height;

  framebuffer.resize(num_pixels, RGB{});
  samplesPerTile.resize(tileset.size(), 0);

  pickers = std::make_unique<LightPickerUcbBufferedQueue>(scene, NumThreads());

  // TODO: adjustible voxel size
  radiance_recorder = std::make_unique<guiding::SurfacePathGuiding>(scene.GetBoundingBox(), 0.1, render_params_, the_task_arena);

  for (int i = 0; i < the_task_arena.max_concurrency(); ++i)
  {
    camerarender_workers.emplace_back(this, i);
  }
}


template<class Container, class Iter, class Func>
inline Container TransformToContainer(Iter begin_, Iter end_, Func &&f)
{
  Container c;
  for (; begin_ != end_; ++begin_)
  {
    c.push_back(f(*begin_));
  }
  return c;
}

template<class Iter, class Func>
inline auto TransformToVector(Iter begin_, Iter end_, Func &&f)
{
  return TransformToContainer<ToyVector<decltype(f(*begin_))>>(
    begin_,end_, std::forward<Func>(f));
}


inline void PathTracingAlgo2::Run()
{
  //#ifdef WRITE_DEBUG_OUT
  scene.WriteObj(guiding::GetDebugFilePrefix() / boost::filesystem::path("scene.obj"));
  //#endif

  Sampler sampler;
  auto guiding_thread_locals = TransformToVector(
    camerarender_workers.begin(), 
    camerarender_workers.end(), 
    [&](auto &w) { return w.GetGuidingLocalData(); });

  {
    const long num_samples_stop = this->num_pixels;
    long num_samples = 32 * 32;
    while (!stop_flag.load() && (num_samples < num_samples_stop))
    {
      std::cout << strconcat("Prepass: ", num_samples, " / ", num_samples_stop, "\n");
      radiance_recorder->BeginRound(AsSpan(guiding_thread_locals));

      the_task_arena.execute([this, num_samples] {
        tbb::parallel_for(tbb::blocked_range<long>(0l, num_samples, 32), [this](const tbb::blocked_range<long> &r)
        {
          const int worker_num = tbb::this_task_arena::current_thread_index();
          camerarender_workers[worker_num].PrepassRender(r.end()-r.begin());
        });
      });

      radiance_recorder->FinalizeRound(AsSpan(guiding_thread_locals));
      radiance_recorder->WriteDebugData();
      radiance_recorder->PrepareAdaptedStructures();

      num_samples *= 2;
    }
  }

  while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
  {
    // Clear the frame buffer to get rid of samples from previous iterations
    // which are assumed to be worse than current samples.
    std::fill(framebuffer.begin(), framebuffer.end(), RGB::Zero());
    std::fill(samplesPerTile.begin(), samplesPerTile.end(), 0ul);

    radiance_recorder->BeginRound(AsSpan(guiding_thread_locals));

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
    
    radiance_recorder->FinalizeRound(AsSpan(guiding_thread_locals));
    radiance_recorder->WriteDebugData();
    radiance_recorder->PrepareAdaptedStructures();

    std::cout << "Sweep " << spp_schedule.GetTotal() << " finished" << std::endl;
    CallInterruptCb(true);

    if (spp_schedule.GetPerIteration() == spp_schedule.GetMaxSppPerIteration())
    {
      spp_schedule.UpdateForNextPass();
      break;
    }
    else
      spp_schedule.UpdateForNextPass();
  } // Pass iteration

  this->record_samples_for_guiding = false;

  while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
  {
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
  radiance_recorder{master->radiance_recorder.get()},
  ray_termination{ master->render_params },
  worker_index{ worker_index },
  enable_nee{ true }
{
  framebuffer = AsSpan(master->framebuffer);
  if (master->render_params.pt_sample_mode == "bsdf")
  {
    enable_nee = false;
  }
}


void CameraRenderWorker::Render(const ImageTileSet::Tile &tile)
{
  const Int2 end = tile.corner + tile.shape;
  const int image_width = master->render_params.width;
  const int samples_per_pixel = master->GetSamplesPerPixel();
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
          ProcessGuidingData(path_coeffs);
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
      ProcessGuidingData(path_coeffs);
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

  coeffs.clear();

  const auto& camera = master->scene.GetCamera();
  p.pixel_index = camera.PixelToUnit({ pixel[0], pixel[1] });
  auto pos = camera.TakePositionSample(p.pixel_index, sampler, p.context);
  p.weight *= pos.value / pos.pdf_or_pmf;
  auto dir = camera.TakeDirectionSampleFrom(p.pixel_index, pos.coordinates, sampler, p.context);
  p.weight *= dir.value / dir.pdf_or_pmf;
  p.ray = { pos.coordinates, dir.coordinates };

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
  Spectral3 light_radiance{};
  Spectral3 path_weight = Spectral3::Ones();
  LightRef light_ref;

  pickers->GetDistributionNee().Sample(sampler, [&](auto &&light, double prob, const LightRef &light_ref_)
  {
    std::tie(segment_to_light, pdf, light_radiance) = light.SampleConnection(interaction, master->scene, sampler, ps.context);
    path_weight /= ((double)(pdf)*prob);
    if constexpr (!std::remove_reference<decltype(light)>::type::IsAngularDistribution())
    {
      path_weight *= 1./Sqr(segment_to_light.length);
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

  Spectral3 measurement_estimator = mis_weight*ps.weight*path_weight*light_radiance;

  {
    auto& coeff = coeffs.back();
    coeff.nee_scatter_times_transmission_weight = mis_weight*path_weight;
    coeff.nee_emission = light_radiance;
  }

  pickers->ObserveReturnNee(this->picker_local, light_ref, measurement_estimator);
  RecordMeasurementToCurrentPixel(measurement_estimator, ps);
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
  const auto& mixture = this->radiance_recorder->FindRadianceEstimate(interaction.pos).radiance_distribution;
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

  coeffs.back().emission = mis_weight*radiance;

#ifdef DEBUG_BUFFERS 
  AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*weight_accum);
#endif
  RecordMeasurementToCurrentPixel(mis_weight*radiance*ps.weight, ps);
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

  coeffs.back().emission = mis_weight*radiance;

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

namespace {

inline Spectral3 AccumulateIncidentRadiance(
  PathCoefficients::const_iterator lit_vertex, 
  PathCoefficients::const_iterator end,
  const bool estimate_only_indirect_lighting)
{
    using I = PathCoefficients::const_iterator;

    Spectral3 accum_incident{Eigen::zero};
    //lit_vertex->this_scatter_path;  // Previously. But, I believe this should be a 1!
    Spectral3 path_weight{Eigen::ones};

    bool first = true;
    // Search forward for light sources
    for (I node = lit_vertex+1; node != end; ++node)
    {
      path_weight *= node->segment_transmission_from_prev;
      const Spectral3 clamped_weight = ClampPathWeight(path_weight);
      const Spectral3 nee = clamped_weight*node->nee_scatter_times_transmission_weight*node->nee_emission;
      accum_incident += nee;
      if (!estimate_only_indirect_lighting || !first)
      {
        const Spectral3 incident = clamped_weight*node->emission;
        accum_incident += incident;
      }
      path_weight *= node->this_scatter_path;
      first = false;
    }
    assert(accum_incident.isFinite().all());

    return accum_incident;
}

}


void CameraRenderWorker::ProcessGuidingData(const PathCoefficients &coeffs)
{
  if (coeffs.size() < 2)
    return;

  for (int i=0; i<isize(coeffs)-1; ++i)
  {
    auto &lit = coeffs[i];
    if (lit.specular)
      continue;

    // Don't consider light that comes from below the surface.
    // If volume, this will be zero, and thus, execution will skip over the continue.
    if (lit.normal.dot(lit.dir) < 0.)
      continue;

    const auto accum_incident = AccumulateIncidentRadiance(
      coeffs.begin() + i,
      coeffs.end(),
      /*estimate_only_indirect_lighting=*/lit.surface
    );

    assert((float)lit.scatter_pdf > 0.f);
    const Spectral3 sample = accum_incident / lit.scatter_pdf;

    if (accum_incident.nonZeros() & accum_incident.isFinite().all())
    {
      radiance_recorder->AddSample(
        this->guiding_local,
        lit.pos,
        sampler,
        lit.dir,
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
        radiance_recorder->AddSample(
          this->guiding_local,
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
