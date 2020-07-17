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
#include "hashgrid.hxx"
#include "rendering_util.hxx"
#include "pathlogger.hxx"
#include "photonintersector.hxx"

#include "renderingalgorithms_interface.hxx"
#include "renderingalgorithms_simplebase.hxx"
#include "lightpicker_trivial.hxx"
#include "lightpicker_ucb.hxx"




namespace Photonmapping
{
using SimplePixelByPixelRenderingDetails::SamplesPerPixelSchedule;
using SimplePixelByPixelRenderingDetails::SamplesPerPixelScheduleConstant;
class PhotonmappingRenderingAlgo;
struct EmitterSampleVisitor;
using Lights::LightRef;
using Lightpickers::UcbLightPicker;
using Lightpickers::PhotonUcbLightPicker;
using Lightpickers::RadianceToObservationValue;

class Kernel2d
{
  double radius_inv_2;

public:  
  Kernel2d(double radius = 1.0) : radius_inv_2{1.0/Sqr(radius)} 
  {}
  
  // Returns negative number if input distance is larger than kernel radius.
  double operator()(double rr) const
  {
    const double tmp = rr*radius_inv_2;
    if (tmp >= 1.)
      return 0.;
    return radius_inv_2*CanonicalKernel2d(tmp);
  }
  
private:
  // The Kernel is normalized in the sense that int_S2 K(r) r dr dtheta = 1
  // Note that this function takes r squared! For efficiency.
  // It is Silverman’s two-dimensional biweight kernel as used by Jarosz et al. (2008)
  static inline double CanonicalKernel2d(double rr)
  {
    return 3./Pi*Sqr(1.-rr);
  }
};


// Constant Kernel.
class Kernel3d
{
  double radius_inv_2;
  double normalization;

public:
  Kernel3d(double radius = 1.0) : 
    radius_inv_2{1.0/Sqr(radius)}, normalization{1.0/(Cubed(radius)*UnitSphereVolume)}
    {}
  
  // Returns negative number if input distance is larger than kernel radius.
  double operator()(double rr) const
  {
    const double indicator = Heaviside(1.0 - rr*radius_inv_2);
    return normalization*indicator;
  }
};


template<class K>
inline double EvalKernel(const K &k, const Double3 &a, const Double3 &b)
{
  return k(LengthSqr(a - b));
}



//#define DEBUG_BUFFERS
#define LOGGING
//#define DEBUG_PATH_THROUGHPUT

class LightPickersUcbCombined
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

  struct alignas(128) ThreadLocal
  {
    ToyVector<std::pair<Lights::LightRef, float>> nee_returns;
    ToyVector<std::pair<int, float>> photon_returns;
  };

  static constexpr size_t BUFFER_SIZE = 10240;
  tbb::flow::graph graph;
  tbb::flow::function_node<Buffer<std::pair<LightRef, float>>> node_nee;
  tbb::flow::function_node<Buffer<std::pair<int, float>>> node_photon;

  ToyVector<ThreadLocal, tbb::cache_aligned_allocator<ThreadLocal>> local;
  UcbLightPicker picker_nee;
  PhotonUcbLightPicker picker_photon;

public:
  LightPickersUcbCombined(const Scene &scene, int num_workers)
    :
    graph(),
    node_nee(this->graph, 1, [&](Buffer<std::pair<LightRef, float>> x) {  picker_nee.ObserveReturns(x); delete x.begin();  }),
    node_photon(this->graph, 1, [&](Buffer<std::pair<int, float>> x) {  picker_photon.ObserveReturns(x); delete x.begin(); }),
    local(num_workers),
    picker_nee(scene), picker_photon(scene)
  {
   
  }
  void OnPassStart(const Span<LightRef> emitters_of_paths)
  {
    picker_photon.OnPassStart(emitters_of_paths);
  }

  void ObserveReturnNee(int worker, LightRef lr, const Spectral3 &value)
  {
    auto &l = local[worker];
    l.nee_returns.push_back(std::make_pair(lr, Lightpickers::RadianceToObservationValue(value)));
    if (l.nee_returns.size() >= BUFFER_SIZE)
    {
      node_nee.try_put(CopyToSpan(l.nee_returns));
      l.nee_returns.clear();
    }
  }

  void ObserveReturnPhoton(int worker, int path_index, const Spectral3 &value)
  {
    auto &l = local[worker];
    l.photon_returns.push_back(std::make_pair(path_index, Lightpickers::RadianceToObservationValue(value)));
    if (l.photon_returns.size() >= BUFFER_SIZE)
    {
      node_photon.try_put(CopyToSpan(l.photon_returns));
      l.photon_returns.clear();
    }
  }

  void OnPassEnd(const Span<LightRef> emitters_of_paths)
  {
    picker_photon.OnPassEnd(emitters_of_paths);
  }

  void ComputeDistribution()
  {
    graph.wait_for_all();

    picker_nee.ComputeDistribution();
    picker_photon.ComputeDistribution();
    std::cout << "NEE LP: ";
    picker_nee.Distribution().Print(std::cout);
    std::cout << "Photon LP: ";
    picker_photon.Distribution().Print(std::cout);
  }

  const Lightpickers::LightSelectionProbabilityMap& GetDistributionNee() const { return picker_nee.Distribution(); }
  const Lightpickers::LightSelectionProbabilityMap& GetDistributionPhotons() const { return picker_photon.Distribution(); }
};


struct Photon
{
  Double3 position;
  Spectral3f weight;
  Float3 direction;
#ifdef LOGGING
  IncompletePaths::SubPathHandle path_handle;
#endif
  int path_index;
  short node_number; // Starts at no 2. First node is on the light source.
  bool monochromatic;
};


class PhotonmappingWorker
{
  friend struct EmitterSampleVisitor;
  PhotonmappingRenderingAlgo *master;
  LightPickersUcbCombined* pickers;
  MediumTracker medium_tracker;
  Sampler sampler;
  int current_node_count = 0;
  int current_photon_index = 0;
  Spectral3 current_emission;
  LambdaSelection lambda_selection;
  PathContext context;
  RayTermination ray_termination;
  bool monochromatic = false;
private:
  Ray GeneratePrimaryRay();
  bool TrackToNextInteractionAndScatter(Ray &ray, Spectral3 &weight_accum);
  bool ScatterAt(Ray &ray, const SurfaceInteraction &interaction, const Spectral3 &track_weigh, Spectral3 &weight_accum);
  bool ScatterAt(Ray &ray, const VolumeInteraction &interaction, const Spectral3 &track_weigh, Spectral3 &weight_accum);
public:  
  // Public access because this class is hardly a chance to missuse.
  // Simply run TracePhotons as much as desired, then get the photons
  // from these members. Then destroy the instance.
  ToyVector<Photon> photons_volume;
  ToyVector<Photon> photons_surface;
#ifdef DEBUG_PATH_THROUGHPUT
  SpectralN max_throughput_weight{0};
  double max_bsdf_correction_weight{0};
  SpectralN max_uncorrected_bsdf_weight{0};
#endif
#ifdef LOGGING
  IncompletePaths logger;
#endif
  PhotonmappingWorker(PhotonmappingRenderingAlgo *master, int worker_index);
  void StartNewPass(const LambdaSelection &lambda_selection);
  void TracePhotons(int start_index, int count);
};


struct PathState
{
  PathState(const Scene &scene, const LambdaSelection &lambda_selection)
    : medium_tracker{ scene },
    context{lambda_selection}
  {}

  MediumTracker medium_tracker;
  PathContext context;
  Ray ray;
  Spectral3 weight;
  boost::optional<Pdf> last_scatter_pdf_value; // For MIS.
  int current_node_count;
  bool monochromatic;
};


class CameraRenderWorker
{
  const PhotonmappingRenderingAlgo * const master;
  LightPickersUcbCombined* const pickers;
  mutable Sampler sampler;
  mutable Span<RGB> framebuffer;
  LambdaSelection lambda_selection;
  RayTermination ray_termination; 
  Kernel2d kernel2d;
  Kernel3d kernel3d;
  const int worker_index;
#ifdef LOGGING
  mutable Pathlogger logger;
#endif
private:
  void InitializePathState(PathState &p, Int2 pixel) const;
  bool TrackToNextInteractionAndRecordPixel(PathState &ps) const;
  void AddPhotonContributions(const SurfaceInteraction &interaction, const PathState &ps) const;
  //void AddPhotonContributions(const VolumeInteraction& interaction, const Double3& incident_dir, const Spectral3& path_weight);
  bool MaybeScatterAtSpecularLayer(const SurfaceInteraction &interaction, PathState &ps) const;
  void RecordMeasurementToCurrentPixel(const Spectral3 &measurement, const PathState &ps) const;
  void MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps) const;
  //void MaybeAddEmission(const VolumeInteraction &interaction, PathState &ps) const;
  void AddEnvEmission(const PathState &ps) const;
  void AddPhotonBeamContributions(const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &track_weight, const PathState &ps) const;
  void MaybeAddDirectLighting(const SurfaceInteraction &interaction, const PathState &ps) const;
#ifdef DEBUG_BUFFERS
  void AddToDebugBuffer(int id, int level, const Spectral3 &measurement_estimate);
#endif
public:
  CameraRenderWorker(PhotonmappingRenderingAlgo *master, int worker_index);
  void StartNewPass(const LambdaSelection &lambda_selection);
  void Render(const ImageTileSet::Tile & tile);

};


class PhotonmappingRenderingAlgo : public RenderingAlgo
{
  friend class PhotonmappingWorker;
  friend class CameraRenderWorker;
private:
  ToyVector<RGB> framebuffer;
  ImageTileSet tileset;
  ToyVector<std::uint64_t> samplesPerTile; // False sharing!

  std::unique_ptr<HashGrid> hashgrid_volume; // Photon Lookup
  std::unique_ptr<HashGrid> hashgrid_surface;
  std::unique_ptr<PhotonIntersector> beampointaccel;
  ToyVector<Photon> photons_volume;
  ToyVector<Photon> photons_surface;
  
  int pass_index = 0;
  int num_photons_traced = 0;
  int num_pixels = 0;
  double current_surface_photon_radius = 0;
  double current_volume_photon_radius = 0;
  double radius_reduction_alpha = 2./3.; // Less means faster reduction. 2/3 is suggested in the unified path sampling paper.
  SamplesPerPixelScheduleConstant spp_schedule;
  tbb::task_group the_task_group;
  tbb::task_arena the_task_arena;
  tbb::atomic<bool> stop_flag = false;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  ToyVector<PhotonmappingWorker> photonmap_workers;
  ToyVector<CameraRenderWorker> camerarender_workers;
  ToyVector<Lights::LightRef> emitter_refs;
  std::unique_ptr<LightPickersUcbCombined> pickers;
#ifdef DEBUG_BUFFERS  
  ToyVector<Spectral3ImageBuffer> debugbuffers;
  static constexpr int DEBUGBUFFER_DEPTH = 10;
  static constexpr int DEBUGBUFFER_ID_PH = 0;
  static constexpr int DEBUGBUFFER_ID_BSDF = 10;
  static constexpr int DEBUGBUFFER_ID_DIRECT = 11;
#endif
#ifdef DEBUG_PATH_THROUGHPUT
  SpectralN max_throughput_weight{0.};
  double max_bsdf_correction_weight{0};
  SpectralN max_uncorrected_bsdf_weight{0};
#endif
public:
  const RenderingParameters &render_params;
  const Scene &scene;

public:
  PhotonmappingRenderingAlgo(const Scene &scene_, const RenderingParameters &render_params_);

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

private:
  void PrepareGlobalPhotonMap();
  void UpdatePhotonRadii();
};



Photonmapping::PhotonmappingRenderingAlgo::PhotonmappingRenderingAlgo(
  const Scene &scene_, const RenderingParameters &render_params_)
  : 
  RenderingAlgo{},
    tileset({ render_params_.width, render_params_.height }),
    spp_schedule{ render_params_ }, 
    render_params{ render_params_ }, scene{ scene_ }
{
  the_task_arena.initialize(std::max(1, this->render_params.num_threads));
  num_pixels = render_params.width * render_params.height;
  current_surface_photon_radius = render_params.initial_photon_radius;
  current_volume_photon_radius = render_params.initial_photon_radius;
#ifdef DEBUG_BUFFERS
  for (int i = 0; i < DEBUGBUFFER_DEPTH + 2; ++i)
  {
    debugbuffers.emplace_back(render_params.width, render_params.height);
  }
#endif

  framebuffer.resize(num_pixels, RGB{});
  samplesPerTile.resize(tileset.size(), 0);

  pickers = std::make_unique<LightPickersUcbCombined>(scene, NumThreads());

  for (int i = 0; i < the_task_arena.max_concurrency(); ++i)
  {
    photonmap_workers.emplace_back(this, i);
    camerarender_workers.emplace_back(this, i);
  }
}





inline void PhotonmappingRenderingAlgo::Run()
{
  Sampler sampler;

  while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
  {
    num_photons_traced = num_pixels * GetSamplesPerPixel();
    emitter_refs.resize(num_photons_traced);

    pickers->ComputeDistribution();

    // Full sweep across spectrum counts as one pass.
    // But one sweep will require multiple photon mapping passes since 
    // each pass only covers part of the spectrum.
    for (int spectrum_sweep_idx = 0;
      (spectrum_sweep_idx <= decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED) && !stop_flag.load();
      ++spectrum_sweep_idx)
    {
      //shared_pixel_index = 0;
      auto lambda_selection = lambda_selection_factory.WithWeights(sampler);

      tbb::parallel_for(0, (int)photonmap_workers.size(), [&](int i) {
        photonmap_workers[i].StartNewPass(lambda_selection);
        camerarender_workers[i].StartNewPass(lambda_selection);
      });

      the_task_arena.execute([this] {
        const int photonsPerBatch = ImageTileSet::basicPixelsPerTile()*GetSamplesPerPixel();

          parallel_for_interruptible(0, num_photons_traced, photonsPerBatch, [this, photonsPerBatch](int i)
          {
              const int worker_num = tbb::this_task_arena::current_thread_index();
              photonmap_workers[worker_num].TracePhotons(i, photonsPerBatch);
          },
              /*irq_handler=*/[this]() -> bool
          {
              return !(this->stop_flag.load());
          }, the_task_group);
      });

      PrepareGlobalPhotonMap();
      pickers->OnPassStart(AsSpan(emitter_refs));

#ifdef DEBUG_BUFFERS
      for (auto &b : debugbuffers) b.AddSampleCount(GetSamplesPerPixel());
#endif

      the_task_arena.execute([this] {
          parallel_for_interruptible(0, this->tileset.size(), 1, [this](int i)
          {
              const int worker_num = tbb::this_task_arena::current_thread_index();
              camerarender_workers[worker_num].Render(this->tileset[i]);
              this->samplesPerTile[i]++;
          },
              /*irq_handler=*/[this]() -> bool
          {
              if (this->stop_flag.load())
                  return false;
              this->CallInterruptCb(false);
              return true;
          }, the_task_group);
      });

      pickers->OnPassEnd(AsSpan(emitter_refs));

#ifdef LOGGING
      IncompletePaths::Clear();
#endif

    } // For spectrum sweep
    std::cout << "Sweep " << spp_schedule.GetTotal() << " finished" << std::endl;
    CallInterruptCb(true);
    spp_schedule.UpdateForNextPass();
    ++pass_index;
    UpdatePhotonRadii();
  } // Pass iteration
}


inline std::unique_ptr<Image> PhotonmappingRenderingAlgo::GenerateImage()
{
  auto bm = std::make_unique<Image>(render_params.width, render_params.height);
  tbb::parallel_for(0, tileset.size(), [this, bm{bm.get()}](int i) {
    const auto tile = tileset[i];
    if (this->samplesPerTile[i] > 0)
    {
      framebuffer::ToImage(*bm, tile, AsSpan(this->framebuffer), this->samplesPerTile[i]);
    }
  });
#ifdef DEBUG_BUFFERS
  int buf_num = 0;
  for (auto &b : debugbuffers)
  {
    Image im(render_params.width, render_params.height);
    tbb::parallel_for(0, render_params.height, [&](int row) {
      b.ToImage(im, row, row + 1, /*convert_linear_to_srgb=*/ false);
    });
    im.write(fmt::format("/tmp/debug{}.png", (buf_num++)));
  }
  {
    Image im(render_params.width, render_params.height);
    tbb::parallel_for(0, render_params.height, [&](int row) {
      buffer.ToImage(im, row, row + 1, /*convert_linear_to_srgb=*/ false);
    });
    im.write("/tmp/debug.png");
  }
#endif
  return bm;
}


void PhotonmappingRenderingAlgo::PrepareGlobalPhotonMap()
{
  photons_volume.clear(); 
  photons_surface.clear();
  
  for(auto &worker : photonmap_workers)
  {
    Append(photons_volume, worker.photons_volume);
    Append(photons_surface, worker.photons_surface);
#ifdef DEBUG_PATH_THROUGHPUT
    max_throughput_weight = max_throughput_weight.cwiseMax(worker.max_throughput_weight);
    max_bsdf_correction_weight = std::max(max_bsdf_correction_weight,worker.max_bsdf_correction_weight);
    max_uncorrected_bsdf_weight = max_uncorrected_bsdf_weight.cwiseMax(worker.max_uncorrected_bsdf_weight);
#endif
  }
#ifdef DEBUG_PATH_THROUGHPUT
  std::cout << "Max throughput = " << max_throughput_weight << std::endl;
  std::cout << "max_bsdf_correction_weight = " << max_bsdf_correction_weight << std::endl;
  std::cout << "max_uncorrected_bsdf_weight= " << max_uncorrected_bsdf_weight << std::endl;
#endif
  ToyVector<Double3> points; points.reserve(photons_surface.size());
  std::transform(photons_surface.begin(), photons_surface.end(), 
                 std::back_inserter(points), [](const Photon& p) { return p.position; });
  hashgrid_surface = std::make_unique<HashGrid>(current_surface_photon_radius, points);
  points.clear();
  std::transform(photons_volume.begin(), photons_volume.end(), 
                 std::back_inserter(points), [](const Photon& p) { return p.position; });  
  hashgrid_volume = std::make_unique<HashGrid>(current_volume_photon_radius, points);
  
  beampointaccel = std::make_unique<PhotonIntersector>(current_surface_photon_radius, points);
}


void PhotonmappingRenderingAlgo::UpdatePhotonRadii()
{
  // This is the formula (15) from Knaus and Zwicker (2011) "Progressive Photon Mapping: A Probabilistic Approach"
  double factor = (pass_index+1+radius_reduction_alpha)/(pass_index+2);
  current_surface_photon_radius *= std::sqrt(factor);
  // Equation (20) which is for volume photons
  current_volume_photon_radius *= std::pow(factor, 1./3.);
  std::cout << "Photon radii: " << current_surface_photon_radius << ", " << current_volume_photon_radius << std::endl;
}




struct EmitterSampleVisitor
{
  PhotonmappingWorker* this_;
  PhotonmappingRenderingAlgo *master;
  Ray out_ray;
  LightRef lightref;
  EmitterSampleVisitor(PhotonmappingWorker* this_) : this_{this_}, master{this_->master} {}
  
  template<
    class LightClass,
    std::enable_if_t<std::is_base_of_v<Lights::Base, LightClass>, int> = 0>
    void operator()(LightClass &&light, double prob, const LightRef &light_ref)
  {
    auto [ray, pdfs, radiance] = light.SampleExitantRay(master->scene, this_->sampler, this_->context);
    this_->current_emission = radiance / (Value(pdfs.first)*Value(pdfs.second)*prob);
    this->out_ray = ray;
    this->lightref = light_ref;
  }
};






PhotonmappingWorker::PhotonmappingWorker(PhotonmappingRenderingAlgo *master, int worker_index)
  : master{master},
    pickers{ master->pickers.get() },
    medium_tracker{master->scene},
    context{},
    ray_termination{master->render_params}
{
  photons_surface.reserve(1024*1024);
  photons_volume.reserve(1024*1024);
}

void PhotonmappingWorker::StartNewPass(const LambdaSelection &lambda_selection)
{
  this->lambda_selection = lambda_selection;
  context = PathContext{lambda_selection, TransportType::IMPORTANCE};
  photons_surface.clear();
  photons_volume.clear();
#ifdef DEBUG_PATH_THROUGHPUT
  max_throughput_weight.setConstant(0.);
#endif
}


Ray PhotonmappingWorker::GeneratePrimaryRay()
{
  EmitterSampleVisitor emission_visitor{this};
  pickers->GetDistributionPhotons().Sample(sampler, emission_visitor);
  master->emitter_refs[current_photon_index] = emission_visitor.lightref;
  return emission_visitor.out_ray;
}
 

void PhotonmappingWorker::TracePhotons(int start_index, int count)
{
  const int end_index = std::min(start_index + count, master->num_photons_traced);
  for (current_photon_index = start_index; 
       current_photon_index < end_index; 
       ++current_photon_index)
  {
    current_node_count = 2;
    monochromatic = false;
    // Weights due to wavelength selection is accounted for in view subpath weights.
    Ray ray = GeneratePrimaryRay();
    medium_tracker.initializePosition(ray.org);
#ifdef LOGGING
    {
      logger.NewPath();
      logger.PushNode();
      auto &l = logger.GetNode(-1);
      l.position = ray.org;
      l.exitant_dir = ray.dir;
      l.weight = this->current_emission;
    }
#endif    
    bool keepgoing = true;
    Spectral3 weight_accum{1.}; 
    do
    {
      keepgoing = TrackToNextInteractionAndScatter(ray, weight_accum);
    }
    while (keepgoing);
  }
}

inline void SpectralMaxInplace(SpectralN &combined, const LambdaSelection &l, const Spectral3 &val)
{
  for (int i=0; i<l.indices.size(); ++i)
  {
    combined[l.indices[i]] = std::max(combined[l.indices[i]], val[i]);
  }
}


bool PhotonmappingWorker::TrackToNextInteractionAndScatter(Ray &ray, Spectral3 &weight_accum)
{
  const bool keepgoing = TrackToNextInteraction(master->scene, ray, context, weight_accum*current_emission, sampler, medium_tracker, nullptr,
    /*surface=*/[&](const SurfaceInteraction &interaction, double distance, const Spectral3 &track_weight) -> bool
    {
#ifdef LOGGING
      logger.GetNode(-1).transmission_weight_to_next = track_weight;
#endif
      weight_accum *= track_weight;
      if (!GetShaderOf(interaction, master->scene).prefer_path_tracing_over_photonmap)
      {
#ifdef LOGGING
        logger.PushNode();
        auto& lg = logger.GetNode(-1);
        lg.position = interaction.pos;
        lg.geom_normal = interaction.geometry_normal;
        lg.is_surface = true;
#endif
        photons_surface.push_back({
          interaction.pos,
          (weight_accum*current_emission).cast<float>(),
          ray.dir.cast<float>(),
#ifdef LOGGING
          logger.GetHandle(),
#endif
          /*photon_path_index = */current_photon_index,
          (short)current_node_count,
          monochromatic
        });
      }
      current_node_count++;
#ifdef DEBUG_PATH_THROUGHPUT
      SpectralMaxInplace(max_throughput_weight, lambda_selection, weight_accum);
#endif
      return ScatterAt(ray, interaction, track_weight, weight_accum);
    },
    /*volume=*/[&](const VolumeInteraction &interaction, double distance, const Spectral3 &track_weight) -> bool
    {
#ifdef LOGGING
      logger.GetNode(-1).transmission_weight_to_next = track_weight;
      logger.PushNode();
      auto& lg = logger.GetNode(-1);
      lg.position = interaction.pos;
#endif
      weight_accum *= track_weight;
      photons_volume.push_back({
        interaction.pos,
        (weight_accum*current_emission).cast<float>(),
        ray.dir.cast<float>(),
#ifdef LOGGING
          logger.GetHandle(),
#endif
        /*photon_path_index = */current_photon_index,
        (short)current_node_count,
        monochromatic
      });
      current_node_count++;
#ifdef DEBUG_PATH_THROUGHPUT
      SpectralMaxInplace(max_throughput_weight, lambda_selection, weight_accum);
#endif
      return ScatterAt(ray, interaction, track_weight, weight_accum);
    },
    /*escape*/[&](const Spectral3 &weight) -> bool
    {
#ifdef LOGGING
      logger.GetNode(-1).transmission_weight_to_next = weight;
#endif
      return false;
    }
  );
  return keepgoing;
}


bool PhotonmappingWorker::ScatterAt(Ray& ray, const SurfaceInteraction& interaction, const Spectral3 &, Spectral3& weight_accum)
{
  const auto &shader = GetShaderOf(interaction,master->scene);
  auto smpl = shader.SampleBSDF(-ray.dir, interaction, sampler, context);
  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler))
  {
    monochromatic |= shader.require_monochromatic;
    smpl.value *= DFactorPBRT(interaction,smpl.coordinates) / smpl.pdf_or_pmf;
    auto corr = BsdfCorrectionFactorPBRT(-ray.dir, interaction, smpl.coordinates, 2.);
#ifdef DEBUG_PATH_THROUGHPUT
    SpectralMaxInplace(max_uncorrected_bsdf_weight, lambda_selection, smpl.value);
    max_bsdf_correction_weight = std::max(max_bsdf_correction_weight, corr);
#endif
    smpl.value *= corr;
    weight_accum *= smpl.value;
    ray.dir = smpl.coordinates;
    ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ray.dir);
    if (Dot(ray.dir, interaction.normal) < 0.)
    {
      // By definition, intersection.normal points to where the intersection ray is coming from.
      // Thus we can determine if the sampled direction goes through the surface by looking
      // if the direction goes in the opposite direction of the normal.
      medium_tracker.goingThroughSurface(ray.dir, interaction);
    }
#ifdef LOGGING
    auto &l = logger.GetNode(-1);
    l.exitant_dir = ray.dir;
    l.weight = smpl.value;
    l.is_specular = smpl.pdf_or_pmf.IsFromDelta();
#endif
    return true;
  }
  else 
    return false;
}


bool PhotonmappingWorker::ScatterAt(Ray& ray, const VolumeInteraction& interaction, const Spectral3 &track_weight, Spectral3& weight_accum)
{
  auto smpl = interaction.medium().SamplePhaseFunction(-ray.dir, interaction.pos, sampler, context);
  bool survive = ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler);
  const Spectral3 weight = interaction.sigma_s*smpl.value / smpl.pdf_or_pmf;
  weight_accum *= weight;
  ray.dir = smpl.coordinates;
  ray.org = interaction.pos;
#ifdef LOGGING
  auto &l = logger.GetNode(-1);
  l.exitant_dir = ray.dir;
  l.weight = smpl.value;
#endif
  return survive;
}



CameraRenderWorker::CameraRenderWorker(Photonmapping::PhotonmappingRenderingAlgo* master, int worker_index)
  : master{master},
    pickers{ master->pickers.get() },
    ray_termination{master->render_params},
    worker_index{ worker_index }
{
  framebuffer = AsSpan(master->framebuffer);
}

void CameraRenderWorker::StartNewPass(const LambdaSelection& lambda_selection)
{
  this->lambda_selection = lambda_selection;
  kernel2d = Kernel2d(master->current_surface_photon_radius);
  kernel3d = Kernel3d(master->current_volume_photon_radius);
}


void CameraRenderWorker::Render(const ImageTileSet::Tile &tile)
{
  const Int2 end = tile.corner + tile.shape;
  const int image_width = master->render_params.width;
  const int samples_per_pixel = master->GetSamplesPerPixel();
  PathState state{ master->scene, lambda_selection };

  for (int iy = tile.corner[1]; iy < end[1]; ++iy)
  {
    for (int ix = tile.corner[0]; ix < end[0]; ++ix)
    {
      for (int i = 0; i < samples_per_pixel; ++i)
      {
        InitializePathState(state, { ix, iy });
        bool keepgoing = true;
        do
        {
          keepgoing = TrackToNextInteractionAndRecordPixel(state);
        } while (keepgoing);
      }
    }
  }
}




void CameraRenderWorker::InitializePathState(PathState &p, Int2 pixel) const
{
    const auto& camera = master->scene.GetCamera();
    p.context.pixel_index = camera.PixelToUnit({ pixel[0], pixel[1] });

    p.current_node_count = 2; // First node on camera. We start with the node code of the next interaction.
    p.monochromatic = false;
    p.weight = lambda_selection.weights;
    p.last_scatter_pdf_value = boost::none;

    auto pos = camera.TakePositionSample(p.context.pixel_index, sampler, p.context);
    p.weight *= pos.value / pos.pdf_or_pmf;
    auto dir = camera.TakeDirectionSampleFrom(p.context.pixel_index, pos.coordinates, sampler, p.context);
    p.weight *= dir.value / dir.pdf_or_pmf;
    p.ray = { pos.coordinates, dir.coordinates };

    p.medium_tracker.initializePosition(pos.coordinates);

#ifdef LOGGING
    logger.NewPath();
    logger.PushNode();
    auto &ln = logger.GetNode(-1);
    ln.position = p.ray.org;
    ln.exitant_dir = p.ray.dir;
    ln.weight = p.weight;
    logger.GetContribution().wavelengths = p.context.lambda_idx;
#endif
}


bool CameraRenderWorker::TrackToNextInteractionAndRecordPixel(PathState &ps) const
{
  bool keepgoing = false;
  TrackBeam(master->scene, ps.ray, ps.context, sampler, ps.medium_tracker,
    /*surface_visitor=*/[&ps, this, &keepgoing](const SurfaceInteraction &interaction, const Spectral3 &track_weight)
    {
#ifdef LOGGING
      {
        logger.GetNode(-1).transmission_weight_to_next = track_weight;
        logger.PushNode();
        auto& ln = logger.GetNode(-1);
        ln.position = interaction.pos;
        ln.is_surface = true;
        ln.geom_normal = interaction.geometry_normal;
      }
#endif
      ps.weight *= track_weight;
      AddPhotonContributions(interaction, ps);
      MaybeAddEmission(interaction, ps);
      MaybeAddDirectLighting(interaction, ps);
      ps.current_node_count++;
      keepgoing = MaybeScatterAtSpecularLayer(interaction, ps);
    },
    /*segment visitor=*/[&ps,this](const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &track_weight)
    {
#ifdef LOGGING
      logger.GetNode(-1).transmission_weight_to_next = track_weight;
      // TODO: deal with these paths ...
#endif
      AddPhotonBeamContributions(segment, medium, pct, track_weight, ps);
    },
    /* escape_visitor=*/[&ps,this](const Spectral3 &track_weight)
    {
#ifdef LOGGING
      logger.GetNode(-1).transmission_weight_to_next = track_weight;
#endif
      ps.weight *= track_weight;
      AddEnvEmission(ps);
    }
  );
  return keepgoing;
}


static double MisWeight(Pdf pdf_or_pmf_taken, double pdf_other)
{
    double mis_weight = 1.;
    if (!pdf_or_pmf_taken.IsFromDelta())
    {
        mis_weight = PowerHeuristic(pdf_or_pmf_taken, {pdf_other});
    }
    return mis_weight;
}


void CameraRenderWorker::MaybeAddDirectLighting(const SurfaceInteraction &interaction, const PathState &ps) const
{
    // To consider for direct lighting NEE with MIS.
    // Segment, DFactors, AntiselfIntersectionOffset, Medium Passage, Transmittance Estimate, BSDF/Le factor, inv square factor.
    // Mis weight. P_light, P_bsdf
    
    const auto &shader = GetShaderOf(interaction,master->scene);
    if (!shader.prefer_path_tracing_over_photonmap)
        return;
    
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
    auto [ray, length] = segment_to_light;
    
    Spectral3 path_weight = ps.weight;
    path_weight *= DFactorPBRT(interaction, ray.dir);

    double bsdf_pdf = 0.;
    Spectral3 bsdf_weight = shader.EvaluateBSDF(-ps.ray.dir, interaction, ray.dir, ps.context, &bsdf_pdf);
    path_weight *= bsdf_weight;

    
    MediumTracker medium_tracker{ps.medium_tracker}; // Copy because well don't want to keep modifications.
    MaybeGoingThroughSurface(medium_tracker, ray.dir, interaction);
    Spectral3 transmittance = TransmittanceEstimate(master->scene, segment_to_light, medium_tracker, ps.context, sampler);
    path_weight *= transmittance;
    
    double mis_weight = MisWeight(pdf, bsdf_pdf);
    
#ifdef DEBUG_BUFFERS 
    AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_DIRECT, 0, path_weight*weight);
#endif
    Spectral3 measurement_estimator = mis_weight*path_weight*light_weight;

    pickers->ObserveReturnNee(this->worker_index, light_ref, measurement_estimator);
    RecordMeasurementToCurrentPixel(measurement_estimator, ps);

#ifdef LOGGING
    {
      auto &ln1 = logger.GetNode(-1);
      ln1.transmission_weight_to_next = transmittance;
      ln1.weight = bsdf_weight;
      ln1.exitant_dir = ray.dir;
    }
    {
      logger.PushNode();
      auto& ln2 = logger.GetNode(-1);
      ln2.weight = light_weight;
      logger.GetContribution().pixel_contribution = measurement_estimator;
      logger.WritePath();
      logger.PopNode();
    }
    { // reset
      auto &ln1 = logger.GetNode(-1);
      ln1.transmission_weight_to_next.setZero();
      ln1.weight.setZero();
      ln1.exitant_dir.setZero();
    }
#endif
}


bool CameraRenderWorker::MaybeScatterAtSpecularLayer(const SurfaceInteraction &interaction, PathState &ps) const
{
  const auto &shader = GetShaderOf(interaction,master->scene);
  auto smpl = shader.SampleBSDF(-ps.ray.dir, interaction, sampler, ps.context);
  if (!smpl.pdf_or_pmf.IsFromDelta() && !shader.prefer_path_tracing_over_photonmap)
    return false;
  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, ps.current_node_count, sampler))
  {
    ps.monochromatic |= shader.require_monochromatic;
    Spectral3 scatter_weight = smpl.value * DFactorPBRT(interaction, smpl.coordinates) / smpl.pdf_or_pmf;
    ps.weight *= scatter_weight;
    ps.ray.dir = smpl.coordinates;
    ps.ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ps.ray.dir);
    MaybeGoingThroughSurface(ps.medium_tracker, ps.ray.dir, interaction);
    ps.last_scatter_pdf_value = smpl.pdf_or_pmf;
#ifdef LOGGING
    {
      auto& ln = logger.GetNode(-1);
      ln.weight = scatter_weight;
      ln.exitant_dir = ps.ray.dir;
      ln.is_specular = smpl.pdf_or_pmf.IsFromDelta();
    }
#endif
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
        const double pdf_cvt = PdfConversion::AreaToSolidAngle(Length(ps.ray.org-interaction.pos), ps.ray.dir, interaction.normal);
        mis_weight = MisWeight(*ps.last_scatter_pdf_value, prob_select*area_pdf*pdf_cvt);
    }

#ifdef DEBUG_BUFFERS 
    AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*weight_accum);
#endif
    Spectral3 measurement_contribution = mis_weight * radiance*ps.weight;
    RecordMeasurementToCurrentPixel(measurement_contribution, ps);

#ifdef LOGGING
    {
      auto& ln = logger.GetNode(-1);
      ln.weight = radiance;
      logger.GetContribution().pixel_contribution = measurement_contribution;
      logger.WritePath();
      ln.weight.setZero();
    }
#endif
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
  
#ifdef DEBUG_BUFFERS 
    AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*path_weight);
#endif
    const Spectral3 measurement_contribution = mis_weight * radiance*ps.weight;
    RecordMeasurementToCurrentPixel(measurement_contribution, ps);

#ifdef LOGGING
    {
      logger.PushNode();
      auto& ln = logger.GetNode(-1);
      ln.position = ps.ray.org + 100. * ps.ray.dir;
      ln.weight = radiance;
      logger.GetContribution().pixel_contribution = measurement_contribution;
      logger.WritePath();
      logger.PopNode();
    }
#endif
}


//void CameraRenderWorker::MaybeAddEmission(const Ray& , const VolumeInteraction &interaction, Spectral3& weight_accum) const
//{
    // TODO: Implement MIS.
//   if (interaction.medium().is_emissive)
//   {
//     Spectral3 irradiance = interaction.medium().EvaluateEmission(interaction.pos, context, nullptr);
//     RecordMeasurementToCurrentPixel(irradiance*weight_accum/UnitSphereSurfaceArea);
//   }
//}


inline Spectral3 MaybeReWeightToMonochromatic(const Spectral3 &w, bool monochromatic)
{
  return monochromatic ? Spectral3{w[0]*3._sp,0,0} : w;
}


void CameraRenderWorker::AddPhotonContributions(const SurfaceInteraction& interaction, const PathState &ps) const
{
  const auto &shader = GetShaderOf(interaction,master->scene);
  if (shader.prefer_path_tracing_over_photonmap)
    return;
  auto path_factors = (ps.weight * (1.0 / master->num_photons_traced)).eval();
  Spectral3 reflect_estimator{0.};
  master->hashgrid_surface->Query(interaction.pos, [&](int photon_idx)
  {
    const auto &photon = master->photons_surface[photon_idx];
    // There is minus one because there are two coincident nodes, photon and camera node. But we only want to count one.
    if (photon.node_number + ps.current_node_count - 1 > ray_termination.max_node_count)
      return;
    const double kernel_val = EvalKernel(kernel2d, photon.position, interaction.pos);
    if (kernel_val <= 0.)
      return;
    Spectral3 bsdf_val = shader.EvaluateBSDF(-ps.ray.dir, interaction, -photon.direction.cast<double>(), ps.context, nullptr);
    {
      // This is Veach's shading correction for the normal (non-adjoint) BSDF, defined in Eq. 5.17, pg. 152.
      // The reason it is added here and why there is no bare cos(Ng,wi), is that when the photon is cast to this point, the integral transform
      // from area integration to solid angle "consumes" the cos(Ng,wi) factor, leaving only the following correction.
      double shading_correction = std::abs(Dot(interaction.shading_normal, photon.direction.cast<double>()))/std::abs(Dot(interaction.normal, photon.direction.cast<double>()));
             shading_correction = std::min(2., shading_correction);
      // Or, seen as only in the path integration framework, the cos factor cancels with the cos factor of the PDF. 
      // See also Eq. 19 in Jarosz (2008) "The Beam Radiance Estimate for Volumetric Photon Mapping"
      bsdf_val *= shading_correction*kernel_val;
    }
    Spectral3 weight = MaybeReWeightToMonochromatic(photon.weight.cast<double>()*bsdf_val, photon.monochromatic | ps.monochromatic);
    reflect_estimator += weight;
    pickers->ObserveReturnPhoton(worker_index, photon.path_index, weight*path_factors);
#ifdef DEBUG_BUFFERS 
    {
      Spectral3 w = path_weight*weight/master->num_photons_traced;
      AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_PH, photon.node_number, w);
      if (last_scatter_pdf_value)
        AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, w);
    }
#endif
  });
  Spectral3 weight = reflect_estimator*path_factors;
  RecordMeasurementToCurrentPixel(weight, ps);
}

#if 0
void CameraRenderWorker::AddPhotonContributions(const VolumeInteraction& interaction, const Double3 &incident_dir, const Spectral3& path_weight)
{
  const auto path_factors = (interaction.sigma_s*path_weight*(1.0 / master->num_photons_traced)).eval();
  Spectral3 inscatter_estimator{0.};
  master->hashgrid_volume->Query(interaction.pos, [&](int photon_idx){
    const auto &photon = master->photons_volume[photon_idx];
    // There is minus one because there are two coincident nodes, photon and camera node. But we only want to count one.
    if (photon.node_number + current_node_count - 1 > ray_termination.max_node_count)
      return;
    const double kernel_val = EvalKernel(kernel3d, photon.position, interaction.pos);
    if (kernel_val <= 0.)
      return;
    Spectral3 scatter_val = interaction.medium().EvaluatePhaseFunction(-incident_dir, interaction.pos, -photon.direction.cast<double>(), context, nullptr);
    Spectral3 weight = MaybeReWeightToMonochromatic(photon.weight.cast<double>()*scatter_val*kernel_val, photon.monochromatic | monochromatic);
    inscatter_estimator += weight;
    pickers->ObserveReturnPhoton(worker_index, photon.path_index, weight*path_factors);
#ifdef DEBUG_BUFFERS
    {
      Spectral3 w = interaction.sigma_s*path_weight*weight/master->num_photons_traced;
      AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_PH, photon.node_number, w);
      if (last_scatter_pdf_value)
        AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, w);
    }
#endif
  });
  RecordMeasurementToCurrentPixel(path_factors * inscatter_estimator);
}
#endif


void CameraRenderWorker::AddPhotonBeamContributions(const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &path_weight, const PathState &ps) const
{
  const auto path_factors = (path_weight * (1.0 / master->num_photons_traced)).eval();
  Spectral3 inscatter_estimator{0.}; // Integral over the query ray.
  int photon_idx[1024];
  float photon_distance[1024];
  const int n = master->beampointaccel->Query(segment.ray, std::min<double>(segment.length, pct.End()), photon_idx, photon_distance, 1024);
  for (int i=0; i<n; ++i)
  {
    const auto &photon = master->photons_volume[photon_idx[i]];
    // There is no -1 because  there is no interaction point on the query path. The interaction comes from the photon.
    if (photon.node_number + ps.current_node_count > ray_termination.max_node_count) 
      continue;
    Spectral3 scatter_val = medium.EvaluatePhaseFunction(-segment.ray.dir, photon.position, -photon.direction.cast<double>(), ps.context, nullptr);
    auto [sigma_s, _] = medium.EvaluateCoeffs(photon.position, ps.context);
    const double kernel_value = EvalKernel(kernel2d, photon.position, segment.ray.PointAt(photon_distance[i]));
    Spectral3 weight = MaybeReWeightToMonochromatic(kernel_value*scatter_val*sigma_s*photon.weight.cast<double>()*pct(photon_distance[i]), photon.monochromatic | ps.monochromatic);
    inscatter_estimator += weight;
    pickers->ObserveReturnPhoton(worker_index, photon.path_index, weight*path_factors);
#ifdef DEBUG_BUFFERS
    {
      Spectral3 w = weight/master->num_photons_traced;
      AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_PH, photon.node_number, w);
      if (last_scatter_pdf_value)
        AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, w);
    }
#endif
  }
  inscatter_estimator *= path_factors;
  RecordMeasurementToCurrentPixel(inscatter_estimator, ps);
}

void CameraRenderWorker::RecordMeasurementToCurrentPixel(const Spectral3 &measurement, const PathState &ps) const
{
  auto color = Color::SpectralSelectionToRGB(measurement, lambda_selection.indices);
  framebuffer[ps.context.pixel_index] += color;
}


#ifdef DEBUG_BUFFERS
void CameraRenderWorker::AddToDebugBuffer(int id, int level, const Spectral3 &measurement_estimate)
{
    assert (level >= 0);
    if (id == PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_PH && (level >= PhotonmappingRenderingAlgo::DEBUGBUFFER_DEPTH))
      return;
    Spectral3ImageBuffer &buf = master->debugbuffers[id+level];
    auto color = Color::SpectralSelectionToRGB(measurement_estimate, lambda_selection.indices);
    buf.Insert(pixel_index, color);
}
#endif


} // Namespace


std::unique_ptr<RenderingAlgo> AllocatePhotonmappingRenderingAlgo(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<Photonmapping::PhotonmappingRenderingAlgo>(scene, params);
}
