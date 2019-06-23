#include <functional>

#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>

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




namespace Photonmapping
{
using SimplePixelByPixelRenderingDetails::SamplesPerPixelSchedule;
using SimplePixelByPixelRenderingDetails::SamplesPerPixelScheduleConstant;
class PhotonmappingRenderingAlgo;


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
//#define LOGGING

struct Photon
{
  Double3 position;
  Spectral3 weight;
  Float3 direction;
  short node_number; // Starts at 2. First node is on the light source.
  bool monochromatic;
};


class PhotonmappingWorker
{
  friend struct EmitterSampleVisitor;
  PhotonmappingRenderingAlgo *master;
  TrivialLightPicker light_picker;
  MediumTracker medium_tracker;
  Sampler sampler;
  int current_node_count = 0;
  Spectral3 current_emission;
  LambdaSelection lambda_selection;
  PathContext context;
  RayTermination ray_termination;
  bool monochromatic = false;
private:
  Ray GeneratePrimaryRay(Spectral3 &path_weight);
  bool TrackToNextInteractionAndScatter(Ray &ray, Spectral3 &weight_accum);
  bool ScatterAt(Ray &ray, const SurfaceInteraction &interaction, const Spectral3 &track_weigh, Spectral3 &weight_accum);
  bool ScatterAt(Ray &ray, const VolumeInteraction &interaction, const Spectral3 &track_weigh, Spectral3 &weight_accum);
public:  
  // Public access because this class is hardly a chance to missuse.
  // Simply run TracePhotons as much as desired, then get the photons
  // from these members. Then destroy the instance.
  ToyVector<Photon> photons_volume;
  ToyVector<Photon> photons_surface;
  int num_photons_traced = 0;
  SpectralN max_throughput_weight{0};
  double max_bsdf_correction_weight{0};
  SpectralN max_uncorrected_bsdf_weight{0};
#ifdef LOGGING
  bool should_log_path = false;
  static constexpr double log_path_weight_threshold = 1000.;
  Pathlogger logger;
#endif
  PhotonmappingWorker(PhotonmappingRenderingAlgo *master);
  void StartNewPass(const LambdaSelection &lambda_selection);
  void TracePhotons(int count);
};


class CameraRenderWorker
{
  friend struct EmitterDirectLightingVisitor;
  PhotonmappingRenderingAlgo *master;
  TrivialLightPicker light_picker;
  MediumTracker medium_tracker;
  Sampler sampler;
  LambdaSelection lambda_selection;
  PathContext context;
  RayTermination ray_termination;
  int current_node_count = 0;
  bool monochromatic = false;
  int pixel_index = 0;
  Kernel2d kernel2d;
  Kernel3d kernel3d;
  boost::optional<Pdf> last_scatter_pdf_value; // For MIS.
private:
  Ray GeneratePrimaryRay(Spectral3 &path_weight);
  bool TrackToNextInteractionAndRecordPixel(Ray &ray, Spectral3 &weight_accum);
  void AddPhotonContributions(const SurfaceInteraction &interaction, const Double3 &incident_dir, const Spectral3 &path_weight);
  void AddPhotonContributions(const VolumeInteraction& interaction, const Double3& incident_dir, const Spectral3& path_weight);
  bool MaybeScatterAtSpecularLayer(Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum, const Spectral3 &track_weight);
  void RecordMeasurementToCurrentPixel(const Spectral3 &measurement);
  void MaybeAddEmission(const Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum);
  void MaybeAddEmission(const Ray& ray, const VolumeInteraction &interaction, Spectral3& weight_accum);
  void AddEnvEmission(const Double3 &travel_dir, const Spectral3 &path_weight);
  void AddPhotonBeamContributions(const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &track_weight);
  void MaybeAddDirectLighting(const Double3 &travel_dir, const SurfaceInteraction &interaction, const Spectral3& weight_accum);
  void AddToImageBuffer(const Spectral3 &measurement_estimate);
#ifdef DEBUG_BUFFERS
  void AddToDebugBuffer(int id, int level, const Spectral3 &measurement_estimate);
#endif
public:
  CameraRenderWorker(PhotonmappingRenderingAlgo *master);
  void StartNewPass(const LambdaSelection &lambda_selection);
  void Render(int pixel_index, int batch_size, int samples_per_pixel);

};


class PhotonmappingRenderingAlgo : public RenderingAlgo
{
  friend class PhotonmappingWorker;
  friend class CameraRenderWorker;
private:
  static constexpr int PIXEL_STRIDE = 64 / sizeof(Spectral3); // Because false sharing.
  Spectral3ImageBuffer buffer;
  std::unique_ptr<HashGrid> hashgrid_volume; // Photon Lookup
  std::unique_ptr<HashGrid> hashgrid_surface;
  std::unique_ptr<PhotonIntersector> beampointaccel;
  ToyVector<Photon> photons_volume;
  ToyVector<Photon> photons_surface;
  int num_threads = 1;
  int num_pixels = 0;
  int pass_index = 0;
  int num_photons_traced = 0;
  double current_surface_photon_radius = 0;
  double current_volume_photon_radius = 0;
  double radius_reduction_alpha = 2./3.; // Less means faster reduction. 2/3 is suggested in the unified path sampling paper.
  SamplesPerPixelScheduleConstant spp_schedule;
  tbb::atomic<int> shared_pixel_index = 0;
  tbb::task_group the_task_group;
  tbb::atomic<bool> stop_flag = false;
  tbb::spin_mutex buffer_mutex;
  Sampler sampler;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  ToyVector<PhotonmappingWorker> photonmap_workers;
  ToyVector<CameraRenderWorker> camerarender_workers;
#ifdef DEBUG_BUFFERS  
  ToyVector<Spectral3ImageBuffer> debugbuffers;
  static constexpr int DEBUGBUFFER_DEPTH = 10;
  static constexpr int DEBUGBUFFER_ID_PH = 0;
  static constexpr int DEBUGBUFFER_ID_BSDF = 10;
  static constexpr int DEBUGBUFFER_ID_DIRECT = 11;
#endif
  SpectralN max_throughput_weight{0.};
  double max_bsdf_correction_weight{0};
  SpectralN max_uncorrected_bsdf_weight{0};
public:
  const RenderingParameters &render_params;
  const Scene &scene;
public:
  PhotonmappingRenderingAlgo(const Scene &scene_, const RenderingParameters &render_params_)
    : RenderingAlgo{}, buffer{render_params_.width, render_params_.height}, 
       num_threads{0}, spp_schedule{render_params_},  render_params{render_params_}, scene{scene_}
      
  {
    num_pixels = render_params.width * render_params.height;
    current_surface_photon_radius = render_params.initial_photon_radius;
    current_volume_photon_radius = render_params.initial_photon_radius;
#ifdef DEBUG_BUFFERS
    for (int i=0; i<DEBUGBUFFER_DEPTH+2; ++i)
    {
      debugbuffers.emplace_back(render_params.width, render_params.height);
    }
#endif
  }
  
  void Run() override
  {
    for (int i=0; i<std::max(1, this->render_params.num_threads); ++i)
    {
      photonmap_workers.emplace_back(this);
      camerarender_workers.emplace_back(this);
    } 
    num_threads = isize(photonmap_workers);
    while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
    {
      // Full sweep across spectrum counts as one pass.
      // But one sweep will require multiple photon mapping passes since 
      // each pass only covers part of the spectrum.
      for (int spectrum_sweep_idx=0; 
           (spectrum_sweep_idx <= decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED) && !stop_flag.load();
           ++spectrum_sweep_idx)
      {
        shared_pixel_index = 0;
        auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
        tbb::parallel_for(0, (int)photonmap_workers.size(), [&](int i) {
          photonmap_workers[i].StartNewPass(lambda_selection);
          camerarender_workers[i].StartNewPass(lambda_selection);
        });
        while_parallel_fed_interruptible(
          /*func=*/[this](int, int worker_num)
          {
            photonmap_workers[worker_num].TracePhotons(PIXEL_STRIDE*GetSamplesPerPixel());
          },
          /*feeder=*/[this]() -> boost::optional<int>
          {
            return this->FeedPixelIndex();
          },
          /*irq_handler=*/[this]() -> bool
          {
            return !(this->stop_flag.load());
          },
          num_threads, the_task_group);
        
        PrepareGlobalPhotonMap();

        buffer.AddSampleCount(GetSamplesPerPixel()); 
#ifdef DEBUG_BUFFERS
        for (auto &b : debugbuffers) b.AddSampleCount(GetSamplesPerPixel());
#endif
        
        shared_pixel_index = 0;
        while_parallel_fed_interruptible(
          /*func=*/[this](int pixel_index, int worker_num)
          {
            camerarender_workers[worker_num].Render(pixel_index, PIXEL_STRIDE, GetSamplesPerPixel());
          },
          /*feeder=*/[this]() -> boost::optional<int>
          {
            return this->FeedPixelIndex();
          },
          /*irq_handler=*/[this]() -> bool
          {
            if (this->stop_flag.load())
              return false;
            this->CallInterruptCb(false);
            return true;
          },
          num_threads, the_task_group);      
      }
      std::cout << "Sweep " << spp_schedule.GetTotal() << " finished" << std::endl;
      CallInterruptCb(true);
      spp_schedule.UpdateForNextPass();
      ++pass_index;
      UpdatePhotonRadii();
    } // Pass iteration
  }
  
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
  
  std::unique_ptr<Image> GenerateImage() override
  {
    auto bm = std::make_unique<Image>(render_params.width, render_params.height);
    tbb::parallel_for(0, render_params.height, [&](int row){
      buffer.ToImage(*bm, row, row+1);  
    });
#ifdef DEBUG_BUFFERS
    int buf_num = 0;
    for (auto &b : debugbuffers)
    {
      Image im(render_params.width, render_params.height);
      tbb::parallel_for(0, render_params.height, [&](int row){
        b.ToImage(im, row, row+1, /*convert_linear_to_srgb=*/ false);  
      });
      im.write(strconcat("/tmp/debug",(buf_num++),".png"));
    }
    {
    Image im(render_params.width, render_params.height);
    tbb::parallel_for(0, render_params.height, [&](int row){
      buffer.ToImage(im, row, row+1, /*convert_linear_to_srgb=*/ false);  
    });
    im.write("/tmp/debug.png");
    }
#endif
    return bm;
  }
  
protected:
  inline int GetNumPixels() const { return num_pixels; }
  
  inline int GetSamplesPerPixel() const { return spp_schedule.GetPerIteration(); }
  
private:
  boost::optional<int> FeedPixelIndex()
  {
    int i = shared_pixel_index.fetch_and_add(PIXEL_STRIDE);
    return i <= num_pixels ? i : boost::optional<int>();
  }
  
  void PrepareGlobalPhotonMap();
  
  void UpdatePhotonRadii();
};


void PhotonmappingRenderingAlgo::PrepareGlobalPhotonMap()
{
  photons_volume.clear();
  photons_surface.clear();
  num_photons_traced = 0;

  for(auto &worker : photonmap_workers)
  {
    photons_volume.insert(photons_volume.end(), worker.photons_volume.begin(), worker.photons_volume.end());
    photons_surface.insert(photons_surface.end(), worker.photons_surface.begin(), worker.photons_surface.end());
    num_photons_traced += worker.num_photons_traced;
    max_throughput_weight = max_throughput_weight.cwiseMax(worker.max_throughput_weight);
    max_bsdf_correction_weight = std::max(max_bsdf_correction_weight,worker.max_bsdf_correction_weight);
    max_uncorrected_bsdf_weight = max_uncorrected_bsdf_weight.cwiseMax(worker.max_uncorrected_bsdf_weight);
  }
  std::cout << "Max throughput = " << max_throughput_weight << std::endl;
  std::cout << "max_bsdf_correction_weight = " << max_bsdf_correction_weight << std::endl;
  std::cout << "max_uncorrected_bsdf_weight= " << max_uncorrected_bsdf_weight << std::endl;
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
  Spectral3 weight;
  EmitterSampleVisitor(PhotonmappingWorker* this_) : this_{this_}, master{this_->master} {}
  
  void operator()(const ROI::EnvironmentalRadianceField &env, double prob)
  {
    auto smpl = env.TakeDirectionSample(this_->sampler, this_->context);
    EnvLightPointSamplingBeyondScene gen{master->scene};
    double pdf = gen.Pdf(smpl.coordinates);
    Double3 org = gen.Sample(smpl.coordinates, this_->sampler);
    weight = smpl.value / (prob*(double)smpl.pdf_or_pmf*(double)pdf);
    out_ray = Ray{org, smpl.coordinates};
    this_->medium_tracker.initializePosition(org);
  }
  void operator()(const ROI::PointEmitter &light, double prob)
  {
    Double3 org = light.Position();
    auto smpl = light.TakeDirectionSampleFrom(org, this_->sampler, this_->context);
    weight = smpl.value / (prob*(double)smpl.pdf_or_pmf);
    out_ray = Ray{org, smpl.coordinates};
    this_->medium_tracker.initializePosition(org);
  }
  void operator()(const PrimRef& prim_ref, double prob)
  {
    const auto &mat = master->scene.GetMaterialOf(prim_ref);
    assert(mat.emitter); // Otherwise would would not have been selected.
    auto area_smpl = mat.emitter->TakeAreaSample(prim_ref, this_->sampler, this_->context);
    auto dir_smpl = mat.emitter->TakeDirectionSampleFrom(area_smpl.coordinates, this_->sampler, this_->context);
    weight = dir_smpl.value / ((double)area_smpl.pdf_or_pmf*(double)dir_smpl.pdf_or_pmf*prob);
    SurfaceInteraction interaction{area_smpl.coordinates};
    weight *= DFactorPBRT(interaction, dir_smpl.coordinates);
    out_ray = Ray{interaction.pos, dir_smpl.coordinates};
    out_ray.org += AntiSelfIntersectionOffset(interaction, out_ray.dir);
    this_->medium_tracker.initializePosition(out_ray.org);
  }
  void operator()(const Medium &medium, double prob)
  {
//     Medium::VolumeSample smpl = medium.SampleEmissionPosition(sampler, context);
//     double pdf_which_cannot_be_delta = 0;
//     auto radiance =  medium.EvaluateEmission(smpl.pos, context, &pdf_which_cannot_be_delta);
//     node.node_type = RW::NodeType::VOLUME_EMITTER;
//     node.interaction.volume = VolumeInteraction(smpl.pos, medium, radiance, Spectral3{0.});
//     // The reason sigma_s is set to 0 in the line above, is that the initial light node will never be used for scattering.
//     pdf = prob*pdf_which_cannot_be_delta;
  }
};






PhotonmappingWorker::PhotonmappingWorker(PhotonmappingRenderingAlgo *master)
  : master{master},
    light_picker{master->scene},
    medium_tracker{master->scene},
    context{},
    ray_termination{master->render_params}
{
  photons_surface.reserve(1024);
  photons_volume.reserve(1024);
}

void PhotonmappingWorker::StartNewPass(const LambdaSelection &lambda_selection)
{
  this->lambda_selection = lambda_selection;
  context = PathContext{lambda_selection, TransportType::IMPORTANCE};
//   photon_volume_blur_weight = 1.0/(UnitSphereVolume*Cubed(master->current_volume_photon_radius));
//   photon_surface_blur_weight = 1.0/(UnitDiscSurfaceArea*Sqr(master->current_surface_photon_radius));
  photons_surface.clear();
  photons_volume.clear();
  num_photons_traced = 0;
  max_throughput_weight.setConstant(0.);
}


Ray PhotonmappingWorker::GeneratePrimaryRay(Spectral3 &path_weight)
{
  EmitterSampleVisitor emission_visitor{this};
  light_picker.PickLight(sampler, emission_visitor);
  path_weight *= emission_visitor.weight;
  return emission_visitor.out_ray;
}


void PhotonmappingWorker::TracePhotons(int count)
{
  for (int photon_num = 0; photon_num < count; ++photon_num, ++num_photons_traced)
  {
    current_node_count = 2;
    monochromatic = false;
    // Weights due to wavelength selection is accounted for in view subpath weights.
    current_emission.setConstant(1.);
    Ray ray = GeneratePrimaryRay(current_emission);
#ifdef LOGGING
    {
      should_log_path = false;
      logger.NewPath();
      auto &l = logger.AddNode();
      l.position = ray.org;
      l.exitant_dir = ray.dir;
      l.weight_after = Spectral3::Ones();
    }
#endif    
    bool keepgoing = true;
    Spectral3 weight_accum{1.}; 
    do
    {
      keepgoing = TrackToNextInteractionAndScatter(ray, weight_accum);
    }
    while (keepgoing);
#ifdef LOGGING
    if (should_log_path)
    {
      logger.WritePath();
    }
#endif
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
      weight_accum *= track_weight;
      if (!GetShaderOf(interaction, master->scene).prefer_path_tracing_over_photonmap)
      {
        photons_surface.push_back({
          interaction.pos,
          /*photon_surface_blur_weight**/weight_accum*current_emission,
          ray.dir.cast<float>(),
          (short)current_node_count,
          monochromatic
        });
      }
      current_node_count++;
      SpectralMaxInplace(max_throughput_weight, lambda_selection, weight_accum);
      return ScatterAt(ray, interaction, track_weight, weight_accum);
    },
    /*volume=*/[&](const VolumeInteraction &interaction, double distance, const Spectral3 &track_weight) -> bool
    {
      weight_accum *= track_weight;
      photons_volume.push_back({
        interaction.pos,
        /*photon_volume_blur_weight**/weight_accum*current_emission,
        ray.dir.cast<float>(),
        (short)current_node_count,
        monochromatic
      });
      current_node_count++;
      SpectralMaxInplace(max_throughput_weight, lambda_selection, weight_accum);
      return ScatterAt(ray, interaction, track_weight, weight_accum);
    },
    /*escape*/[&](const Spectral3 &weight) -> bool
    {
      return false;
    }
  );
  return keepgoing;
}


bool PhotonmappingWorker::ScatterAt(Ray& ray, const SurfaceInteraction& interaction, const Spectral3 &, Spectral3& weight_accum)
{
#ifdef LOGGING
  auto &l = logger.AddNode();
  l.position = interaction.pos;
  l.normal = interaction.shading_normal;
  l.geom_normal = interaction.normal;
  l.incident_dir = ray.dir;
  l.weight_before = weight_accum;
should_log_path |= weight_accum.maxCoeff() > log_path_weight_threshold;
#endif
  const auto &shader = GetShaderOf(interaction,master->scene);
  auto smpl = shader.SampleBSDF(-ray.dir, interaction, sampler, context);
  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler))
  {
    monochromatic |= shader.require_monochromatic;
    smpl.value *= DFactorPBRT(interaction,smpl.coordinates) / smpl.pdf_or_pmf;
    SpectralMaxInplace(max_uncorrected_bsdf_weight, lambda_selection, smpl.value);
    auto corr = BsdfCorrectionFactorPBRT(-ray.dir, interaction, smpl.coordinates, 2.);
    smpl.value *= corr;
    max_bsdf_correction_weight = std::max(max_bsdf_correction_weight, corr);
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
    l.exitant_dir = ray.dir;
    l.weight_after = weight_accum;
    should_log_path |= weight_accum.maxCoeff() > log_path_weight_threshold;
#endif
    return true;
  }
  else 
    return false;
}


bool PhotonmappingWorker::ScatterAt(Ray& ray, const VolumeInteraction& interaction, const Spectral3 &track_weight, Spectral3& weight_accum)
{
#ifdef LOGGING
  auto &l = logger.AddNode();
  l.position = interaction.pos;
  l.incident_dir = ray.dir;
  l.weight_before = weight_accum;
  l.is_surface = false;
  should_log_path |= weight_accum.maxCoeff() > log_path_weight_threshold;
#endif
  auto smpl = interaction.medium().SamplePhaseFunction(-ray.dir, interaction.pos, sampler, context);
  bool survive = ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler);
  weight_accum *= interaction.sigma_s*smpl.value / smpl.pdf_or_pmf;
  ray.dir = smpl.coordinates;
  ray.org = interaction.pos;
#ifdef LOGGING
  l.exitant_dir = ray.dir;
  l.weight_after = weight_accum;
  should_log_path |= weight_accum.maxCoeff() > log_path_weight_threshold;
#endif
  return survive;
}



CameraRenderWorker::CameraRenderWorker(Photonmapping::PhotonmappingRenderingAlgo* master)
  : master{master}, 
    light_picker{master->scene},
    medium_tracker{master->scene}, context{}, 
    ray_termination{master->render_params}
{
}

void CameraRenderWorker::StartNewPass(const LambdaSelection& lambda_selection)
{
  this->lambda_selection = lambda_selection;
  context = PathContext{lambda_selection, TransportType::RADIANCE};
  kernel2d = Kernel2d(master->current_surface_photon_radius);
  kernel3d = Kernel3d(master->current_volume_photon_radius);
}


void CameraRenderWorker::Render(int pixel_index_, int batch_size, int samples_per_pixel)
{
  const int end = std::min(master->num_pixels, pixel_index_+batch_size);
  for (this->pixel_index = pixel_index_; pixel_index<end; ++pixel_index)
  {
    for (int i=0; i<samples_per_pixel; ++i)
    {
      current_node_count = 2; // First node on camera. We start with the node code of the next interaction.
      monochromatic = false;
      Spectral3 weight_accum = lambda_selection.weights;
      Ray ray = GeneratePrimaryRay(weight_accum);
      bool keepgoing = true;
      do
      {
        keepgoing = TrackToNextInteractionAndRecordPixel(ray, weight_accum);
      }
      while (keepgoing);
    }
  }
}


void CameraRenderWorker::RecordMeasurementToCurrentPixel(const Spectral3 &measurement)
{
  auto color = Color::SpectralSelectionToRGB(measurement, lambda_selection.indices);
  master->buffer.Insert(pixel_index, color);
}


Ray CameraRenderWorker::GeneratePrimaryRay(Spectral3& path_weight)
{
    const ROI::PointEmitterArray& camera = master->scene.GetCamera();
    auto pos = camera.TakePositionSample(pixel_index, sampler, context);
    path_weight *= pos.value / pos.pdf_or_pmf;
    auto dir = camera.TakeDirectionSampleFrom(pixel_index, pos.coordinates, sampler, context);
    path_weight *= dir.value / dir.pdf_or_pmf;
    medium_tracker.initializePosition(pos.coordinates);
    last_scatter_pdf_value = boost::none;
    return Ray{pos.coordinates, dir.coordinates};
}


bool CameraRenderWorker::TrackToNextInteractionAndRecordPixel(Ray& ray, Spectral3& weight_accum)
{
#if 0
  return TrackToNextInteraction(master->scene, ray, context, weight_accum, sampler, medium_tracker, nullptr,
    /*surface=*/[&](const SurfaceInteraction &interaction, double distance, const Spectral3 &track_weight) -> bool
    {
      weight_accum *= track_weight;
      AddPhotonContributions(interaction, ray.dir, weight_accum);
      MaybeAddEmission(ray, interaction, weight_accum);
      current_node_count++;
      return MaybeScatterAtSpecularLayer(ray, interaction, weight_accum, track_weight);
    },
    /*volume=*/[&](const VolumeInteraction &interaction, double distance, const Spectral3 &track_weight) -> bool
    {
      weight_accum *= track_weight;
      AddPhotonContributions(interaction, ray.dir, weight_accum);
      MaybeAddEmission(ray, interaction, weight_accum);
      return false;
    },
    /*escape*/[&](const Spectral3 &track_weight) -> bool
    {
      const auto &emitter = master->scene.GetTotalEnvLight();
      const auto radiance = emitter.Evaluate(-ray.dir, context);
      RecordMeasurementToCurrentPixel(radiance * track_weight * weight_accum);
      return false;
    }
  );
#else
  bool keepgoing = false;
  TrackBeam(master->scene, ray, context, sampler, medium_tracker,
    /*surface_visitor=*/[&](const SurfaceInteraction &interaction, const Spectral3 &track_weight)
    {
      weight_accum *= track_weight;
      AddPhotonContributions(interaction, ray.dir, weight_accum);
      MaybeAddEmission(ray, interaction, weight_accum);
      MaybeAddDirectLighting(ray.dir, interaction, weight_accum);
      current_node_count++;
      keepgoing = MaybeScatterAtSpecularLayer(ray, interaction, weight_accum, track_weight);
    },
    /*segment visitor=*/[&](const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &track_weight)
    {
      AddPhotonBeamContributions(segment, medium, pct, weight_accum*track_weight);
    },
    /* escape_visitor=*/[&](const Spectral3 &track_weight)
    {
      AddEnvEmission(ray.dir, track_weight*weight_accum);
    }
  );
  return keepgoing;
#endif
}


struct EmitterDirectLightingVisitor
{
  CameraRenderWorker* this_;
  PhotonmappingRenderingAlgo *master;
  const SurfaceInteraction &surface;
  const Double3 reverse_incident_dir;
  
  RaySegment segment_to_light;
  Spectral3 weight;
  Pdf pdf;
  
  EmitterDirectLightingVisitor(CameraRenderWorker* this_, const Double3 &reverse_incident_dir, const SurfaceInteraction &surface) 
    : this_{this_}, master{this_->master}, surface{surface}, reverse_incident_dir{reverse_incident_dir} {}
  
  void operator()(const ROI::EnvironmentalRadianceField &env, double prob)
  {
    auto smpl = env.TakeDirectionSample(this_->sampler, this_->context);
    this->weight = smpl.value / (prob*(double)smpl.pdf_or_pmf);
    this->pdf = smpl.pdf_or_pmf;
    this->segment_to_light = SegmentToEnv(smpl.coordinates);

  }
  void operator()(const ROI::PointEmitter &light, double prob)
  {
    this->segment_to_light = SegmentToPoint(light.Position());
    this->weight = light.Evaluate(light.Position(), -this->segment_to_light.ray.dir, this_->context, nullptr);
    this->weight /= prob;
    this->weight /= Sqr(this->segment_to_light.length);
    this->pdf = Pdf::MakeFromDelta(1.);
  }
  void operator()(const PrimRef& prim_ref, double prob)
  {
    const auto &mat = master->scene.GetMaterialOf(prim_ref);
    assert(mat.emitter); // Otherwise would would not have been selected.
    ROI::AreaSample area_smpl = mat.emitter->TakeAreaSample(prim_ref, this_->sampler, this_->context);
    SurfaceInteraction light_surf{area_smpl.coordinates};
    this->segment_to_light = SegmentToPoint(light_surf.pos);
    this->weight = mat.emitter->Evaluate(area_smpl.coordinates, -segment_to_light.ray.dir, this_->context, nullptr);
    this->weight *= 1.0 / (prob * Sqr(this->segment_to_light.length) * (double)area_smpl.pdf_or_pmf);
    this->weight *= DFactorPBRT(surface, -segment_to_light.ray.dir);
    this->pdf = PdfConversion::AreaToSolidAngle(segment_to_light.length, segment_to_light.ray.dir, light_surf.normal)*area_smpl.pdf_or_pmf;
  }
  void operator()(const Medium &medium, double prob)
  {
      assert(!"Not implemented!");
//     Medium::VolumeSample smpl = medium.SampleEmissionPosition(sampler, context);
//     double pdf_which_cannot_be_delta = 0;
//     auto radiance =  medium.EvaluateEmission(smpl.pos, context, &pdf_which_cannot_be_delta);
//     node.node_type = RW::NodeType::VOLUME_EMITTER;
//     node.interaction.volume = VolumeInteraction(smpl.pos, medium, radiance, Spectral3{0.});
//     // The reason sigma_s is set to 0 in the line above, is that the initial light node will never be used for scattering.
//     pdf = prob*pdf_which_cannot_be_delta;
  }

private:
  RaySegment SegmentToEnv(const Double3 &dir)
  {
      Ray ray{surface.pos, dir};
      const double length = 10.*UpperBoundToBoundingBoxDiameter(master->scene);
      ray.org += AntiSelfIntersectionOffset(surface, ray.dir);
      return {ray, length};
  }
  
  RaySegment SegmentToPoint(const Double3 &pos)
  {
      Ray ray{surface.pos, pos - surface.pos}; 
      double length = Length(ray.dir);
      if (length > 0.)
        ray.dir /= length;
      else // Just pick something which will not result in NaN.
      {
        length = Epsilon;
        ray.dir = surface.normal;
      }
//       if constexpr (endpoint.on_surface)
//       {DirectLightSegmentToEnv(const SurfaceInteraction &surf
//         end_pos += AntiSelfIntersectionOffset(endpoint.surface, -ray.dir);
//         ray.dir = end_pos - ray.org;
//         length = Length(ray.dir);
//         ray.dir /= length;
//       }
      ray.org += AntiSelfIntersectionOffset(surface, ray.dir);
      return {ray, length};
  }
};


static double MisWeight(Pdf pdf_or_pmf_taken, double pdf_other)
{
    double mis_weight = 1.;
    if (!pdf_or_pmf_taken.IsFromDelta())
    {
        mis_weight = PowerHeuristic(pdf_or_pmf_taken, {pdf_other});
    }
    return mis_weight;
}


void CameraRenderWorker::MaybeAddDirectLighting(const Double3 &travel_dir, const SurfaceInteraction &interaction, const Spectral3& weight_accum)
{
    // To consider for direct lighting NEE with MIS.
    // Segment, DFactors, AntiselfIntersectionOffset, Medium Passage, Transmittance Estimate, BSDF/Le factor, inv square factor.
    // Mis weight. P_light, P_bsdf
    
    const auto &shader = GetShaderOf(interaction,master->scene);
    if (!shader.prefer_path_tracing_over_photonmap)
        return;
    
    EmitterDirectLightingVisitor lighting_visitor{this, -travel_dir, interaction};
    light_picker.PickLight(sampler, lighting_visitor);
    auto [ray, length] = lighting_visitor.segment_to_light;
    
    Spectral3 path_weight = weight_accum;
    path_weight *= DFactorPBRT(interaction, ray.dir);

    double bsdf_pdf = 0.;
    Spectral3 bsdf_weight = shader.EvaluateBSDF(-travel_dir, interaction, ray.dir, context, &bsdf_pdf);
    path_weight *= bsdf_weight;

    
    MediumTracker medium_tracker{this->medium_tracker}; // Copy because well don't want to keep modifications.
    MaybeGoingThroughSurface(medium_tracker, ray.dir, interaction);
    Spectral3 transmittance = TransmittanceEstimate(master->scene, lighting_visitor.segment_to_light, medium_tracker, context, sampler);
    path_weight *= transmittance;
    
    double mis_weight = MisWeight(lighting_visitor.pdf, bsdf_pdf);
    
#ifdef DEBUG_BUFFERS 
    AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_DIRECT, 0, path_weight*lighting_visitor.weight);
#endif
    AddToImageBuffer(mis_weight*path_weight*lighting_visitor.weight);
}


bool CameraRenderWorker::MaybeScatterAtSpecularLayer(Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum, const Spectral3 &track_weight)
{
  const auto &shader = GetShaderOf(interaction,master->scene);
  auto smpl = shader.SampleBSDF(-ray.dir, interaction, sampler, context);
  if (!smpl.pdf_or_pmf.IsFromDelta() && !shader.prefer_path_tracing_over_photonmap)
    return false;
  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler))
  {
    monochromatic |= shader.require_monochromatic;
    weight_accum *= smpl.value * DFactorPBRT(interaction, smpl.coordinates) / smpl.pdf_or_pmf;
    ray.dir = smpl.coordinates;
    ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ray.dir);
    MaybeGoingThroughSurface(medium_tracker, ray.dir, interaction);
    last_scatter_pdf_value = smpl.pdf_or_pmf;
    return true;
  }
  else
    return false;
}


void CameraRenderWorker::MaybeAddEmission(const Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum)
{
    const auto emitter = GetMaterialOf(interaction, master->scene).emitter; 
    if (!emitter)
        return;
         
    Spectral3 radiance = emitter->Evaluate(interaction.hitid, -ray.dir, context, nullptr);

    double mis_weight = 1.0;
    if (last_scatter_pdf_value) // Should be set if this is secondary ray.
    {
        const double prob_select = light_picker.PmfOfLight(interaction.hitid);
        const double area_pdf = emitter->EvaluatePdf(interaction.hitid, context);
        const double pdf_cvt = PdfConversion::AreaToSolidAngle(Length(ray.org-interaction.pos), ray.dir, interaction.normal);
        mis_weight = MisWeight(*last_scatter_pdf_value, prob_select*area_pdf*pdf_cvt);
    }

#ifdef DEBUG_BUFFERS 
    AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*weight_accum);
#endif
    RecordMeasurementToCurrentPixel(mis_weight*radiance*weight_accum);
}


void CameraRenderWorker::AddEnvEmission(const Double3 &travel_dir, const Spectral3 &path_weight)
{
    const auto &emitter = master->scene.GetTotalEnvLight();
    const auto radiance = emitter.Evaluate(-travel_dir, context);
    
    double mis_weight = 1.0;
    if (last_scatter_pdf_value) // Should be set if this is secondary ray.
    {
        const double prob_select = light_picker.PmfOfLight(emitter);
        const double pdf_env = emitter.EvaluatePdf(-travel_dir, context);
        mis_weight = MisWeight(*last_scatter_pdf_value, pdf_env*prob_select);
    }
  
#ifdef DEBUG_BUFFERS 
    AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*path_weight);
#endif
    RecordMeasurementToCurrentPixel(mis_weight*radiance*path_weight);
}


void CameraRenderWorker::MaybeAddEmission(const Ray& , const VolumeInteraction &interaction, Spectral3& weight_accum)
{
    // TODO: Implement MIS.
//   if (interaction.medium().is_emissive)
//   {
//     Spectral3 irradiance = interaction.medium().EvaluateEmission(interaction.pos, context, nullptr);
//     RecordMeasurementToCurrentPixel(irradiance*weight_accum/UnitSphereSurfaceArea);
//   }
}


inline Spectral3 MaybeReWeightToMonochromatic(const Spectral3 &w, bool monochromatic)
{
  return monochromatic ? Spectral3{w[0]*3._sp,0,0} : w;
}


void CameraRenderWorker::AddPhotonContributions(const SurfaceInteraction& interaction, const Double3 &incident_dir, const Spectral3& path_weight)
{
  const auto &shader = GetShaderOf(interaction,master->scene);
  if (shader.prefer_path_tracing_over_photonmap)
    return;
  Spectral3 reflect_estimator{0.};
  master->hashgrid_surface->Query(interaction.pos, [&](int photon_idx)
  {
    const auto &photon = master->photons_surface[photon_idx];
    // There is minus one because there are two coincident nodes, photon and camera node. But we only want to count one.
    if (photon.node_number + current_node_count - 1 > ray_termination.max_node_count)
      return;
    const double kernel_val = EvalKernel(kernel2d, photon.position, interaction.pos);
    if (kernel_val <= 0.)
      return;
    Spectral3 bsdf_val = shader.EvaluateBSDF(-incident_dir, interaction, -photon.direction.cast<double>(), context, nullptr);
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
    Spectral3 weight = MaybeReWeightToMonochromatic(photon.weight*bsdf_val, photon.monochromatic | monochromatic);
    reflect_estimator += weight;
#ifdef DEBUG_BUFFERS 
    {
      Spectral3 w = path_weight*weight/master->num_photons_traced;
      AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_PH, photon.node_number, w);
      if (last_scatter_pdf_value)
        AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, w);
    }
#endif
  });
  Spectral3 weight = path_weight*reflect_estimator*(1.0/master->num_photons_traced);
  AddToImageBuffer(weight);
}


void CameraRenderWorker::AddPhotonContributions(const VolumeInteraction& interaction, const Double3 &incident_dir, const Spectral3& path_weight)
{
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
    Spectral3 weight = MaybeReWeightToMonochromatic(photon.weight*scatter_val*kernel_val, photon.monochromatic | monochromatic);
    inscatter_estimator += weight;
#ifdef DEBUG_BUFFERS
    {
      Spectral3 w = interaction.sigma_s*path_weight*weight/master->num_photons_traced;
      AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_PH, photon.node_number, w);
      if (last_scatter_pdf_value)
        AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, w);
    }
#endif
  });
  Spectral3 weight = interaction.sigma_s*path_weight*inscatter_estimator*(1.0/master->num_photons_traced);
  AddToImageBuffer(weight);
}


void CameraRenderWorker::AddPhotonBeamContributions(const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &path_weight)
{
  Spectral3 inscatter_estimator{0.}; // Integral over the query ray.
  int photon_idx[1024];
  float photon_distance[1024];
  const int n = master->beampointaccel->Query(segment.ray, std::min<double>(segment.length, pct.End()), photon_idx, photon_distance, 1024);
  for (int i=0; i<n; ++i)
  {
    const auto &photon = master->photons_volume[photon_idx[i]];
    // There is no -1 because  there is no interaction point on the query path. The interaction comes from the photon.
    if (photon.node_number + current_node_count > ray_termination.max_node_count) 
      continue;
    Spectral3 scatter_val = medium.EvaluatePhaseFunction(-segment.ray.dir, photon.position, -photon.direction.cast<double>(), context, nullptr);
    auto [sigma_s, _] = medium.EvaluateCoeffs(photon.position, context);
    const double kernel_value = EvalKernel(kernel2d, photon.position, segment.ray.PointAt(photon_distance[i]));
    Spectral3 weight = MaybeReWeightToMonochromatic(kernel_value*scatter_val*sigma_s*photon.weight*pct(photon_distance[i]), photon.monochromatic | monochromatic);
    inscatter_estimator += weight;
#ifdef DEBUG_BUFFERS
    {
      Spectral3 w = weight/master->num_photons_traced;
      AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_PH, photon.node_number, w);
      if (last_scatter_pdf_value)
        AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, w);
    }
#endif
  }
  inscatter_estimator *= path_weight*(1.0/master->num_photons_traced);
  AddToImageBuffer(inscatter_estimator);
}


void CameraRenderWorker::AddToImageBuffer(const Spectral3& measurement_estimate)
{
  auto color = Color::SpectralSelectionToRGB(measurement_estimate, lambda_selection.indices);
  master->buffer.Insert(pixel_index, color);
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
