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

//#define DEBUG_BUFFERS
#define LOGGING

struct Photon
{
  Double3 position;
  Spectral3 weight;
  Float3 direction;
  int level;
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
//   double photon_volume_blur_weight;
//   double photon_surface_blur_weight;
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
  Pathlogger logger;
  
  PhotonmappingWorker(PhotonmappingRenderingAlgo *master);
  void StartNewPass(const LambdaSelection &lambda_selection);
  void TracePhotons(int count);
};


class CameraRenderWorker
{
  friend struct CameraSampleVisitor;
  PhotonmappingRenderingAlgo *master;
  MediumTracker medium_tracker;
  Sampler sampler;
  LambdaSelection lambda_selection;
  PathContext context;
  RayTermination ray_termination;
  int current_node_count = 0;
  int pixel_index = 0;
  double volume_photon_radius_2;
  double surface_photon_radius_2;
  double photon_volume_blur_weight;
  double photon_surface_blur_weight;
private:
  Ray GeneratePrimaryRay(Spectral3 &path_weight);
  bool TrackToNextInteractionAndRecordPixel(Ray &ray, Spectral3 &weight_accum);
  void AddPhotonContributions(const SurfaceInteraction &interaction, const Double3 &incident_dir, const Spectral3 &path_weight);
  void AddPhotonContributions(const VolumeInteraction& interaction, const Double3& incident_dir, const Spectral3& path_weight);
  bool MaybeScatterAtSpecularLayer(Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum, const Spectral3 &track_weight);
  void RecordMeasurementToCurrentPixel(const Spectral3 &measurement);
  void MaybeAddEmission(Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum);
  void MaybeAddEmission(Ray& ray, const VolumeInteraction &interaction, Spectral3& weight_accum);
  void AddPhotonBeamContributions(const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &track_weight);
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
#endif
  SpectralN max_throughput_weight{0.};
  double max_bsdf_correction_weight{0};
  SpectralN max_uncorrected_bsdf_weight{0};
public:
  const RenderingParameters &render_params;
  const Scene &scene;
public:
  PhotonmappingRenderingAlgo(const Scene &scene_, const RenderingParameters &render_params_)
    : RenderingAlgo{}, render_params{render_params_}, scene{scene_},
      buffer{render_params_.width, render_params_.height}, num_threads{0},
      spp_schedule{render_params_}
  {
    num_pixels = render_params.width * render_params.height;
    current_surface_photon_radius = render_params.initial_photon_radius;
    current_volume_photon_radius = render_params.initial_photon_radius;
#ifdef DEBUG_BUFFERS
    for (int i=0; i<10; ++i)
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
    num_threads = photonmap_workers.size();
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
    weight *= DFactorOf(interaction, dir_smpl.coordinates);
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
  context = PathContext{lambda_selection.indices, TransportType::IMPORTANCE};
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
    // Weights due to wavelength selection is accounted for in view subpath weights.
    current_emission.setConstant(1.);
    Ray ray = GeneratePrimaryRay(current_emission);
#ifdef LOGGING
    {
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
    if (weight_accum.maxCoeff() > 1000.)
    {
      logger.WritePath();
    }
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
      photons_surface.push_back({
        interaction.pos,
        /*photon_surface_blur_weight**/weight_accum*current_emission,
        ray.dir.cast<float>(),
        current_node_count
      });
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
        current_node_count
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
  l.incident_dir = ray.dir;
  l.weight_before = weight_accum;
#endif
  auto smpl = GetShaderOf(interaction,master->scene).SampleBSDF(-ray.dir, interaction, sampler, context);
  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler))
  {
    smpl.value *= DFactorOf(interaction, smpl.coordinates) / smpl.pdf_or_pmf;
    SpectralMaxInplace(max_uncorrected_bsdf_weight, lambda_selection, smpl.value);
    auto corr = BsdfCorrectionFactor(-ray.dir, interaction, smpl.coordinates, context.transport);
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
#endif
  auto smpl = interaction.medium().SamplePhaseFunction(-ray.dir, interaction.pos, sampler, context);
  bool survive = ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler);
  weight_accum *= interaction.sigma_s*smpl.value / smpl.pdf_or_pmf;
  ray.dir = smpl.coordinates;
  ray.org = interaction.pos;
#ifdef LOGGING
  l.exitant_dir = ray.dir;
  l.weight_after = weight_accum;
#endif
  return survive;
}



CameraRenderWorker::CameraRenderWorker(Photonmapping::PhotonmappingRenderingAlgo* master)
  : master{master}, medium_tracker{master->scene}, context{}, 
    ray_termination{master->render_params}
{
}

void CameraRenderWorker::StartNewPass(const LambdaSelection& lambda_selection)
{
  this->lambda_selection = lambda_selection;
  context = PathContext{lambda_selection.indices, TransportType::RADIANCE};
  surface_photon_radius_2 = Sqr(master->current_surface_photon_radius);
  volume_photon_radius_2 = Sqr(master->current_volume_photon_radius);
  photon_volume_blur_weight = 1.0/(UnitSphereVolume*Cubed(master->current_volume_photon_radius));
  photon_surface_blur_weight = 1.0/(UnitDiscSurfaceArea*Sqr(master->current_surface_photon_radius));
}


void CameraRenderWorker::Render(int pixel_index_, int batch_size, int samples_per_pixel)
{
  const int end = std::min(master->num_pixels, pixel_index_+batch_size);
  for (this->pixel_index = pixel_index_; pixel_index<end; ++pixel_index)
  {
    for (int i=0; i<samples_per_pixel; ++i)
    {
      current_node_count = 2;
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
      //AddPhotonContributions(interaction, ray.dir, weight_accum);
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
      current_node_count++;
      keepgoing = MaybeScatterAtSpecularLayer(ray, interaction, weight_accum, track_weight);
    },
    /*segment visitor=*/[&](const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &track_weight)
    {
      AddPhotonBeamContributions(segment, medium, pct, weight_accum*track_weight);
    },
    /* escape_visitor=*/[&](const Spectral3 &track_weight)
    {
      const auto &emitter = master->scene.GetTotalEnvLight();
      const auto radiance = emitter.Evaluate(-ray.dir, context);
      RecordMeasurementToCurrentPixel(radiance * track_weight * weight_accum);
    }
  );
  return keepgoing;
#endif
}


bool CameraRenderWorker::MaybeScatterAtSpecularLayer(Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum, const Spectral3 &track_weight)
{
  auto smpl = GetShaderOf(interaction,master->scene).SampleBSDF(-ray.dir, interaction, sampler, context);
  if (!smpl.pdf_or_pmf.IsFromDelta())
    return false;
  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{1.}, current_node_count, sampler))
  {
    weight_accum *= smpl.value * (BsdfCorrectionFactor(-ray.dir, interaction, smpl.coordinates, context.transport) *
                  DFactorOf(interaction, smpl.coordinates) / smpl.pdf_or_pmf);
    ray.dir = smpl.coordinates;
    ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ray.dir);
    if (Dot(ray.dir, interaction.normal) < 0.)
    {
      // By definition, intersection.normal points to where the intersection ray is coming from.
      // Thus we can determine if the sampled direction goes through the surface by looking
      // if the direction goes in the opposite direction of the normal.
      medium_tracker.goingThroughSurface(ray.dir, interaction);
    }
    return true;
  }
  else 
    return false;
}


void CameraRenderWorker::MaybeAddEmission(Ray& ray, const SurfaceInteraction &interaction, Spectral3& weight_accum)
{
  const auto emitter = GetMaterialOf(interaction, master->scene).emitter; 
  if (emitter)
  {
    Spectral3 radiance = emitter->Evaluate(interaction.hitid, -ray.dir, context, nullptr);
    RecordMeasurementToCurrentPixel(radiance*weight_accum);
  }
}


void CameraRenderWorker::MaybeAddEmission(Ray& , const VolumeInteraction &interaction, Spectral3& weight_accum)
{
  if (interaction.medium().is_emissive)
  {
    Spectral3 irradiance = interaction.medium().EvaluateEmission(interaction.pos, context, nullptr);
    RecordMeasurementToCurrentPixel(irradiance*weight_accum/UnitSphereSurfaceArea);
  }
}


void CameraRenderWorker::AddPhotonContributions(const SurfaceInteraction& interaction, const Double3 &incident_dir, const Spectral3& path_weight)
{
  Spectral3 reflect_estimator{0.};
    master->hashgrid_surface->Query(interaction.pos, [&](int photon_idx){
    const auto &photon = master->photons_surface[photon_idx];
    if ((photon.position - interaction.pos).squaredNorm() > surface_photon_radius_2)
      return;
    Spectral3 bsdf_val = GetShaderOf(interaction,master->scene).EvaluateBSDF(-incident_dir, interaction, -photon.direction.cast<double>(), context, nullptr);
    bsdf_val *= BsdfCorrectionFactor(-incident_dir, interaction, -photon.direction.cast<double>(), context.transport);
    //bsdf_val *= DFactorOf(interaction, -photon.direction.cast<double>());
    reflect_estimator += photon.weight*bsdf_val;
#ifdef DEBUG_BUFFERS
    if (photon.level < master->debugbuffers.size()) {
      Spectral3 w = path_weight*photon.weight*bsdf_val*photon_surface_blur_weight/master->num_photons_traced;
      auto color = Color::SpectralSelectionToRGB(w, lambda_selection.indices);
      master->debugbuffers[photon.level].Insert(pixel_index, color);
    }
#endif
  });
  Spectral3 weight = path_weight*reflect_estimator*(photon_surface_blur_weight/master->num_photons_traced);
  auto color = Color::SpectralSelectionToRGB(weight, lambda_selection.indices);
  master->buffer.Insert(pixel_index, color);
}


void CameraRenderWorker::AddPhotonContributions(const VolumeInteraction& interaction, const Double3 &incident_dir, const Spectral3& path_weight)
{
  Spectral3 inscatter_estimator{0.};
  master->hashgrid_volume->Query(interaction.pos, [&](int photon_idx){
    const auto &photon = master->photons_volume[photon_idx];
    if ((photon.position - interaction.pos).squaredNorm() > volume_photon_radius_2)
      return;
    Spectral3 scatter_val = interaction.medium().EvaluatePhaseFunction(-incident_dir, interaction.pos, -photon.direction.cast<double>(), context, nullptr);
    inscatter_estimator += photon.weight*scatter_val;
#ifdef DEBUG_BUFFERS
    if (photon.level < master->debugbuffers.size()) {
      Spectral3 w = interaction.sigma_s*path_weight*photon.weight*scatter_val*photon_volume_blur_weight*master->num_photons_traced;
      auto color = Color::SpectralSelectionToRGB(w, lambda_selection.indices);
      master->debugbuffers[photon.level].Insert(pixel_index, color);
    }
#endif
  });
  Spectral3 weight = interaction.sigma_s*path_weight*inscatter_estimator*(photon_volume_blur_weight/master->num_photons_traced);
  auto color = Color::SpectralSelectionToRGB(weight, lambda_selection.indices);
  master->buffer.Insert(pixel_index, color);
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
    Spectral3 scatter_val = medium.EvaluatePhaseFunction(-segment.ray.dir, photon.position, -photon.direction.cast<double>(), context, nullptr);
    auto [sigma_s, _] = medium.EvaluateCoeffs(photon.position, context);
    inscatter_estimator += scatter_val*sigma_s*photon.weight*pct(photon_distance[i]);
  }
  inscatter_estimator *= path_weight*(photon_surface_blur_weight/master->num_photons_traced);
  auto color = Color::SpectralSelectionToRGB(inscatter_estimator, lambda_selection.indices);
  master->buffer.Insert(pixel_index, color);
}


} // Namespace


std::unique_ptr<RenderingAlgo> AllocatePhotonmappingRenderingAlgo(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<Photonmapping::PhotonmappingRenderingAlgo>(scene, params);
}
