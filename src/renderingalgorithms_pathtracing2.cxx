#include <cmath>
#include <functional>

#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>
//#include <tbb/cache_aligned_allocator.h>
#include <tbb/flow_graph.h>

#include "scene.hxx"
#include "shader.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "camera.hxx"
#include "renderbuffer.hxx"
#include "util_thread.hxx"
#include "rendering_util.hxx"
#include "pathlogger.hxx"
#include "spectral.hxx"
#include "media_integrator.hxx"

#include "renderingalgorithms_interface.hxx"
#include "renderingalgorithms_simplebase.hxx"
#include "lightpicker_ucb.hxx"

namespace pathtracing2
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



struct PathState
{
  PathState(const Scene &scene)
    : medium_tracker{ scene },
    context{}
  {}

  MediumTracker medium_tracker;
  PathContext context;
  Ray ray;
  nullpath::Spectral33 weights_track;
  nullpath::Spectral33 weights_track_then_null;
  Spectral3 path_weights; // Excluding the factors for transmission
  std::optional<Pdf> last_scatter_pdf_value; // For MIS.
  double shader_roughness = 0.;
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
  LambdaSelectionStrategy lambda_selection_factory;
  //static constexpr int num_lambda_sweeps = decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED;
  bool use_nee = true;
  bool use_emission = true;

private:
  void InitializePathState(PathState &p, Int2 pixel) const;
  bool TrackToNextInteractionAndRecordPixel(PathState &ps) const;

  bool MaybeScatter(const SomeInteraction &interaction, PathState &ps) const;

  void MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps) const;
  void AddEnvEmission(const PathState &ps) const;
  
  void AddDirectLighting(const SomeInteraction &interaction, const PathState &ps) const;
  
  void RecordMeasurementToCurrentPixel(const Spectral3 &measurement, const PathState &ps) const;

  enum SmplNodeType : uint32_t
  {
    NEE_LIGHT = 1,
    BSDF = 0,
  };

  void PrepSamplerDimension(const PathState &ps, SmplNodeType t) const;

  std::pair<double,Spectral3> MisWeight(std::optional<Pdf> scatter_pdf, double scatter_pdf_other, const nullpath::Spectral33 &distance_pdfs, const nullpath::Spectral33 &distance_pdfs_other) const;

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
  inline int GetTotalSamplesPerPixel() const { return spp_schedule.GetTotal(); }
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
      framebuffer::ToImage(*bm, tile, AsSpan(this->framebuffer), this->samplesPerTile[i], !render_params.linear_output);
    }
  });
  return bm;
}




CameraRenderWorker::CameraRenderWorker(PathTracingAlgo2* master, int worker_index)
  : master{ master },
  pickers{ master->pickers.get() },
  sampler{ master->render_params.qmc },
  ray_termination{ master->render_params }
{
  framebuffer = AsSpan(master->framebuffer);
  if (master->render_params.pt_sample_mode == "bsdf")
  {
    use_nee = false;
  }
  else if (master->render_params.pt_sample_mode == "lights")
  {
    use_emission = false;
  }
}


void CameraRenderWorker::PrepSamplerDimension(const PathState &ps, SmplNodeType t) const
{
  sampler.SetSubsequenceId((ps.current_node_count << 1) | (uint32_t)t);
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
      sampler.SetPixelIndex({ ix, iy });
      for (int i = 0; i < samples_per_pixel; ++i)
      {
        sampler.SetPointNum(i + master->GetTotalSamplesPerPixel());
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


LambdaSelection SingleWavelength(const LambdaSelectionStrategy &lss, Sampler &sampler)
{
  // Only keep first wavelength, assuming that it will cover all strata.
  // Zero out weights for other indices.
  auto ls = lss.WithWeights(sampler);
  for (int i=1; i<static_size<Spectral3>(); ++i)
  {
    ls.indices[i] = ls.indices[0];
    ls.wavelengths[i] = ls.wavelengths[0];
    ls.weights[i] = 0.;
  }
  ls.weights[0] *= static_size<Spectral3>();
  return ls;
}


void CameraRenderWorker::InitializePathState(PathState &p, Int2 pixel) const
{
  const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);

  const auto& camera = master->scene.GetCamera();
  p.context = PathContext(lambda_selection, camera.PixelToUnit({ pixel[0], pixel[1] }));

  p.current_node_count = 0;
  p.monochromatic = false;
  p.path_weights = lambda_selection.weights;
  p.weights_track.setOnes();
  p.weights_track_then_null.setOnes();
  p.last_scatter_pdf_value = {};
  p.shader_roughness = 0.;
  
  PrepSamplerDimension(p, BSDF);
  auto pos = camera.TakePositionSample(p.context.pixel_index, sampler, p.context);
  p.path_weights *= pos.value / pos.pdf_or_pmf;
  p.current_node_count++;
  PrepSamplerDimension(p, BSDF);
  auto dir = camera.TakeDirectionSampleFrom(p.context.pixel_index, pos.coordinates, sampler, p.context);
  p.path_weights *= dir.value / dir.pdf_or_pmf;
  p.ray = { pos.coordinates, dir.coordinates };
  p.medium_tracker.initializePosition(pos.coordinates);
  // First node on camera. We start with the node code of the next interaction. Which is 2.
  p.current_node_count++;
}


bool CameraRenderWorker::TrackToNextInteractionAndRecordPixel(PathState &ps) const
{
  // TODO: fix the nonsensical use of initial weight in Atmosphere Material!!
  auto[interaction, tfar, factors] = nullpath::Tracking(master->scene, ps.ray, sampler, ps.medium_tracker, ps.context);
  
  // Careful here. First part is from delta tracking. Last part for connection, corresponding to NEE technique.
  ps.weights_track_then_null = ps.weights_track;
  nullpath::AccumulateContributions(ps.weights_track_then_null, factors.weights_nulls); 
  // Now update the pure tracking pdf.
  nullpath::AccumulateContributions(ps.weights_track, factors.weights_track);

  if (!interaction)
  {
    AddEnvEmission(ps);
    return false;
  }

  return mpark::visit(
    Overload(
      [&](const SurfaceInteraction &si)  {
        PrepSamplerDimension(ps, NEE_LIGHT);
        MaybeAddEmission(si, ps);
        AddDirectLighting(si, ps);
        PrepSamplerDimension(ps, BSDF);
        return MaybeScatter(si, ps);
      },
      [&](const VolumeInteraction &vi) {
        PrepSamplerDimension(ps, NEE_LIGHT);
        AddDirectLighting(vi, ps);
        PrepSamplerDimension(ps, BSDF);
        return MaybeScatter(vi, ps);
      }
  ), *interaction);
}


static double Average(const Spectral3 &x, const Spectral3 &y)
{
  return 0.5*(x.mean() + y.mean());
}


std::pair<double,Spectral3> CameraRenderWorker::MisWeight(std::optional<Pdf> scatter_pdf, double scatter_pdf_other, const nullpath::Spectral33 &distance_pdfs, const nullpath::Spectral33 &distance_pdfs_other) const
{
  /*

    f = (m_bsdf * f_bsdf + m_nee * f_nee)
    m_bsdf = (p_bsdf * p_dist_i) * c_missing_wavelengths / (sum_i [p_bsdf * p_bsdf_dist_i + p_nee * p_nee_dist_i])
    c_missing_wavelengths = num_wavelengths = 3
    -> m_bsdf = (p_bsdf * p_dist_i) / (p_bsdf * <{p_bsdf_dist_i}> + p_nee * <{p_nee_dist_i}>})
    <.> := average
  */

  if (!scatter_pdf || scatter_pdf->IsFromDelta() || !(use_nee && use_emission))
  {
    return { 1., distance_pdfs.rowwise().mean().eval() };
  }

  const Spectral3 distance_weights_total = nullpath::FixNan(distance_pdfs * (*scatter_pdf)).rowwise().mean();
  const Spectral3 distance_weights_other_total = nullpath::FixNan(distance_pdfs_other * scatter_pdf_other).rowwise().mean();

  const double mis_weight = *scatter_pdf;
  const Spectral3 distance_part_inv_contrib = distance_weights_total+distance_weights_other_total;
  assert((distance_part_inv_contrib > 0).all());

  return { mis_weight, distance_part_inv_contrib };
}


void CameraRenderWorker::AddDirectLighting(const SomeInteraction & interaction, const PathState & ps) const
{
  if (!use_nee)
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
  auto[ray, length] = segment_to_light;

  Spectral3 path_weights = ps.path_weights;

  MediumTracker medium_tracker{ ps.medium_tracker }; // Copy because well don't want to keep modifications.
  
  double bsdf_pdf = 0.;

  if (auto si = mpark::get_if<SurfaceInteraction>(&interaction))
  {
    // Surface specific
    path_weights *= DFactorPBRT(*si, ray.dir);
    const auto &shd = GetShaderOf(*si, master->scene);
    ShaderQuery query {
      std::cref(*si),
      std::cref(ps.context),
      ps.shader_roughness,
    };
    Spectral3 bsdf_weight = shd.EvaluateBSDF(-ps.ray.dir, query, ray.dir, &bsdf_pdf);
    path_weights *= bsdf_weight;
    MaybeGoingThroughSurface(medium_tracker, ray.dir, *si);
  }
  else // must be volume interaction
  {
    auto vi = mpark::get_if<VolumeInteraction>(&interaction);
    Spectral3 bsdf_weight = vi->medium().EvaluatePhaseFunction(-ps.ray.dir, vi->pos, ray.dir, ps.context, &bsdf_pdf);
    auto coeffs = vi->medium().EvaluateCoeffs(vi->pos, ps.context);
    
    path_weights *= bsdf_weight;
    path_weights *= coeffs.sigma_s / (coeffs.sigma_t + Epsilon);
  }

  auto factors = nullpath::Transmission(master->scene, segment_to_light, sampler, medium_tracker, ps.context);

  nullpath::AccumulateContributions(factors.weights_track, ps.weights_track);
  nullpath::AccumulateContributions(factors.weights_nulls, ps.weights_track);

  auto [mis_weight, mis_pdf_mix] = MisWeight(pdf, bsdf_pdf, factors.weights_nulls, factors.weights_track);

  /*
  contrib = distance_throughput * scatter_weight / distance_pdf
  -> MIS ->
  weighted_contrib = [ distance_pdf * scatter_pdfs^p / (sum_i sum_j distance_pdf_i scatter_pdf_j^p) ] distance_throughput * scatter_weight / distance_pdf
  ==
  MisWeight(scatter_pdfs,{scatter_pdf_j}) * distance_throughput * scatter_weight / (sum_i distance_pdf_i)
  */

  Spectral3 measurement_estimator = mis_weight*(path_weights*light_weight)/mis_pdf_mix;

  pickers->ObserveReturnNee(this->picker_local, light_ref, measurement_estimator);
  RecordMeasurementToCurrentPixel(measurement_estimator, ps);
}


namespace
{


ScatterSample SampleScatterer(const SurfaceInteraction &interaction, const Double3 &reverse_incident_dir, const Scene &scene, Sampler &sampler, PathState &ps)
{
  const auto &shd = GetShaderOf(interaction, scene);
  materials::ShaderQuery query{
    std::cref(interaction),
    std::cref(ps.context),
    ps.shader_roughness
  };
  ps.shader_roughness = std::max(shd.MyRoughness(query), ps.shader_roughness);
  return shd.SampleBSDF(reverse_incident_dir, query, sampler);
}

ScatterSample SampleScatterer(const VolumeInteraction &interaction, const Double3 &reverse_incident_dir, const Scene &scene, Sampler &sampler, PathState &ps)
{
  ScatterSample s = interaction.medium().SamplePhaseFunction(reverse_incident_dir, interaction.pos, sampler, ps.context);
  auto coeffs = interaction.medium().EvaluateCoeffs(interaction.pos, ps.context);
  s.value *= coeffs.sigma_s / (coeffs.sigma_t + Epsilon);
  return s;
}

void PrepareStateForAfterScattering(PathState &ps, const SurfaceInteraction &interaction, const Scene &scene, const ScatterSample &scatter_sample)
{
  ps.monochromatic |= GetShaderOf(interaction, scene).require_monochromatic;
  ps.path_weights *= scatter_sample.value * DFactorPBRT(interaction, scatter_sample.coordinates) / scatter_sample.pdf_or_pmf;
  ps.ray.dir = scatter_sample.coordinates;
  ps.ray.org = interaction.pos + AntiSelfIntersectionOffset(interaction, ps.ray.dir);
  MaybeGoingThroughSurface(ps.medium_tracker, ps.ray.dir, interaction);
  ps.last_scatter_pdf_value = scatter_sample.pdf_or_pmf;
}

void PrepareStateForAfterScattering(PathState &ps, const VolumeInteraction &interaction, const Scene &scene, const ScatterSample &scatter_sample)
{
  ps.path_weights *= scatter_sample.value / scatter_sample.pdf_or_pmf;
  ps.ray.dir = scatter_sample.coordinates;
  ps.ray.org = interaction.pos;
  ps.last_scatter_pdf_value = scatter_sample.pdf_or_pmf;
}


} // anonymous namespace


bool CameraRenderWorker::MaybeScatter(const SomeInteraction &interaction, PathState &ps) const
{
  auto smpl = mpark::visit([this, &ps, &scene{ master->scene }](auto &&ia) {
    return SampleScatterer(ia, -ps.ray.dir, scene, sampler, ps);
  }, interaction);
  
  ps.current_node_count++;

  if (ray_termination.SurvivalAtNthScatterNode(smpl.value, Spectral3{ 1. }, ps.current_node_count, sampler))
  {
    mpark::visit([&ps, &scene{ master->scene }, &smpl](auto &&ia) {
      return PrepareStateForAfterScattering(ps, ia, scene, smpl);
    }, interaction);
    return true;
  }
  else
    return false;
}




void CameraRenderWorker::MaybeAddEmission(const SurfaceInteraction &interaction, const PathState &ps) const
{
  if (!use_emission)
    return;

  const auto emitter = GetMaterialOf(interaction, master->scene).emitter;
  if (!emitter)
    return;

  Spectral3 radiance = emitter->Evaluate(interaction.hitid, -ps.ray.dir, ps.context, nullptr);

  auto [mis_weight, mis_pdf_mix] = [&]() {
    if (ps.last_scatter_pdf_value) // Should be set if this is secondary ray.
    {
      const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, interaction.hitid));
      const double area_pdf = emitter->EvaluatePdf(interaction.hitid, ps.context);
      const double pdf_cvt = PdfConversion::AreaToSolidAngle(Length(ps.ray.org - interaction.pos), ps.ray.dir, interaction.normal);
      return MisWeight(ps.last_scatter_pdf_value, prob_select*area_pdf*pdf_cvt, ps.weights_track, ps.weights_track_then_null);
    }
    else
    {
      return MisWeight({}, 1., ps.weights_track, ps.weights_track_then_null);
    }
  }();

  Spectral3 contributions = mis_weight*(radiance*ps.path_weights) / mis_pdf_mix;

#ifdef DEBUG_BUFFERS 
  AddToDebugBuffer(PhotonmappingRenderingAlgo::DEBUGBUFFER_ID_BSDF, 0, radiance*weight_accum);
#endif
  RecordMeasurementToCurrentPixel(contributions, ps);
}


void CameraRenderWorker::AddEnvEmission(const PathState &ps) const
{
  if (!use_emission || !master->scene.HasEnvLight())
    return;

  const auto &emitter = master->scene.GetTotalEnvLight();
  const auto radiance = emitter.Evaluate(-ps.ray.dir, ps.context);

  auto [mis_weight, mis_pdf_mix] = [&]() 
  {
    if (ps.last_scatter_pdf_value) // Should be set if this is secondary ray.
    {
      const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, emitter));
      const double pdf_env = emitter.EvaluatePdf(-ps.ray.dir, ps.context);
      return MisWeight(*ps.last_scatter_pdf_value, pdf_env*prob_select, ps.weights_track, ps.weights_track_then_null);
    }
    else
      return MisWeight({}, 1., ps.weights_track, ps.weights_track_then_null);
  }();

  Spectral3 contributions = mis_weight*(radiance*ps.path_weights) / mis_pdf_mix;
  
  RecordMeasurementToCurrentPixel(contributions, ps);
}




inline Spectral3 MaybeReWeightToMonochromatic(const Spectral3 &w, bool monochromatic)
{
  return monochromatic ? Spectral3{ w[0] * 3._sp,0,0 } : w;
}


void CameraRenderWorker::RecordMeasurementToCurrentPixel(const Spectral3 &measurement, const PathState &ps) const
{
  assert(measurement.isFinite().all());
  auto color = Color::SpectralSelectionToRGB(measurement, ps.context.lambda_idx);
  framebuffer[ps.context.pixel_index] += color;
}



} // namespace pathtracing2



std::unique_ptr<RenderingAlgo> AllocatePathtracing2RenderingAlgo(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<pathtracing2::PathTracingAlgo2>(scene, params);
}
