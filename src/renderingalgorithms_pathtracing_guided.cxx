#include <functional>
#include <type_traits>
#include <optional>

#include <tbb/atomic.h>
#include <tbb/mutex.h>
//#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/cache_aligned_allocator.h>
//#include <tbb/flow_graph.h>

//#include <range/v3/view/transform.hpp>
//#include <range/v3/view/generate_n.hpp>
//#include <range/v3/range/conversion.hpp>
//#include <range/v3/numeric/accumulate.hpp>

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

inline void GenerateImage(Image &img, Span<const RGB> framebuffer, Span<const uint64_t> samples_per_tile, const ImageTileSet &tileset, const Int2 img_size, bool apply_gamma)
{
  assert(framebuffer.size() == img_size.prod());
  img.init(img_size[0], img_size[1]);
  tbb::parallel_for(0, tileset.size(), [&img, framebuffer, samples_per_tile, &tileset, apply_gamma](int i)
  {
    const auto tile = tileset[i];
    if (samples_per_tile[i] > 0)
    {
      framebuffer::ToImage(img, tile, framebuffer, samples_per_tile[i], apply_gamma);
    }
  });
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


inline double ScatterInCellProbability(double albedo, double transmittance, double l_at_cell_start, double li_distribution) noexcept
{
  // Volume Path Guiding Based on Zero-Variance Random Walk Theory
  // Eq 20
  assert (std::isfinite(albedo));
  assert (std::isfinite(transmittance));
  assert (std::isfinite(li_distribution));

  static constexpr double MIN_PROB = 0.;
  static constexpr double MAX_PROB = 1.;

  const double prob_interact = 
    (1. - transmittance)*albedo*li_distribution/(l_at_cell_start);
  // 0 / 0 -> MIN_PROB
  // inf -> MAX_PROB
  // Output clamped to [MIN_PROB, MAX_PROB] 
  return prob_interact > MIN_PROB ? (prob_interact < MAX_PROB ? prob_interact : MAX_PROB) : MIN_PROB;
}

inline std::pair<double,double> SampleTransmittanceWithinRange(double sigma_t, double r, double tnear, double tfar) noexcept
{
  // Kulla & M. Fajardo / Importance Sampling Techniques for Path Tracing in Participating Media
  // Eq (9)
  assert (tnear <= tfar);
  assert (sigma_t >= 0.);
  const double t1 = 1. - std::exp(-(tfar-tnear)*sigma_t);
  const double t2 = std::log(1. - r * t1);
  const double t3 = tnear - t2/sigma_t;
  assert (t2 <= 0.);
  // Should sigma_t be 0, we'd get t3 == nan. This means that the mean free path length is
  // much larger then tfar-tnear. Same if tfar==tnear. In this case we may assume a uniform distribution.
  // (The profile of the exponential can be assumed to be constant or flat.)
  const double t = t3 < tfar ? (t3 > tnear ? t3 : tnear) : (tnear + r * (tfar-tnear));
  // Now the pdf, with a slightly rearranged equation that is numerically stable
  const double t4 = sigma_t * std::exp((tnear - t)*sigma_t);
  const double pdf = t4/t1;
  // pdf can be 0/0==nan if the tfar-tnear << the mean free path length. 
  // In this case it may make sense to set pdf==1./(tfar-tnear)
  const double fixed_pdf = std::isfinite(pdf) ? pdf : (1./(tfar-tnear));
  return std::make_pair(t, fixed_pdf);
}


#if 0
struct RadianceFilterState
{
  // l: Fused radiance. Convex combination of value in current bin and extrapolation.
  double l;
};

inline RadianceFilterState RadianceFilterExtrapolate(
  const RadianceFilterState state,
  double inscatter,
  double bin_transmittance,
  double albedo)
{
  // Volume path guiding paper. Eq 23
  const auto l_1 = state.l;
  const double t1 = l_1 - (1.-bin_transmittance)*albedo*inscatter;
  const double t2 = t1 / bin_transmittance;
  const double current_l_2 = std::max(0., t2);
  return { current_l_2 };
}

inline RadianceFilterState RadianceFilterFuse(const RadianceFilterState &state, double l)
{
    constexpr double A = 0.75;
    const double current_l_1 = (1.-A)*state.l + A*l;
    return { current_l_1 };
}


inline double GetValue(const RadianceFilterState &state)
{
  return state.l;
}


/* Ray is the ray to shoot. It must already include the anti-self-intersection offset.
  */
std::tuple<MaybeSomeInteraction, double, Spectral3>
inline TrackToNextInteractionGuided(
  const Scene &scene,
  double initial_radiance, // incident from the direction in which the ray is shot
  const guiding::PathGuiding &pathguiding,
  const Ray &ray,
  const PathContext &context,
  Sampler &sampler,
  MediumTracker &medium_tracker)
{
  // Volume Path Guiding Based on Zero-Variance Random Walk Theory
  // Algo 1

  using RetType = std::tuple<MaybeSomeInteraction, double, Spectral3>;

  RadianceFilterState filt_l{ initial_radiance };

  double tfar = LargeNumber;
  const auto hit = scene.FirstIntersection(ray, 0., tfar);
  // The factor by which tfar is decreased is meant to prevent intersections which lie close to the end of the query segment.
  tfar *= 0.9999;
  
  double pdf = 1.;
  Spectral3 transmittance{Eigen::ones};

  auto iter = guiding::CombinedIntervalsIterator<guiding::CellIterator, SegmentIterator>{
    pathguiding.MakeCellIterator(ray, 0., tfar),
    VolumeSegmentIterator(scene, ray, medium_tracker, 0., tfar)
  };
  for (; iter; ++iter)
  {
    auto[snear, sfar] = iter.Interval();
    if (snear == sfar)
      continue;
    assert (snear < sfar);
    const Medium& medium = iter.DereferenceSecond();
    const guiding::CellData::CurrentEstimate& radiance_estimate = iter.DereferenceFirst();
    
    // For now assume constant coefficients
    const auto material_coeffs = medium.EvaluateCoeffs(ray.PointAt(0.5*(snear+sfar)), context);
    const Spectral3 bin_transmittance = medium.EvaluateTransmission(RaySegment{ray, snear, sfar}, sampler, context);
    const Spectral3 half_transmittance = bin_transmittance.sqrt(); // Because the product from the half-segments has to equal total transmittance.

    // Simplified for uniform phase function!
    // And monochromatic medium!
    // TODO: Generalize!
    // This should be the convolution of the radiance with the phase function!
    const double inscatter_radiance = radiance_estimate.incident_flux_density / UnitSphereSurfaceArea;
    const double albedo = material_coeffs.sigma_s[0]/(material_coeffs.sigma_t[0] + Epsilon);

    filt_l = RadianceFilterExtrapolate(filt_l, inscatter_radiance, half_transmittance[0], albedo);
    filt_l = RadianceFilterFuse(filt_l, guiding::FittedRadiance(radiance_estimate, ray.dir));

    const double prob_scatter_within_boundaries_guided = ScatterInCellProbability(
      albedo, 
      bin_transmittance[0],
      GetValue(filt_l),
      inscatter_radiance);

    // MIS combination of guided transmisstion probability with regular transmission based sampling.
    // TODO: generalize to chromatic media!
    const double prob_scatter_transmittance = 1. - bin_transmittance[0];
    
    const double prob_scatter_within_boundaries = 0.5*(
      prob_scatter_within_boundaries_guided + 
      prob_scatter_transmittance
    );

    assert(prob_scatter_within_boundaries >= 0.);

    if (sampler.Uniform01() < prob_scatter_within_boundaries)
    {
      assert (pdf > 0.);
      pdf *= prob_scatter_within_boundaries;
      auto [s, pdf_bin] = SampleTransmittanceWithinRange(material_coeffs.sigma_t[0], sampler.Uniform01(), snear, sfar);
      assert (pdf_bin > 0.);
      pdf *= pdf_bin;
      // The usual stuff
      const VolumeInteraction vi{ ray.PointAt(s), medium, Spectral3{ 0. }, material_coeffs.sigma_s };
      const Spectral3 transmittance_to_s = medium.EvaluateTransmission(RaySegment{ray, snear, s}, sampler, context);
      const Spectral3 weight = transmittance*transmittance_to_s / pdf;
      assert(weight.allFinite());
      assert(0 <= s && s <= tfar);
      return RetType{ vi, s, weight };
    }
    else
    {
      pdf *= 1.-prob_scatter_within_boundaries;
      transmittance *= bin_transmittance;
      filt_l = RadianceFilterExtrapolate(filt_l, inscatter_radiance, half_transmittance[0], albedo);
    }
  }

  const Spectral3 weight = transmittance / pdf;
  assert(weight.allFinite());

  // If we get here, there was no scattering in the medium.
  if (hit)
  {
    return RetType{ *hit, tfar, weight };
  }
  else
  {
    return RetType{ MaybeSomeInteraction{}, LargeNumber, weight };
  }
};
#endif


namespace
{


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

template<int N>
class MixtureDistributionWrapper
{
  const vmf_fitting::VonMisesFischerMixture<N> *mixture;
public:
  MixtureDistributionWrapper(const vmf_fitting::VonMisesFischerMixture<N>& mixture) :
    mixture{ &mixture }
  {
  }

  Double3 Sample(Sampler &sampler) const
  {
    auto r1 = sampler.Uniform01();
    auto r2 = sampler.Uniform01();
    auto r3 = sampler.Uniform01();
    // Float32 can be lower precision than I have otherwise. So renormalize the direction.
    Eigen::Vector3d w = vmf_fitting::Sample(*mixture, { r1,r2,r3 }).template cast<double>();
    const double n2 = w.squaredNorm();
    if (std::abs(n2 - 1.) >= 1.e-11)
    {
      w /= std::sqrt(n2);
    }
    return w;
  }

  double Pdf(const Double3 &dir) const
  {
    return vmf_fitting::Pdf(*mixture, dir.cast<float>());
  }
};


template<class Comp1, class Comp2>
class BiComponentDistributionWrapper
{
public:
  Comp1 comp1;
  Comp2 comp2;
  double mix_factor; // fraction of component 2

  Double3 Sample(Sampler &sampler) const
  {
    if (sampler.Uniform01() < mix_factor)
      return comp2.Sample(sampler);
    else
      return comp1.Sample(sampler);
  }

  double Pdf(const Double3 &dir) const
  {
    return Lerp(comp1.Pdf(dir), comp2.Pdf(dir), mix_factor);
  }
};

template<class Comp1, class Comp2>
BiComponentDistributionWrapper<Comp1,Comp2> MakeBiComponentDistributionWrapper(const Comp1 &comp1, const Comp2 &comp2, double f)
{
  return { comp1, comp2, f };
}


class ScatterFunctionSurface
{
  const SurfaceInteraction *surface;
  const Shader* shader;
  Double3 reverse_incident_dir;
public:
  ScatterFunctionSurface(const Shader& shader, const SurfaceInteraction& surface, const Double3& reverse_incident_dir)
    : surface{&surface}, shader{&shader}, reverse_incident_dir{reverse_incident_dir} 
    {}

  ScatterSample Sample(Sampler &sampler, const PathContext &context) const { return shader->SampleBSDF(reverse_incident_dir, *surface, sampler, context); }
  Spectral3 Evaluate(const Double3 &dir, const PathContext& context, double *pdf) const { return shader->EvaluateBSDF(reverse_incident_dir, *surface, dir, context, pdf); }
};


class ScatterFunctionVolume
{
  const VolumeInteraction *vi;
  const Double3 &reverse_incident_dir;
public:
  ScatterFunctionVolume(const VolumeInteraction &vi, const Double3& reverse_incident_dir)
    : vi{&vi}, reverse_incident_dir{reverse_incident_dir}
    {}

  ScatterSample Sample(Sampler &sampler, const PathContext &context) const { return vi->medium().SamplePhaseFunction(reverse_incident_dir, vi->pos, sampler, context); }
  Spectral3 Evaluate(const Double3 &dir, const PathContext& context, double *pdf) const { return vi->medium().EvaluatePhaseFunction(reverse_incident_dir, vi->pos, dir, context, pdf); }
};


static constexpr double PROB_BSDF = 0.5;


template<class ScatterFunction, class AlternatePdf>
ScatterSample SampleWithBinaryMixtureDensity(
  const ScatterFunction &scatter_function,
  const AlternatePdf &alternate_pdf, 
  const Double3 &reverse_incident_dir, 
  Sampler &sampler, const PathContext &context,
  const double prob_bsdf)
{
  //const auto otherDistribution = MixtureDistributionWrapper{mixture};
  //auto otherDistribution = Rotated<UniformHemisphericalDistribution>(frame, UniformHemisphericalDistribution{});

  // This is dump, I first have to sample the shader because it could generate the sample from a delta-peak.
  // Only when I know, it does not, I can deal with continuous distributions. In that case it may happen that 
  // I have to throw the sample away to sample from the alternative distribution! :-(((

  // WARNING Actually, this is even wrong, because I must multiply the alternative pdf value with the probability
  // to not-sample the delta-peak. But this value is not known by design!! :-(((((((
  ScatterSample smpl = scatter_function.Sample(sampler, context);
  assert(reverse_incident_dir.array().isFinite().all());

  if (smpl.pdf_or_pmf.IsFromDelta())
  {
    return smpl;
  }

  if (sampler.Uniform01() < prob_bsdf)
  {
    double mix_pdf = alternate_pdf.Pdf(smpl.coordinates);
    smpl.pdf_or_pmf = Lerp(
      mix_pdf,
      (double)(smpl.pdf_or_pmf),
      prob_bsdf);
    return smpl;
  }
  else
  {
    Double3 dir = alternate_pdf.Sample(sampler);
    double mix_pdf = alternate_pdf.Pdf(dir);
    double bsdf_pdf = 0.;
    Spectral3 bsdf_val = scatter_function.Evaluate(dir, context, &bsdf_pdf);
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


template<class ScatterFunction, class AlternatePdf>
Spectral3 EvaluateWithBinaryMixtureDensity(
  const ScatterFunction &scatter_function,
  const AlternatePdf &alternate_pdf,
  const Double3 &dir,
  const PathContext &context,
  double *pdf,
  double prob_bsdf
)
{
  const Spectral3 ret = scatter_function.Evaluate(dir, context, pdf);
  if (pdf)
  {
    double pdf2 = alternate_pdf.Pdf(dir);
    *pdf = Lerp(pdf2, *pdf, prob_bsdf);
  }
  return ret;
}


} // anonymous namespace


using RadianceFit = guiding::CellData::CurrentEstimate;

struct VertexCoefficients
{
  Spectral3 segment_transmission_from_prev{Eigen::zero};
  Spectral3 scatter_from_prev{Eigen::zero};
  Spectral3 nee_radiance{Eigen::zero};
  Spectral3 indirect_radiance_accumulator{Eigen::zero};
  Spectral3 emission{Eigen::zero};
  double emission_mis_factor = 1.;
  double rr_weight = 1.; // Weight multiplied to paths reaching to this node due to RR&Splitting.
};


// After computation of scattering direction.
struct PathIntermediateState
{
  MediumTracker medium_tracker;
  Ray ray; // Adjusted position with anti-self-intersection offset
  bool monochromatic;
};


struct PathNode
{
  VertexCoefficients coeffs;
  MaybeSomeInteraction interaction;
  MediumTracker medium_tracker;
  Ray incident_ray;
  Spectral3 weight; // Accumulated. Used for pixel contribution estimate and debug rendering.
  std::optional<Pdf> last_scatter_pdf_value{}; // For MIS.
  PathNode *prev = nullptr;
  const RadianceFit* radiance_fit = nullptr;
  short num = 2;
  short split_budget = 100;
  bool monochromatic = false;

  PathNode(const PathNode &) = default;
  PathNode& operator=(const PathNode &) = default;
  PathNode(PathNode &&) noexcept = default;
  PathNode& operator=(PathNode &&) noexcept = default;

  // PathNode(const Scene &scene, PathContext &context);
  // PathNode(const Scene &scene, PathContext &context, int prev_index, const PathNode &prev, const PathIntermediateState &pis, const MaybeSomeInteraction &interaction, double tfar, Spectral3 track_weight);
};

struct RGBErr {
  RGB value{Eigen::zero};
  RGB err{Eigen::zero};
};


struct Spectral3Err
{
  Spectral3 value{Eigen::zero};
  Spectral3 err{Eigen::zero};

  inline friend Spectral3Err operator*(const Spectral3 &w, const Spectral3Err &x)
  {
    return { w*x.value, w*x.err };
  }

  Spectral3Err& operator += (const Spectral3Err &x)
  {
    value += x.value;
    err += x.err;
    return *this;
  }
  Spectral3Err operator/= (int w)
  {
    value /= w;
    err /= w;
    return *this;
  }
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
  mutable Span<RGB> debugbuffer;
  Span<RGBErr> pixel_intensity_approximations;
  RayTermination ray_termination;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  static constexpr int num_lambda_sweeps = decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED;
  PathContext context;
  bool enable_nee;

public:
  CameraRenderWorker(PathTracingAlgo2 *master, int worker_index);
  void Render(const ImageTileSet::Tile & tile, const int samples_per_pixel);
  void PrepassRender(long sample_count);
  
  auto* GetGuidingLocalDataSurface() { return &radrec_local_surface;  }
  auto* GetGuidingLocalDataVolume() { return &radrec_local_volume; }

  int min_node_count = 10; // inclusive
  int max_node_count = 10; // inclusive
  int max_split      = 1;

private:
  void RenderPixel(int pixel);
  void PathTraceRecursive(PathNode &ps);

  PathNode GenerateFirstInteractionNode(const LambdaSelection &lambda_selection);
  PathIntermediateState PrepareStartAfterScatter(const PathNode &pn, const ScatterSample &smpl) const;
  PathNode GeneratePathNode(const PathNode &ps, const ScatterSample &scatter_smpl, const PathIntermediateState &after_scatter, const MaybeSomeInteraction &interaction, const Spectral3 &track_weight, double rr_weight);
  void DoTheSplittingAndRussianRoutlettePartPushingSuccessorNodes(PathNode &ps);

  ScatterSample SampleScatterKernel(const SurfaceInteraction &interaction, const PathNode &pn, const Double3 &reverse_incident_dir) const;
  ScatterSample SampleScatterKernel(const VolumeInteraction &interaction, const PathNode &pn, const Double3 &reverse_incident_dir) const;

  std::pair<Spectral3, double> EvaluateScatterKernel(const Double3 &incident_dir, const SurfaceInteraction &interaction, const PathNode &pn, const Double3 &outgoing_dir) const;
  std::pair<Spectral3, double> EvaluateScatterKernel(const Double3 &incident_dir, const VolumeInteraction &interaction, const PathNode &pn, const Double3 &outgoing_dir) const;

  std::pair<int, double> ComputeNumberOfDistanceSamples(const PathNode &pn, const ScatterSample &smpl) const;
  std::pair<int, double> ComputeNumberOfDirectionSamples(const PathNode &pn) const;
  std::pair<int, double> ComputeNumberOfSplits(const PathNode &ps, const Spectral3Err &contribution_estimate) const;
  bool CanUseIlluminationApproximationInRRAndSplit(const PathNode &ps) const;

  void AddEmission(const PathNode &ps, VertexCoefficients &coeffs) const;
  void AddEmission(const PathNode &ps, const SurfaceInteraction &si, VertexCoefficients &coeffs) const;
  void AddEmission(const PathNode &ps, const VolumeInteraction &vi, VertexCoefficients &coeffs) const;
  void AddEnvEmission(const PathNode &ps, VertexCoefficients &coeffs) const;

  void MaybeAddNeeLighting(const PathNode &ps, double weight, VertexCoefficients &coeffs) const;
  
  void RecordMeasurementToCurrentPixel(const PathNode &root) const;

  void AddToTrainingData(const PathNode &ps, const Spectral3 &radiance, const Double3 &dir, const Pdf pdf);
  void PropagateIlluminationToParents(PathNode &pn);

  const RadianceFit* FindRadianceEstimate(const SomeInteraction &interaction) const;
  const Spectral3Err ComputeApproximateInscatter(const PathNode &pn, const VolumeInteraction &vi) const;
  const Spectral3Err ComputeApproximateInscatter(const PathNode &pn, const SurfaceInteraction &si) const;

  void RecordMeasurementToDebugBuffer(const Spectral3 estimate) const;
};



class PathTracingAlgo2 : public RenderingAlgo
{
  friend class CameraRenderWorker;
  friend class ApproximatePixelWorker;
private:
  ToyVector<RGB> framebuffer;
  ToyVector<RGB> debugbuffer;
  ToyVector<RGBErr> pixel_intensity_approximations;
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

  void InitializeScene(Scene &scene) override;
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


class alignas(128) ApproximatePixelWorker
{
  PathTracingAlgo2 const * master;
  guiding::PathGuiding const * radiance_recorder_surface;
  guiding::PathGuiding const * radiance_recorder_volume;
  LightPickerUcbBufferedQueue* const pickers;
  Sampler sampler;
  Span<RGBErr> framebuffer;
  LambdaSelectionStrategyShuffling lambda_selection_factory;
  static constexpr int num_lambda_sweeps = decltype(lambda_selection_factory)::NUM_SAMPLES_REQUIRED;
  ToyVector<BoundaryIntersection> boundary_intersection_buffer;

public:
  ApproximatePixelWorker(PathTracingAlgo2* master, Span<RGBErr> framebuffer)
    : master{ master },
    radiance_recorder_surface{master->radiance_recorder_surface.get()},
    radiance_recorder_volume{master->radiance_recorder_volume.get()},
    pickers{master->pickers.get()},
    framebuffer{framebuffer}
  {
    boundary_intersection_buffer.reserve(1024);
  }

  //OnlineVariance::Accumulator<Eigen::Array3d,long> average_intensity{Eigen::Array3d{Eigen::zero}};

  void Render(const ImageTileSet::Tile &tile)
  {
    MediumTracker medium_tracker{master->scene};

    const Int2 end = tile.corner + tile.shape;
    for (int iy = tile.corner[1]; iy < end[1]; ++iy) 
    for (int ix = tile.corner[0]; ix < end[0]; ++ix)
    for (int lambda_sweep = 0; lambda_sweep < lambda_selection_factory.NUM_SAMPLES_REQUIRED; ++lambda_sweep)
    {
      auto& camera = master->scene.GetCamera();
      auto pixel = camera.PixelToUnit({ix, iy});

      const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
      PathContext context{lambda_selection, pixel};

      auto pos = camera.TakePositionSample(pixel, sampler, context);
      auto dir = camera.TakeDirectionSampleFrom(pixel, pos.coordinates, sampler, context);

      medium_tracker.initializePosition(pos.coordinates);

      Spectral3Err measurement_estimator;

      Spectral3 weight = pos.value * dir.value * lambda_selection.weights / (Value(pos.pdf_or_pmf) * Value(dir.pdf_or_pmf));
      auto ray = Ray{ pos.coordinates, dir.coordinates };
      
      bool keep_going = true;
      for (int ray_depth = 0; ray_depth < master->render_params.max_ray_depth && keep_going; ++ray_depth)
      {
        //auto[interaction, tfar, track_weight] = TrackToNextInteraction(master->scene, ray, context, Spectral3::Ones(), sampler, medium_tracker, nullptr);
        auto [interaction, tfar, track_weight, inscatter] = RayMarchAccumulatingInscatter(
          /*master->scene, *radiance_recorder_volume,*/ ray, context, sampler, medium_tracker);
        
        measurement_estimator += weight * inscatter;

        if (!interaction)
          keep_going = false;
        else if (auto *si = mpark::get_if<SurfaceInteraction>(&*interaction))
        {
          if (ray_depth == 0) // If camera ray hits an emissive surface ...
          {
            if (const auto *emitter = GetMaterialOf(*si, master->scene).emitter; emitter)
            {
              Spectral3 radiance = emitter->Evaluate(si->hitid, -ray.dir, context, nullptr);
              measurement_estimator.value += weight * track_weight * radiance;
            }
          }

          const Shader& shader = GetShaderOf(*si, master->scene);
          auto bsdf_sample = shader.SampleBSDF(-ray.dir, *si, sampler, context);
          if (bsdf_sample.pdf_or_pmf.IsFromDelta()) // || shader.prefer_path_tracing_over_photonmap)
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

            const Spectral3Err indirect_lighting = ComputeIndirectLighting(*si, -ray.dir, sampler, context);
            measurement_estimator += (weight * track_weight).eval() * indirect_lighting;

            keep_going = false;
          }
        }
        // else if (auto *vi = mpark::get_if<VolumeInteraction>(&*interaction))
        // {
        //   // track weight ~= 1/sigma_t * 1/p_lambda = 100 * 12
        //   const Spectral3Err radiance_estimate = ComputeInscatteredRadiance(*vi, ray.dir, sampler, context);
        //   // if (ix == 180 && iy == 180)
        //   //   std::cout << radiance_estimate << " " << vi->sigma_s << " " << (weight * track_weight) << std::endl;
        //   measurement_estimator += weight * track_weight * vi->sigma_s * radiance_estimate;
        //   keep_going = false;
        // }
      }
      measurement_estimator /= lambda_selection_factory.NUM_SAMPLES_REQUIRED;
      RecordMeasurementToCurrentPixel(measurement_estimator, pixel, lambda_selection);
    }
  }


  std::tuple<MaybeSomeInteraction, double, Spectral3, Spectral3Err>
  inline RayMarchAccumulatingInscatter(
    const Ray &ray,
    const PathContext &context,
    Sampler &sampler,
    MediumTracker &medium_tracker)
  {
    // Volume Path Guiding Based on Zero-Variance Random Walk Theory
    // Algo 1

    double tfar = LargeNumber;
    const auto hit = master->scene.FirstIntersection(ray, 0., tfar);
    // The factor by which tfar is decreased is meant to prevent intersections which lie close to the end of the query segment.
    tfar *= 0.9999;
    
    Spectral3 transmittance{Eigen::ones};
    Spectral3Err integrated_inscatter;

    Span<BoundaryIntersection> boundary_intersections = master->scene.IntersectionsWithVolumes(ray, 0., tfar);
    
    // boundary_intersection_buffer.clear();
    // std::copy(boundary_intersections.begin(), boundary_intersections.end(), std::back_inserter(boundary_intersection_buffer));
    // boundary_intersections = AsSpan(boundary_intersection_buffer);

    auto iter = guiding::CombinedIntervalsIterator<guiding::CellIterator, SegmentIterator>{
      radiance_recorder_volume->MakeCellIterator(ray, 0., tfar),
      SegmentIterator { ray, medium_tracker, boundary_intersections, 0., tfar }
    };
    for (; iter; ++iter)
    {
      auto[snear, sfar] = iter.Interval();
      if (snear == sfar)
        continue;
      assert (snear < sfar);
      const Medium& medium = iter.DereferenceSecond();
      
      // For now assume constant coefficients
      const auto material_coeffs = medium.EvaluateCoeffs(ray.PointAt(0.5*(snear+sfar)), context);
      const Spectral3 bin_transmittance = medium.EvaluateTransmission(RaySegment{ray, snear, sfar}, sampler, context);
      
      //const double s = Lerp(snear, sfar, 0.5); //sampler.Uniform01());
      //pdf *= 1./(sfar - snear);
      auto [s, pdf_bin] = SampleTransmittanceWithinRange(material_coeffs.sigma_t[0], sampler.Uniform01(), snear, sfar);
      assert (pdf_bin > 0.);

      const Spectral3 transmittance_to_s = medium.EvaluateTransmission(RaySegment{ray, snear, s}, sampler, context);
      const Spectral3 weight = transmittance*transmittance_to_s / pdf_bin;
      
      const Spectral3Err inscatter = ComputeInscatteredRadiance(iter.DereferenceFirst());
      integrated_inscatter += (weight * material_coeffs.sigma_s).eval() * inscatter;

      assert(weight.allFinite());
      assert(0 <= s && s <= tfar);

      transmittance *= bin_transmittance;
    }

    // If we get here, there was no scattering in the medium.
    if (hit)
    {
      return { *hit, tfar, transmittance, integrated_inscatter };
    }
    else
    {
      return { {}, LargeNumber, transmittance, integrated_inscatter };
    }
  }


  Spectral3Err ComputeIndirectLighting(const SurfaceInteraction &interaction, const Double3 &reverse_incident_dir, Sampler &sampler, const PathContext &context)
  {
    const auto& estimate = radiance_recorder_surface->FindRadianceEstimate(interaction.pos);
    const auto& distribution = estimate.radiance_distribution;
    const auto& shader = GetShaderOf(interaction, master->scene);

    Spectral3Err result;
    const int sample_count = 2;

    if (shader.supports_lobes)
    {
      const auto lobes = shader.ComputeLobes(reverse_incident_dir, interaction, context);
      auto product = vmf_fitting::Product(lobes, estimate.radiance_distribution);
      double tmp = product.weights.sum();
      result.value = Spectral3::Constant(tmp * estimate.incident_flux_density);
      result.err = Spectral3::Constant(tmp * estimate.incident_flux_confidence_bounds);
    }
    else
    {
      // Using Monte Carlo
      for (int i = 0; i<sample_count; ++i)
      {
        auto s = SampleWithBinaryMixtureDensity(
          ScatterFunctionSurface{shader, interaction, reverse_incident_dir},
          MixtureDistributionWrapper<8>(estimate.radiance_distribution),
          reverse_incident_dir, sampler, context,
          0.5);
        s.value *= DFactorPBRT(interaction, s.coordinates) / (double)s.pdf_or_pmf;
        auto [val, err] = guiding::FittedRadianceWithErr(estimate, s.coordinates);
        result += s.value * Spectral3Err{Spectral3::Constant(val),Spectral3::Constant(err)};
      }
      result /= sample_count;
    }
    

    // for (int i = 0; i<sample_count; ++i)
    // {
      // {
      //   const Eigen::Vector3f dir = vmf_fitting::Sample(distribution, { sampler.Uniform01(), sampler.Uniform01(), sampler.Uniform01() });
      //   // Drops out because Li = pdf * estimate.incident_flux_density. Dividing by pdf because Monte Carlo,
      //   // makes pdf drop out of the equation.
      //   //const float pdf = vmf_fitting::Pdf(distribution, dir);
      //   const Spectral3 bsdf_weight = shader.EvaluateBSDF(-incident_dir, interaction, dir.cast<double>(), context, nullptr);
      //   result += bsdf_weight /*/ (double)pdf*/ * DFactorPBRT(interaction, dir.cast<double>()) * estimate.incident_flux_density;
      // }
      // {
      //   // Sample the bsdf, too!
      //   const auto smpl = shader.SampleBSDF(-incident_dir, interaction, sampler, context);
      //   const float radiance_fit = guiding::FittedRadiance(estimate, smpl.coordinates);
      //   result += smpl.value * radiance_fit * DFactorPBRT(interaction, smpl.coordinates) / ((double)smpl.pdf_or_pmf);
      // }
    // }
    // Bad approximation because the estimate incident flux is averaged over wavelengths!
    // result  /= (2*sample_count);
    return result;
  }


  Spectral3Err ComputeInscatteredRadiance(const guiding::CellData::CurrentEstimate &radiance_fit) const
  {
    // TODO: Abolish assumption of uniform phase function
    Spectral3 result = Spectral3::Constant(radiance_fit.incident_flux_density / UnitSphereSurfaceArea);
    Spectral3 err = Spectral3::Constant(radiance_fit.incident_flux_confidence_bounds / UnitSphereSurfaceArea);
    return { result, err };
  }


  Spectral3Err ComputeInscatteredRadiance(const VolumeInteraction &interaction, const Double3& /*rev_incident_dir*/, Sampler& /*sampler*/, const PathContext& /*context*/)
  {
    const auto& estimate = radiance_recorder_volume->FindRadianceEstimate(interaction.pos);
    return ComputeInscatteredRadiance(estimate);
  }


  void RecordMeasurementToCurrentPixel(const Spectral3Err &measurement, int pixel_index, const LambdaSelection &lambda_selection)
  {
    assert(measurement.value.isFinite().all());
    auto color = Color::SpectralSelectionToRGB(measurement.value, lambda_selection.indices);
    auto color_err = Color::SpectralSelectionToRGB(measurement.err, lambda_selection.indices);
    framebuffer[pixel_index].value += color;
    framebuffer[pixel_index].err += color_err;
    //average_intensity += Eigen::Array3d{value(color[0]),value(color[1]),value(color[2])};
  }

}; // class ApproximatePixelWorker


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

  framebuffer.resize(num_pixels, RGB{Eigen::zero});
  debugbuffer.resize(num_pixels, RGB{Eigen::zero});
  samplesPerTile.resize(tileset.size(), 0);

  pixel_intensity_approximations.resize(num_pixels, RGBErr{});

  pickers = std::make_unique<LightPickerUcbBufferedQueue>(scene, NumThreads());

  radiance_recorder_surface = std::make_unique<guiding::PathGuiding>(scene.GetBoundingBox(), 0.1, render_params_, the_task_arena, "surface");
  radiance_recorder_volume = std::make_unique<guiding::PathGuiding>(scene.GetBoundingBox(), 0.1, render_params_, the_task_arena, "volume");

  for (int i = 0; i < the_task_arena.max_concurrency(); ++i)
  {
    camerarender_workers.emplace_back(this, i);
  }
}


void PathTracingAlgo2::InitializeScene(Scene &scene)
{
  for (Scene::index_t i=0; i<scene.GetNumShaders(); ++i)
  {
    auto& shd = scene.GetShader(i);
    shd.IntializeLobes();
  }
}


inline void PathTracingAlgo2::Run()
{
  #ifdef WRITE_DEBUG_OUT
  scene.WriteObj(guiding::GetDebugFilePrefix() / boost::filesystem::path("scene.obj"));
  #endif

  RenderRadianceEstimates(guiding::GetDebugFilePrefix() / fs::path{"initial_approx.png"});

  auto radrec_local_surface = TransformVector(camerarender_workers, [](auto &w) { return w.GetGuidingLocalDataSurface();  });
  auto radrec_local_volume = TransformVector(camerarender_workers, [](auto &w) { return w.GetGuidingLocalDataVolume();  });

  {
    long num_samples = 1;
    while (!stop_flag.load() && num_samples <= render_params.guiding_max_spp)
    {
      std::cout << "Guiding sweep " << num_samples << " start" << std::endl;
      // Clear the frame buffer to get rid of samples from previous iterations
      // which are assumed to be worse than current samples.
      std::fill(framebuffer.begin(), framebuffer.end(), RGB::Zero());
      std::fill(debugbuffer.begin(), debugbuffer.end(), RGB::Zero());
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
      radiance_recorder_surface->PrepareAdaptedStructures();
      radiance_recorder_volume->FinalizeRound(AsSpan(radrec_local_volume));
      radiance_recorder_volume->PrepareAdaptedStructures();

      std::cout << "Guiding sweep " << num_samples << " finished" << std::endl;
      CallInterruptCb(true);

      RenderRadianceEstimates(guiding::GetDebugFilePrefix() / fs::path{strconcat("trainpass_approx",num_samples,".png")});

      the_task_arena.execute([this,num_samples] 
      {  
        Image img;
        ::GenerateImage(img, AsSpan(framebuffer), AsSpan(samplesPerTile), tileset, {render_params.width, render_params.height}, !render_params.linear_output);
        img.write((guiding::GetDebugFilePrefix() / fs::path{strconcat("trainpass_backimg_",num_samples,".png")}).string());
        // img.clear();
        // ::GenerateImage(img, AsSpan(debugbuffer), AsSpan(samplesPerTile), tileset, {render_params.width, render_params.height}, !render_params.linear_output);
        // img.write((guiding::GetDebugFilePrefix() / fs::path{strconcat("trainpass_fwdimg_",num_samples,".png")}).string());
      });

      //num_samples = num_samples + std::max(1, int(num_samples*0.5));
      num_samples *= 2;
    } // Pass iteration
  }

  // TODO: Fix RR weights messing up training samples??!!!
  for (auto &w : camerarender_workers)
  {
    w.max_split = 10;
    w.min_node_count = 2;
    w.max_node_count = 20;
  }
  this->record_samples_for_guiding = false;

  std::fill(framebuffer.begin(), framebuffer.end(), RGB::Zero());
  std::fill(debugbuffer.begin(), debugbuffer.end(), RGB::Zero());
  std::fill(samplesPerTile.begin(), samplesPerTile.end(), 0ul);

  while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
  {
    std::cout << "Sweep " << spp_schedule.GetTotal() << " start" << std::endl;

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

    the_task_arena.execute([this,num_samples=spp_schedule.GetTotal()] 
    {  
      Image img;
      ::GenerateImage(img, AsSpan(framebuffer), AsSpan(samplesPerTile), tileset, {render_params.width, render_params.height}, !render_params.linear_output);
      img.write((guiding::GetDebugFilePrefix() / fs::path{strconcat("mainpass_backimg_",num_samples,".png")}).string());
      // img.clear();
      // ::GenerateImage(img, AsSpan(debugbuffer), AsSpan(samplesPerTile), tileset, {render_params.width, render_params.height}, !render_params.linear_output);
      // img.write((guiding::GetDebugFilePrefix() / fs::path{strconcat("mainpass_fwdimg_",num_samples,".png")}).string());
    });

    spp_schedule.UpdateForNextPass();
  } // Pass iteration
}


inline std::unique_ptr<Image> PathTracingAlgo2::GenerateImage()
{
  auto bm = std::make_unique<Image>();
  the_task_arena.execute([this, bm = bm.get()] 
  {  
    ::GenerateImage(*bm, AsSpan(framebuffer), AsSpan(samplesPerTile), tileset, {render_params.width, render_params.height}, !render_params.linear_output);
  });
  return bm;
}



void PathTracingAlgo2::RenderRadianceEstimates(fs::path filename)
{
  std::fill(pixel_intensity_approximations.begin(),pixel_intensity_approximations.end(), RGBErr{});

  the_task_arena.execute([this, &filename] ()
  {
    std::cout << "Rendering Radiance Estimates " << filename << std::endl;

    ToyVector<ApproximatePixelWorker> workers; workers.reserve(the_task_arena.max_concurrency());
    for (int i = 0; i<the_task_arena.max_concurrency(); ++i)
      workers.emplace_back(this, AsSpan(pixel_intensity_approximations));

    tbb::parallel_for(0, this->tileset.size(), 1, [this, &workers](int i)
    {
      const int worker_num = tbb::this_task_arena::current_thread_index();
      workers[worker_num].Render(this->tileset[i]);
    });

    auto bm = Image(render_params.width, render_params.height);

    ToyVector<RGB> rgb_vals(pixel_intensity_approximations.size());
    ToyVector<RGB> rgb_errs(pixel_intensity_approximations.size());
    for (int i=0; i<isize(pixel_intensity_approximations); ++i)
    {
      rgb_vals[i] = pixel_intensity_approximations[i].value;
      rgb_errs[i] = (pixel_intensity_approximations[i].err).min(1._rgb).max(0._rgb);
    }

    tbb::parallel_for(0, tileset.size(), [this, &bm, fb=AsSpan(rgb_vals)](int i) {
      const auto tile = tileset[i];
      framebuffer::ToImage(bm, tile, fb, 1, !render_params.linear_output);
    });

    bm.write(filename.string());

    tbb::parallel_for(0, tileset.size(), [this, &bm, fb=AsSpan(rgb_errs)](int i) {
      const auto tile = tileset[i];
      framebuffer::ToImage(bm, tile, fb, 1, !render_params.linear_output);
    });

    bm.write(filename.replace_extension("_err.png").string());

    //Eigen::Array3d average_intensity = ranges::accumulate((workers | ranges::views::transform([](auto &w) { w.average_intensity.Mean(); })), Eigen::Array3d{Eigen::zero});
    // Eigen::Array3d average_intensity{Eigen::zero};
    // for (auto &w: workers)
    //   average_intensity += w.average_intensity.Mean();
    // average_intensity /= workers.size();

    // std::cout << "Done. Avg intensity = " << average_intensity << std::endl;
  });
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
  debugbuffer = AsSpan(master->debugbuffer);
  pixel_intensity_approximations = AsSpan(master->pixel_intensity_approximations);
  if (master->render_params.pt_sample_mode == "bsdf")
  {
    enable_nee = false;
  }
}


void CameraRenderWorker::Render(const ImageTileSet::Tile &tile, const int samples_per_pixel)
{
  const Int2 end = tile.corner + tile.shape;

  for (int iy = tile.corner[1]; iy < end[1]; ++iy)
  for (int ix = tile.corner[0]; ix < end[0]; ++ix)
  {
    const auto pixel = master->scene.GetCamera().PixelToUnit({ ix, iy});      
    for (int i = 0; i < samples_per_pixel; ++i)
    {
      RenderPixel(pixel);
    }
  }
}


void CameraRenderWorker::RenderPixel(int pixel)
{
  const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
  context = PathContext(lambda_selection, pixel);

  PathNode root = GenerateFirstInteractionNode(lambda_selection);

#if 1
  PathTraceRecursive(root);
#else
  if (root.interaction)
  {
    MaybeAddNeeLighting(root, 1., root.coeffs);
    auto smpl = mpark::visit([this, &root](auto &&ia) {
      return SampleScatterKernel(ia, root, -root.incident_ray.dir);
    }, *root.interaction);
    PathIntermediateState after_scatter = PrepareStartAfterScatter(root, smpl);
    auto[interaction, tfar, track_weight] = TrackToNextInteraction(
      master->scene, 
      after_scatter.ray, 
      context, 
      Spectral3::Ones(), 
      sampler, 
      after_scatter.medium_tracker, 
      nullptr);
    if (interaction)
    {
      PathNode successor = GeneratePathNode(root, smpl, after_scatter, interaction, track_weight);
      MaybeAddNeeLighting(successor, 1., successor.coeffs);
      PropagateIlluminationToParents(successor);
    }
  }
#endif

  RecordMeasurementToCurrentPixel(root);
}


void CameraRenderWorker::PrepassRender(long sample_count)
{
  while (sample_count-- > 0)
  {
    const auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    int ix = sampler.UniformInt(0, master->render_params.width-1);
    int iy = sampler.UniformInt(0, master->render_params.height-1);
    const auto pixel = master->scene.GetCamera().PixelToUnit({ ix, iy});      
    context = PathContext(lambda_selection, pixel);

    PathNode root = GenerateFirstInteractionNode(lambda_selection);

    PathTraceRecursive(root);

    RecordMeasurementToCurrentPixel(root);
  }
}


void CameraRenderWorker::PathTraceRecursive(PathNode &ps)
{
  if (ps.interaction)
  {
    DoTheSplittingAndRussianRoutlettePartPushingSuccessorNodes(ps);
    AddEmission(ps, ps.coeffs);
  }
  else // if not interaction
  {
    AddEnvEmission(ps, ps.coeffs);
  }
  if (ps.prev)
    PropagateIlluminationToParents(ps);
}


void CameraRenderWorker::DoTheSplittingAndRussianRoutlettePartPushingSuccessorNodes(PathNode &ps)
{
  const guiding::PathGuiding* guiding_records = mpark::visit(Overload(
    [this](const SurfaceInteraction&) { return this->radiance_recorder_surface; },
    [this](const VolumeInteraction&) { return this->radiance_recorder_volume; }
  ), *ps.interaction);

  // TODO: how about specular surfaces???!!!!!
  auto [successors_dir, rr_dir_weight] = ComputeNumberOfDirectionSamples(ps);
  for (int i=0; i<successors_dir; ++i)
  {
    auto smpl = mpark::visit([this, &ps](auto &&ia) {
      return SampleScatterKernel(ia, ps, -ps.incident_ray.dir);
    }, *ps.interaction);

    auto [successors_dist, rr_dist_weight] = ComputeNumberOfDistanceSamples(ps, smpl);

    if (successors_dist)
    {
      for (int j=0; j<successors_dist; ++j)
      {
          PathIntermediateState after_scatter = PrepareStartAfterScatter(ps, smpl);

          // TODO: fix the nonsensical use of initial weight in Atmosphere Material!!
          // auto[interaction, tfar, track_weight] = TrackToNextInteractionGuided(
          //   master->scene,
          //   guiding::FittedRadiance(*ps.radiance_fit, after_scatter.ray.dir),
          //   *guiding_records,
          //   after_scatter.ray, 
          //   context,
          //   sampler, 
          //   after_scatter.medium_tracker);

          auto[interaction, tfar, track_weight] = TrackToNextInteraction(
            master->scene,
            after_scatter.ray,
            context,
            Spectral3{Eigen::ones},
            sampler, 
            after_scatter.medium_tracker,
            nullptr);

          PathNode successor = GeneratePathNode(ps, smpl, after_scatter, interaction, track_weight, rr_dist_weight*rr_dir_weight);
          successor.split_budget /= (successors_dir*successors_dist);
          
          PathTraceRecursive(successor);

          //MaybeAddNeeLighting(ps, rr_dir_weight*rr_dist_weight, ps.coeffs);
      }
    }
  }

  MaybeAddNeeLighting(ps, 1., ps.coeffs);
}

namespace 
{

std::pair<int, double> ComputeNumberOfSplits2(double value, double value_err, double reference, double reference_err, double r, int max_splits, bool must_continue)
{
#if 1
  // Weight window based on "Adjoint-Driven Russian Roulette and Splitting in Light Transport Simulation"
  // Additionally considering error bars in the inputs.
  
  /* Original formulae, which centers the window around the reference:
  const double ww_s = 5.; // See Sec 5.1
  const double ww_low_tmp = (2.*reference)/(1.+ww_s);
  const double ww_high = ww_s * ww_low_tmp;
  */

  const double ww_s = 5.;
  const double ww_low_tmp = (2.*reference)/(1.+ww_s);
  const double ww_high_tmp = ww_s * ww_low_tmp;
  const double ww_low = must_continue ? 0. :  std::max(0., ww_low_tmp - 2.*value_err - 2.*reference_err);
  const double ww_high = ww_high_tmp + 2.*value_err + 2.*reference_err;

  if (value < ww_low)
  {
    // Survival probability is value/ww_low = q
    if (r * ww_low < value)
    {
      return { 1, ww_low/value };
    }
    else
    {
      return { 0, 1. };
    }
  }
  else if ((value > ww_high) & (ww_high > 0.))
  {
    const double q = std::min(value/ww_high, static_cast<double>(max_splits));
    int n = static_cast<int>(q);
    double prob = q - n;
    if (r < prob)
      ++n;
    return {n, 1./q};
  }
  else
    return {1, 1.};
#else
  return { 1, 1.};
#endif
}

} // anom namespace


bool CameraRenderWorker::CanUseIlluminationApproximationInRRAndSplit(const PathNode &ps) const
{
  return mpark::visit(Overload(
    [scene = &master->scene](const SurfaceInteraction &ia) -> bool {
      return GetShaderOf(ia, *scene).is_pure_diffuse;
    },
    [](const VolumeInteraction &) -> bool{
      return true;
    }
  ), *ps.interaction);
}



std::pair<int, double> CameraRenderWorker::ComputeNumberOfSplits(const PathNode &ps, const Spectral3Err &contribution_estimate) const
{
  const int max_split = (ps.split_budget <= 1) ? 1 : this->max_split;
  const double r = sampler.Uniform01();
  const auto reference = pixel_intensity_approximations[context.pixel_index].value.mean();
  const auto reference_err = pixel_intensity_approximations[context.pixel_index].err.mean();
  const auto value = Color::SpectralSelectionToRGB(contribution_estimate.value, context.lambda_idx).mean();
  const auto err = Color::SpectralSelectionToRGB(contribution_estimate.err, context.lambda_idx).mean();
  return ComputeNumberOfSplits2((double)value, (double)err, (double)reference, (double)reference_err, r, max_split, /*must_continue=*/ps.num < min_node_count);
}


std::pair<int, double> CameraRenderWorker::ComputeNumberOfDirectionSamples(const PathNode &ps) const
{
  assert (ps.interaction);
  if (ps.num >= max_node_count) 
    return { 0, 1. };
  if (!CanUseIlluminationApproximationInRRAndSplit(ps))
    return { 1, 1. };
#if 1
  // How this works:
  //  * Grab L distribution from cache
  //  * Convolve with phase function to get Li (assume uniform phase function for now. TODO: BSDF and proper PF support)
  //  * Predict path contribution
  //  * Grab pixel estimate
  //  * Compare
  const Spectral3Err li_approximation = mpark::visit([&](const auto &ia) { return ComputeApproximateInscatter(ps, ia); }, *ps.interaction);
  const Spectral3Err path_contribution = ps.weight * li_approximation; 
  return ComputeNumberOfSplits(ps, path_contribution);
#elif 0 // RR based on throughput
  return ComputeNumberOfSplits2(ps.weight.mean(), 1., sampler.Uniform01(), max_split, ps.num < min_node_count);
#else
  return { 1, 1. };
#endif
}


std::pair<int, double> CameraRenderWorker::ComputeNumberOfDistanceSamples(const PathNode &ps, const ScatterSample &direction_sample) const
{
  assert (ps.interaction);
  if (ps.num >= max_node_count) 
    return { 0, 1. };
  if (!CanUseIlluminationApproximationInRRAndSplit(ps))
    return { 1, 1. };
#if 1
  // And this one:
  //  * Grab L distribution from the cache
  //  * Compute approximate path contribution 
  //  * Grab pixel estimate
  //  * Compare

  auto [val, err] = guiding::FittedRadianceWithErr(*ps.radiance_fit, direction_sample.coordinates);
  const Spectral3Err l_approximation{Spectral3::Constant(val), Spectral3::Constant(err)};
  const Spectral3Err path_contribution = (ps.weight * direction_sample.value).eval() * l_approximation;
  return ComputeNumberOfSplits(ps, path_contribution);
#elif 1 // RR based on throughput
  const Spectral3 throughput = ps.weight * direction_sample.value;
  return ComputeNumberOfSplits2(throughput.mean(), 1., sampler.Uniform01(), max_split, ps.num < min_node_count);
#else
  return { 1, 1. };
#endif
}


const Spectral3Err CameraRenderWorker::ComputeApproximateInscatter(const PathNode &pn, const SurfaceInteraction &si) const
{
  assert (pn.radiance_fit);
  const Shader& shd = GetShaderOf(si, master->scene);
  if (shd.supports_lobes)
  {
    const auto lobes = shd.ComputeLobes(-pn.incident_ray.dir, si, context);
    auto product = vmf_fitting::Product(lobes, pn.radiance_fit->radiance_distribution);
    double integral = product.weights.sum();
    Spectral3 val = Spectral3::Constant(integral * pn.radiance_fit->incident_flux_density);
    Spectral3 err = Spectral3::Constant(integral * pn.radiance_fit->incident_flux_confidence_bounds);
    return {val, err};
  }
  else
  {
    // Out of pure desparation I assume a lambertian surface with 0.5 albedo, also ignoring the d-factor.
    // TODO: Here it would be very nice to be able to convolve the bsdf with the light distributions ...
    Spectral3 val = Spectral3::Constant(0.5)* (pn.radiance_fit->incident_flux_density / UnitSphereSurfaceArea);
    Spectral3 err = Spectral3::Constant(0.5)* (pn.radiance_fit->incident_flux_confidence_bounds / UnitSphereSurfaceArea);
    return {val, err};
  }
}


const Spectral3Err CameraRenderWorker::ComputeApproximateInscatter(const PathNode &pn, const VolumeInteraction &vi) const
{
  assert (pn.radiance_fit);
  // TODO: Convolution with the real phase function
  Spectral3 val = vi.sigma_s*(pn.radiance_fit->incident_flux_density / UnitSphereSurfaceArea);
  Spectral3 err = vi.sigma_s*(pn.radiance_fit->incident_flux_confidence_bounds / UnitSphereSurfaceArea);
  return { val, err };
}


PathNode CameraRenderWorker::GenerateFirstInteractionNode(const LambdaSelection &lambda_selection)
{
  const auto& camera = master->scene.GetCamera();
  
  Spectral3 weight = lambda_selection.weights;

  auto pos = camera.TakePositionSample(context.pixel_index, sampler, context);
  weight *= pos.value / pos.pdf_or_pmf;
  auto dir = camera.TakeDirectionSampleFrom(context.pixel_index, pos.coordinates, sampler, context);
  weight *= dir.value / dir.pdf_or_pmf;
  Ray ray{ pos.coordinates, dir.coordinates };

  MediumTracker medium_tracker{master->scene};
  medium_tracker.initializePosition(pos.coordinates);

  auto[interaction, tfar, track_weight] = TrackToNextInteraction(master->scene, ray, context, Spectral3::Ones(), sampler, medium_tracker, nullptr);

  const RadianceFit* radiance_fit = interaction ? FindRadianceEstimate(*interaction) : nullptr;

  VertexCoefficients coeffs{};
  coeffs.segment_transmission_from_prev = track_weight;
  coeffs.scatter_from_prev = weight;

  return PathNode{
    coeffs,
    interaction,
    medium_tracker,
    ray,
    weight * track_weight,
    {},
    nullptr,
    radiance_fit,
  };
}


const RadianceFit* CameraRenderWorker::FindRadianceEstimate(const SomeInteraction &interaction) const
{
    return mpark::visit(Overload(
      [&](const SurfaceInteraction &ia) { return &radiance_recorder_surface->FindRadianceEstimate(ia.pos); },
      [&](const VolumeInteraction& ia) { return &radiance_recorder_volume->FindRadianceEstimate(ia.pos); }
    ), interaction);
}


PathNode CameraRenderWorker::GeneratePathNode(
  const PathNode &ps, 
  const ScatterSample &scatter_smpl, 
  const PathIntermediateState &after_scatter, 
  const MaybeSomeInteraction &interaction, 
  const Spectral3 &track_weight, 
  double rr_weight)
{
  const RadianceFit* radiance_fit = interaction ? FindRadianceEstimate(*interaction) : nullptr;

  VertexCoefficients coeffs{};
  coeffs.scatter_from_prev = scatter_smpl.value;
  coeffs.segment_transmission_from_prev = track_weight;
  coeffs.rr_weight = rr_weight;

  const Spectral3 weight = ps.weight * scatter_smpl.value * track_weight * rr_weight;

  return {
    coeffs,
    interaction,
    after_scatter.medium_tracker,
    after_scatter.ray,
    weight,
    scatter_smpl.pdf_or_pmf,
    const_cast<PathNode*>(&ps),
    radiance_fit,
    static_cast<short>(ps.num + 1),
    ps.split_budget,
    ps.monochromatic || after_scatter.monochromatic
  };
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


std::pair<Spectral3, double> CameraRenderWorker::EvaluateScatterKernel(const Double3 &incident_dir, const SurfaceInteraction &interaction, const PathNode &pn, const Double3 &outgoing_dir) const
{
  const auto& mixture = pn.radiance_fit->radiance_distribution;
  const Shader& shd = GetShaderOf(interaction, master->scene);
  double bsdf_pdf = 0.;
  if (shd.supports_lobes)
  {
    const auto lobes = shd.ComputeLobes(incident_dir, interaction, context);
    auto product = vmf_fitting::Product(lobes, mixture);
    vmf_fitting::Normalize(product);

    Spectral3 scatter_kernel = EvaluateWithBinaryMixtureDensity(
      ScatterFunctionSurface{shd, interaction, incident_dir},
      MakeBiComponentDistributionWrapper (
        MixtureDistributionWrapper<16>(product),
        MixtureDistributionWrapper<8>(mixture),
        0.5
      ),
      outgoing_dir,
      context,
      &bsdf_pdf,
      0.33
    );
    scatter_kernel *= DFactorPBRT(interaction, outgoing_dir);
    return std::make_pair(scatter_kernel, bsdf_pdf);
  }
  else
  {
    Spectral3 scatter_kernel = EvaluateWithBinaryMixtureDensity(
      ScatterFunctionSurface{shd, interaction, incident_dir},
      MixtureDistributionWrapper<8>(mixture),
      outgoing_dir,
      context,
      &bsdf_pdf,
      PROB_BSDF
    );
    scatter_kernel *= DFactorPBRT(interaction, outgoing_dir);
    return std::make_pair(scatter_kernel, bsdf_pdf);
  }
}


std::pair<Spectral3, double> CameraRenderWorker::EvaluateScatterKernel(const Double3 &incident_dir, const VolumeInteraction &vi, const PathNode &pn, const Double3 &outgoing_dir) const
{
  double bsdf_pdf = 0.;
  Spectral3 scatter_kernel = EvaluateWithBinaryMixtureDensity(
    ScatterFunctionVolume{vi, incident_dir},
    MixtureDistributionWrapper<8>(pn.radiance_fit->radiance_distribution),
    outgoing_dir,
    context,
    &bsdf_pdf,
    PROB_BSDF
  );
  scatter_kernel *= vi.sigma_s;
  return std::make_pair(scatter_kernel, bsdf_pdf);
}


ScatterSample CameraRenderWorker::SampleScatterKernel(
  const SurfaceInteraction &interaction, const PathNode &pn, const Double3 &reverse_incident_dir) const
{
  const auto& mixture = pn.radiance_fit->radiance_distribution;
  const Shader& shd = GetShaderOf(interaction, master->scene);

  if (shd.supports_lobes)
  {
    const auto lobes = shd.ComputeLobes(reverse_incident_dir, interaction, context);
    auto product = vmf_fitting::Product(lobes, mixture);
    vmf_fitting::Normalize(product);

    auto s = SampleWithBinaryMixtureDensity(
      ScatterFunctionSurface{shd, interaction, reverse_incident_dir},
      MakeBiComponentDistributionWrapper (
        MixtureDistributionWrapper<16>(product),
        MixtureDistributionWrapper<8>(mixture),
        0.5
      ),
      reverse_incident_dir, sampler, context,
      0.33);
    s.value *= DFactorPBRT(interaction, s.coordinates) / (double)s.pdf_or_pmf;
    return s;
  }
  else
  {
    auto s = SampleWithBinaryMixtureDensity(
      ScatterFunctionSurface{shd, interaction, reverse_incident_dir},
      MixtureDistributionWrapper<8>(mixture),
      reverse_incident_dir, sampler, context,
      PROB_BSDF);
    s.value *= DFactorPBRT(interaction, s.coordinates) / (double)s.pdf_or_pmf;
    return s;
  }
}



ScatterSample CameraRenderWorker::SampleScatterKernel(
  const VolumeInteraction &interaction, const PathNode &pn, const Double3 &reverse_incident_dir) const
{
  const auto& mixture = pn.radiance_fit->radiance_distribution;
  ScatterSample s = SampleWithBinaryMixtureDensity(
    ScatterFunctionVolume{interaction, reverse_incident_dir},
    MixtureDistributionWrapper<8>(mixture),
    reverse_incident_dir, sampler, context,
    PROB_BSDF);
  s.value *= interaction.sigma_s / (double)s.pdf_or_pmf;
  return s;
}


void CameraRenderWorker::MaybeAddNeeLighting(const PathNode & ps, double rr_weight, VertexCoefficients &coeffs) const
{
  assert (ps.interaction);

  auto [incident_radiance_estimator, pdf, segment_to_light, light_ref] = ::ComputeDirectLighting(
    master->scene, *ps.interaction, pickers->GetDistributionNee(), ps.medium_tracker, context, sampler);
  
  auto [scatter_kernel, bsdf_pdf] = mpark::visit([&,segment_to_light=segment_to_light](auto &&ia) {
    return EvaluateScatterKernel(-ps.incident_ray.dir, ia, ps, segment_to_light.ray.dir);
  }, *ps.interaction);

  const double mis_weight = MisWeight(pdf, bsdf_pdf);

  // The outgoing radiance, scattered into the previous node.
  // Accumulate because potentially multiple calls for this node.
  const Spectral3 outgoing_radiance = rr_weight * mis_weight * scatter_kernel * incident_radiance_estimator;

  coeffs.nee_radiance += outgoing_radiance;
  
  Spectral3 measurement_estimator = ps.weight * outgoing_radiance;
  pickers->ObserveReturnNee(this->picker_local, light_ref, measurement_estimator);

  RecordMeasurementToDebugBuffer(measurement_estimator);
}


PathIntermediateState CameraRenderWorker::PrepareStartAfterScatter(const PathNode &pn, const ScatterSample &smpl) const
{
  assert (pn.interaction);

  PathIntermediateState result {
    pn.medium_tracker,
    {
      /*pos = */ mpark::visit([](const auto &ia) { return ia.pos; }, *pn.interaction),
      /*dir = */ smpl.coordinates
    },
    false
  };

  if (const SurfaceInteraction* si = mpark::get_if<SurfaceInteraction>(&*pn.interaction); si)
  {
    result.ray.org += AntiSelfIntersectionOffset(*si, result.ray.dir);
    result.monochromatic = GetShaderOf(*si, master->scene).require_monochromatic;
    MaybeGoingThroughSurface(result.medium_tracker, result.ray.dir, *si);
  }

  return result;
}


void CameraRenderWorker::AddEmission(const PathNode &ps, VertexCoefficients &coeffs) const
{
  assert(ps.interaction);
  mpark::visit([this, &ps,&coeffs](auto &ia) { AddEmission(ps, ia, coeffs); }, *ps.interaction);
}


void CameraRenderWorker::AddEmission(const PathNode &ps, const SurfaceInteraction &interaction, VertexCoefficients &coeffs) const
{
  const auto emitter = GetMaterialOf(interaction, master->scene).emitter;
  if (!emitter)
    return;

  Spectral3 radiance = emitter->Evaluate(interaction.hitid, -ps.incident_ray.dir, context, nullptr);

  double mis_weight = 1.0;
  if (ps.last_scatter_pdf_value && enable_nee) // Should be set if this is secondary ray.
  {
    const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, interaction.hitid));
    const double area_pdf = emitter->EvaluatePdf(interaction.hitid, context);
    const double pdf_cvt = PdfConversion::AreaToSolidAngle(Length(ps.incident_ray.org - interaction.pos), ps.incident_ray.dir, interaction.normal);
    mis_weight = MisWeight(*ps.last_scatter_pdf_value, prob_select*area_pdf*pdf_cvt);
  }

  coeffs.emission = radiance;
  coeffs.emission_mis_factor = mis_weight;

  RecordMeasurementToDebugBuffer(mis_weight*radiance*ps.weight);
}


void CameraRenderWorker::AddEmission(const PathNode &ps, const VolumeInteraction &vi, VertexCoefficients &coeffs) const
{
  if (!vi.medium().is_emissive)
    return;

  // TODO: is this right?
  Spectral3 radiance = vi.medium().EvaluateEmission(vi.pos, context, nullptr);

  RecordMeasurementToDebugBuffer(radiance*ps.weight);
}


void CameraRenderWorker::AddEnvEmission(const PathNode &ps, VertexCoefficients &coeffs) const
{
  if (!master->scene.HasEnvLight())
    return;

  const auto &emitter = master->scene.GetTotalEnvLight();
  const auto radiance = emitter.Evaluate(-ps.incident_ray.dir, context);

  double mis_weight = 1.0;
  if (ps.last_scatter_pdf_value && enable_nee) // Should be set if this is secondary ray.
  {
    const double prob_select = pickers->GetDistributionNee().Pmf(Lights::MakeLightRef(master->scene, emitter));
    const double pdf_env = emitter.EvaluatePdf(-ps.incident_ray.dir, context);
    mis_weight = MisWeight(*ps.last_scatter_pdf_value, pdf_env*prob_select);
  }

  coeffs.emission = radiance;
  coeffs.emission_mis_factor = mis_weight;

  RecordMeasurementToDebugBuffer(mis_weight*radiance*ps.weight);
}



inline Spectral3 MaybeReWeightToMonochromatic(const Spectral3 &w, bool monochromatic)
{
  return monochromatic ? Spectral3{ w[0] * 3._sp,0,0 } : w;
}


void CameraRenderWorker::RecordMeasurementToCurrentPixel(const PathNode &node) const
{
  const Spectral3 direct   = node.coeffs.emission;
  const Spectral3 indirect = node.coeffs.nee_radiance + node.coeffs.indirect_radiance_accumulator;
  const Spectral3 measurement = node.coeffs.rr_weight * node.coeffs.scatter_from_prev * node.coeffs.segment_transmission_from_prev * (direct + indirect);

  assert(measurement.isFinite().all());
  auto color = Color::SpectralSelectionToRGB(measurement, context.lambda_idx);
  framebuffer[context.pixel_index] += color;
}


void CameraRenderWorker::RecordMeasurementToDebugBuffer(const Spectral3 measurement) const
{
  assert(measurement.isFinite().all());
  auto color = Color::SpectralSelectionToRGB(measurement, context.lambda_idx);
  debugbuffer[context.pixel_index] += color;
}


void CameraRenderWorker::PropagateIlluminationToParents(PathNode &node)
{
  assert (node.prev);
  PathNode &prev = *node.prev;

  const Spectral3 direct   = node.coeffs.segment_transmission_from_prev * node.coeffs.emission;
  const Spectral3 indirect = node.coeffs.segment_transmission_from_prev * (node.coeffs.nee_radiance + node.coeffs.indirect_radiance_accumulator);

  assert (node.last_scatter_pdf_value);
  if (master->record_samples_for_guiding)
    AddToTrainingData(prev, direct+indirect, node.incident_ray.dir, *node.last_scatter_pdf_value);

  prev.coeffs.indirect_radiance_accumulator += node.coeffs.rr_weight * node.coeffs.scatter_from_prev * (node.coeffs.emission_mis_factor * direct + indirect);
}


void CameraRenderWorker::AddToTrainingData(const PathNode &lit, const Spectral3 &radiance, const Double3 &dir, const Pdf pdf)
{
  if (pdf.IsFromDelta())
    return;

  assert (lit.interaction);
  assert((float)pdf > 0.f);
  assert(radiance.isFinite().all());

  const Spectral3 estimator = radiance / (double)pdf;

  // TODO: take only such samples generated by the radiance distribution?!
  mpark::visit(Overload(
    [&](const SurfaceInteraction &ia) {
      // if (ia.normal.dot(dir) <= 0.)
      //   return;
      radiance_recorder_surface->AddSample(
          this->radrec_local_surface,
          ia.pos,
          sampler,
          dir,
          estimator);
    },
    [&](const VolumeInteraction &vi) {
      radiance_recorder_volume->AddSample(
        this->radrec_local_volume,
        vi.pos,
        sampler,
        dir,
        estimator);
    }
  ), *lit.interaction);
}



} // namespace pathtracing_guided



std::unique_ptr<RenderingAlgo> AllocatePathtracingGuidedRenderingAlgo(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<pathtracing_guided::PathTracingAlgo2>(scene, params);
}
