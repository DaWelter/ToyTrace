#pragma once

#include "util.hxx"
#include "vec3f.hxx"
#include "spectral.hxx"
#include "box.hxx"
#include "span.hxx"
#include "distribution_mixture_models.hxx"
#include "path_guiding_quadtree.hxx"
#include "path_guiding_tree.hxx"
#include "json_fwd.hxx"

//#include <random>
#include <cmath>
#include <fstream>
#include <tuple>

//#include <boost/pool/object_pool.hpp>
#include <boost/filesystem/path.hpp>

#include <tbb/spin_mutex.h>
#include <tbb/mutex.h>
#include <tbb/atomic.h>
#include <tbb/task_group.h>
#include <tbb/task_arena.h>

struct SurfaceInteraction;
struct RenderingParameters;

#define WRITE_DEBUG_OUT 
//#define PATH_GUIDING_WRITE_SAMPLES

#if (defined PATH_GUIDING_WRITE_SAMPLES & !defined NDEBUG & defined HAVE_JSON)
#   define PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED 1
#endif



namespace guiding
{

inline static constexpr int CACHE_LINE_SIZE = 64;
boost::filesystem::path GetDebugFilePrefix();

using Color::Spectral3f;
struct CellData;


struct IncidentRadiance
{
    Double3 pos;
    Float3 reverse_incident_dir;
    float weight;
    bool is_original = true;
};


using LeafStatistics = Accumulators::OnlineCovariance<double, 3, int64_t>;

// Should be quite lightweight.
static_assert(sizeof(tbb::spin_mutex) <= 16);

namespace MoVmfRadianceDistribution
{

class RadianceDistributionLearned;

class RadianceDistributionSampled
{
  friend class RadianceDistributionLearned;
  // Normalized to the total incident flux. So radiance_distribution(w) * incident_flux_density is the actual radiance from direction w.
  vmf_fitting::VonMisesFischerMixture<> normalized_distribution;
  double incident_flux_density{ 1. };
  double incident_flux_confidence_bounds{ 0. };

public:
  RadianceDistributionSampled()
  {
    vmf_fitting::InitializeForUnitSphere(normalized_distribution);
  }

  double Pdf(const Eigen::Vector3d &dir) const
  {
    return vmf_fitting::Pdf(normalized_distribution, dir.cast<float>());
  }

  std::pair<Double3, double> Sample(Sampler &sampler) const
  {
    auto r1 = sampler.Uniform01();
    auto r2 = sampler.Uniform01();
    auto r3 = sampler.Uniform01();
    Eigen::Vector3d w = vmf_fitting::Sample(normalized_distribution, { r1,r2,r3 }).template cast<double>();
    // Float32 can be lower precision than I have otherwise. So renormalize the direction.
    const double n2 = w.squaredNorm();
    if (std::abs(n2 - 1.) >= 1.e-11)
    {
      w /= std::sqrt(n2);
    }
    const double pdf = Pdf(w);
    return { w, pdf };
  }

  double Evaluate(const Eigen::Vector3d &dir) const
  {
    return Pdf(dir) * incident_flux_density;
  }

  std::pair<double, double> EvaluateErr(const Eigen::Vector3d &dir) const
  {
    const double pdf = Pdf(dir);
    return { pdf*incident_flux_density, pdf*incident_flux_confidence_bounds };
  }

  std::pair<double, double> EvaluateFluxErr() const
  {
    return std::make_pair(incident_flux_density, incident_flux_confidence_bounds);
  }

  Float3 ComputeStochasticFilteredDirection(const IncidentRadiance & rec, Sampler &sampler) const;

  rapidjson::Value ToJSON(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> &a) const;
};


class RadianceDistributionLearned
{
  vmf_fitting::VonMisesFischerMixture<> normalized_distribution;
  vmf_fitting::incremental::Data<> fitdata;
  Accumulators::OnlineVariance<double, int64_t> incident_flux_density_accum;

public:
  class Parameters
  {
    friend class RadianceDistributionLearned;
    vmf_fitting::incremental::Params<> params;
    vmf_fitting::VonMisesFischerMixture<> prior_mode;
  public:
    Parameters(const CellData &cd);
  };

  RadianceDistributionLearned()
  {
    vmf_fitting::InitializeForUnitSphere(normalized_distribution);
  }

  void InitialFit(Span<const IncidentRadiance> buffer, const Parameters &params)
  {
    IncrementalFit(buffer, params);
  }

  void IncrementalFit(Span<const IncidentRadiance> buffer, const Parameters &params)
  {
    for (const auto &in : buffer)
    {
      if (in.is_original)
        incident_flux_density_accum += in.weight;

      if (in.weight <= 0.)
        continue;

      float weight = in.weight;
      const auto xs = Span<const Eigen::Vector3f>(&in.reverse_incident_dir, 1);
      const auto ws = Span<const float>(&weight, 1);
      vmf_fitting::incremental::Fit(normalized_distribution, fitdata, params.params, xs, ws);
    }
  }

  RadianceDistributionSampled IterationUpdateAndBake(const RadianceDistributionSampled &previous_radiance_dist)
  {
    return Bake();
  }

  RadianceDistributionSampled Bake() const
  {
    RadianceDistributionSampled dst;
    dst.normalized_distribution = this->normalized_distribution;
    dst.incident_flux_density = this->incident_flux_density_accum.Mean();
    dst.incident_flux_confidence_bounds = this->incident_flux_density_accum.Count() >= 10 ?
      std::sqrt(this->incident_flux_density_accum.Var() / static_cast<double>(this->incident_flux_density_accum.Count())) : LargeNumber;
    assert(std::isfinite(dst.incident_flux_confidence_bounds));
    return dst;
  }

  rapidjson::Value ToJSON(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> &a) const;
};

} // namespace MoVmfRadianceDistribution


namespace quadtree_radiance_distribution
{

class RadianceDistributionLearned;

Eigen::Vector2f MapSphereToTree(const Eigen::Vector3f &dir);
Eigen::Vector3f MapTreeToSphere(const Eigen::Vector2f &uv);

inline constexpr double JacobianInv = 1./(4.*Pi);
inline constexpr double safe_error_number = 0.01*static_cast<double>(LargeFloat)*util::Pow(0.25, quadtree::detail::MAX_DEPTH);

class RadianceDistributionSampled
{
  friend class RadianceDistributionLearned;
  quadtree::Tree tree;
  Eigen::ArrayXd node_means;
  Eigen::ArrayXd node_stddev;
  Eigen::ArrayXf node_sample_probs;
  Eigen::ArrayXf node_sample_prior;
  // double incident_flux_density{ 0. };
  // double incident_flux_confidence_bounds{ 0. };

  std::pair<int, float> FindNodeAndInvSolidAngle(const Eigen::Vector2f &uv) const
  {
    float area_inv = JacobianInv;
    quadtree::detail::DecentHelper d{uv, tree};
    for (;!d.IsLeaf();)
    {
      d.TraverseInplace();
      area_inv *= 4.f;
    }
    const int idx = d.GetNode().idx;
    return {idx, area_inv};
  }

  static auto AssertFinite(const std::pair<double, double> &x)
  {
    assert(std::isfinite(x.first));
    assert(std::isfinite(x.second));
    return x;
  }

  static auto AssertFinite(double x)
  {
    assert(std::isfinite(x));
    return x;
  }

public:
  RadianceDistributionSampled() :
    tree{}, node_means(tree.NumNodes()), node_stddev(tree.NumNodes()), node_sample_probs(tree.NumNodes())
  {
    node_means.setOnes();
    node_stddev.setZero();
    node_sample_probs.setOnes();
    node_sample_prior.setOnes();
  }

  double Pdf(const Eigen::Vector3d &dir) const
  {
    const auto uv = MapSphereToTree(dir.cast<float>());
    const double ret = quadtree::Pdf(tree, AsSpan(node_sample_probs), uv)*JacobianInv;
    assert(std::isfinite(ret));
    return ret;
  }

  std::pair<Double3, double> Sample(Sampler &sampler) const
  {
    auto [uv, pdf] = quadtree::Sample(tree, AsSpan(node_sample_probs), sampler);
    pdf *= JacobianInv;

    assert (uv.allFinite());
    assert (std::isfinite(pdf) && pdf > 0.f);
    auto w = MapTreeToSphere(uv).cast<double>().eval();
    // Float32 can be lower precision than I have otherwise. So renormalize the direction.
    const double n2 = w.squaredNorm();
    if (std::abs(n2 - 1.) >= 1.e-11)
    {
      w /= std::sqrt(n2);
    }
    assert(w.allFinite());
    return std::make_pair(w, pdf);
  }

  double Evaluate(const Eigen::Vector3d &dir) const
  {
    const auto uv = MapSphereToTree(dir.cast<float>());
    auto [idx, area_inv] = FindNodeAndInvSolidAngle(uv);
    return AssertFinite(area_inv*node_means[idx]);
  }

  std::pair<double, double> EvaluateErr(const Eigen::Vector3d &dir) const
  {
    const auto uv = MapSphereToTree(dir.cast<float>());
    auto [idx, area_inv] = FindNodeAndInvSolidAngle(uv);
    return AssertFinite({ area_inv*node_means[idx], area_inv*node_stddev[idx] });
  }

  std::pair<double, double> EvaluateFluxErr() const
  {
    return AssertFinite({ node_means[tree.GetRoot().idx], node_stddev[tree.GetRoot().idx] });
  }

  Float3 ComputeStochasticFilteredDirection(const IncidentRadiance & rec, Sampler &sampler) const;

  rapidjson::Value ToJSON(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> &a) const;
};


class RadianceDistributionLearned
{
  quadtree::Tree tree;
  Accumulators::SoaOnlineVariance<double, long> node_weights;

  inline Eigen::ArrayXd CalcRelativeCounts() const
  {
    auto ret = node_weights.Counts().cast<double>().eval();
    auto total_sample_count = ret[tree.GetRoot().idx];
    assert(total_sample_count > 0);
    ret *= 1./total_sample_count;
    return ret;
  }

  static constexpr double min_sample_count = 2;
  static constexpr double good_sample_count = 100;

public:
  class Parameters
  {
    friend class RadianceDistributionLearned;
  public:
    Parameters(const CellData &cd, int round) : round{round} {}
    int round;
  };

  RadianceDistributionLearned()
    : tree{}, node_weights(1)
  {
  }

  void IncrementalFit(Span<const IncidentRadiance> buffer, const Parameters &params, RadianceDistributionSampled &sampled_distrib);
  void InitialFit(Span<const IncidentRadiance> buffer, const Parameters &params);

  RadianceDistributionSampled IterationUpdateAndBake(const RadianceDistributionSampled &previous_radiance_dist);
  RadianceDistributionSampled Bake() const;

  rapidjson::Value ToJSON(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> &a) const;
};


} // namespace quadtree_radiance_distribution


using quadtree_radiance_distribution::RadianceDistributionLearned;
using quadtree_radiance_distribution::RadianceDistributionSampled;

// using MoVmfRadianceDistribution::RadianceDistributionLearned;
// using MoVmfRadianceDistribution::RadianceDistributionSampled;


// About 1100 Bytes.
struct CellData
{
    CellData() = default;

    CellData(const CellData &) = delete;
    CellData& operator=(const CellData &) = delete;
    
    CellData(CellData &&) = default;
    CellData& operator=(CellData &&) = default;

    alignas (CACHE_LINE_SIZE) struct CurrentEstimate {
      RadianceDistributionSampled radiance_distribution;
      Box cell_bbox{};
      Eigen::Matrix3d points_cov_frame{Eigen::zero}; // U*sqrt(Lambda), where U is composed of Eigenvectors, and Lambda composed of Eigenvalues.
      Eigen::Vector3d points_mean{Eigen::zero};
      Eigen::Vector3d points_stddev{Eigen::zero};
    } current_estimate;
    
    alignas (CACHE_LINE_SIZE) struct Learned { 
      RadianceDistributionLearned radiance_distribution;
      LeafStatistics leaf_stats;
    } learned;
    
    long last_num_samples = 0;
    long max_num_samples = 0;
    int index = -1;
};


#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
class CellDebug
{
public:
  CellDebug() = default;
  ~CellDebug();

  CellDebug(const CellDebug &) = delete;
  CellDebug& operator=(const CellDebug &) = delete;

  CellDebug(CellDebug &&) = delete;
  CellDebug& operator=(CellDebug &&) = delete;

  void Open(std::string filename_);
  void Write(const Double3 &pos, const Float3 &dir, float weight);
  void Close();
  const std::string_view GetFilename() const { return filename; }

  vmf_fitting::incremental::Params<> params{};
private:
  std::ofstream file;
  std::string filename;
};
#endif


class CellIterator : kdtree::LeafIterator
{
    Span<const CellData> celldata;
  public:
    CellIterator(const kdtree::Tree &tree_, Span<const CellData> celldata_, const Ray &ray_, double tnear_init, double tfar_init)  noexcept
      : kdtree::LeafIterator{tree_, ray_, tnear_init, tfar_init}, celldata{celldata_}
    {}

    const CellData::CurrentEstimate& operator*() const noexcept
    {
      return celldata[kdtree::LeafIterator::Payload()].current_estimate;
    }

    using kdtree::LeafIterator::Interval;
    using kdtree::LeafIterator::operator bool;
    using kdtree::LeafIterator::operator++;
};



class PathGuiding
{
    public:
        using RadianceEstimate = CellData::CurrentEstimate;

        struct ThreadLocal 
        {
            ToyVector<IncidentRadiance> samples;
        };

        PathGuiding(const Box &region, double cellwidth, const RenderingParameters &params, tbb::task_arena &the_task_arena, const char* name);

        void BeginRound(Span<ThreadLocal*> thread_locals);

        void AddSample(
          ThreadLocal& tl, const Double3 &pos, 
          Sampler &sampler, const Double3 &reverse_incident_dir, 
          const Spectral3 &radiance);

        
        const RadianceEstimate& FindRadianceEstimate(const Double3 &p) const;

        void FinalizeRound(Span<ThreadLocal*> thread_locals);

        void PrepareAdaptedStructures();

        CellIterator MakeCellIterator(const Ray &ray, double tnear_init, double tfar_init) const
        {
          return CellIterator{recording_tree, AsSpan(cell_data), ray, tnear_init, tfar_init};
        }

    private:
        void WriteDebugData();
        void AdaptIncremental();
        void AdaptInitial(Span<ThreadLocal*> thread_locals);
        void FitTheSamples(Span<ThreadLocal*> thread_locals);
        ToyVector<int> ComputeCellIndices(Span<const IncidentRadiance> samples) const;
        ToyVector<ToyVector<IncidentRadiance>> SortSamplesIntoCells(Span<const int> cell_indices, Span<const IncidentRadiance> samples) const;
        void GenerateStochasticFilteredSamplesInplace(Span<int> cell_indices, Span<IncidentRadiance> samples) const;

        static IncidentRadiance ComputeStochasticFilterPosition(const IncidentRadiance & rec, const CellData &cd, Sampler &sampler);
        
        void FitTheSamples(CellData &cell, Span<IncidentRadiance> buffer) const;
        
        void Enqueue(int cell_idx, ToyVector<IncidentRadiance> &sample_buffer);
        CellData& LookupCellData(const Double3 &p);

        Box region;
        kdtree::Tree recording_tree;
        ToyVector<CellData, AlignedAllocator<CellData, CACHE_LINE_SIZE>> cell_data;
        std::string name;
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
        std::unique_ptr<CellDebug[]> cell_data_debug;
#endif
        int param_num_initial_samples;
        int param_em_every;
        double param_prior_strength;
        int64_t previous_max_samples_per_cell = 0;
        int64_t previous_total_samples = 0;

        tbb::task_arena *the_task_arena;
        tbb::task_group the_task_group;

        int round = 0;
        int sub_round = 0;
};


template<class Iter1_, class Iter2_>
class CombinedIntervalsIterator
{
  using Iter1 = Iter1_; //guiding::kdtree::LeafIterator;
  using Iter2 = Iter2_; //SegmentIterator;
  Iter1 leaf_iter;
  Iter2 boundary_iter;
  double tnear, tfar;
  double li_tnear, li_tfar;
  double bi_tnear, bi_tfar;

public: 
  CombinedIntervalsIterator(Iter1 leaf_iter, Iter2 boundary_iter)
    : leaf_iter{leaf_iter}, boundary_iter{boundary_iter}
  {
    std::tie(li_tnear, li_tfar) = leaf_iter.Interval();
    std::tie(bi_tnear, bi_tfar) = boundary_iter.Interval();
    tnear = std::max(li_tnear, bi_tnear);
    tfar = std::min(li_tfar, bi_tfar);    
  }

  operator bool() const noexcept
  {
    return leaf_iter && boundary_iter;
  }

  void operator++()
  {
    // If the interval ends of the two iterators coincide, then here an interval of length zero 
    // will be produced, at the place of the boundary.
    tnear = tfar;
    if (li_tfar <= bi_tfar)
    {
      ++leaf_iter;
      if (leaf_iter)
      {
        std::tie(li_tnear, li_tfar) = leaf_iter.Interval();
      }
    }
    else
    {
      ++boundary_iter;
      if (boundary_iter)
      {
        std::tie(bi_tnear, bi_tfar) = boundary_iter.Interval();
      }
    }
    tfar = std::min(li_tfar, bi_tfar);
  }

  auto Interval() const noexcept
  {
    return std::make_pair(tnear, tfar);
  }

  decltype(*leaf_iter) DereferenceFirst() const
  {
    return *leaf_iter;
  }

  decltype(*boundary_iter) DereferenceSecond() const
  {
    return *boundary_iter;
  }
};


} // namespace guiding