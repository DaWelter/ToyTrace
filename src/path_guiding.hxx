#pragma once

#include "util.hxx"
#include "vec3f.hxx"
#include "spectral.hxx"
#include "box.hxx"
#include "span.hxx"
#include "distribution_mixture_models.hxx"
#include "path_guiding_tree.hxx"
#include "json_fwd.hxx"

//#include <random>
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

//#define WRITE_DEBUG_OUT 
//#define PATH_GUIDING_WRITE_SAMPLES

#if (defined PATH_GUIDING_WRITE_SAMPLES & !defined NDEBUG & defined HAVE_JSON)
#   define PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED 1
#endif



namespace guiding
{

inline static constexpr int CACHE_LINE_SIZE = 64;
boost::filesystem::path GetDebugFilePrefix();

using Color::Spectral3f;

struct AllEqual
{
    template<class T, int rows>
    bool operator()(const Eigen::Array<T, rows, 1> &a, const Eigen::Array<T, rows, 1> &b) const
    {
        return (a == b).all();
    }
};


// TODO: This needs to go away ... 
template<class T>
Span<T> CopyToSpan(const ToyVector<T> &v)
{
    // May the gods have mercy ...
    auto mem = std::make_unique<T[]>(v.size());
    memcpy(mem.get(), v.data(), sizeof(T)*v.size());
    return Span<T>(mem.release(), v.size());
}

template<class T>
void FreeSpan(Span<T> &v)
{
    delete[] v.data();
    v = Span<T>{};
}


struct IncidentRadiance
{
    Double3 pos;
    Float3 reverse_incident_dir;
    float weight;
    bool is_original = true;
};


using LeafStatistics = OnlineVariance::Accumulator<Eigen::Array3d, int64_t>;

// Should be quite lightweight.
static_assert(sizeof(tbb::spin_mutex) <= 16);

// About 1100 Bytes.
struct CellData
{
    CellData() = default;

    CellData(const CellData &) = delete;
    CellData& operator=(const CellData &) = delete;
    
    CellData(CellData &&) = default;
    CellData& operator=(CellData &&) = default;

    alignas (CACHE_LINE_SIZE) struct CurrentEstimate {
      // Normalized to the total incident flux. So radiance_distribution(w) * incident_flux_density is the actual radiance from direction w.
      vmf_fitting::VonMisesFischerMixture radiance_distribution;
      Double3 cell_size = Double3::Constant(NaN);
      double incident_flux_density{0.};
    } current_estimate;
    
    alignas (CACHE_LINE_SIZE) struct Learned { 
      vmf_fitting::VonMisesFischerMixture radiance_distribution;
      vmf_fitting::incremental::Data fitdata;
      LeafStatistics leaf_stats;
    } learned;
    
    int index = -1;
    long last_num_samples = 0;
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
  void Write(Span<const IncidentRadiance> records);
  void Close();
  const std::string_view GetFilename() const { return filename; }
private:
  bool first = true;
  std::ofstream file;
  std::string filename;
};
#endif


struct CellDataTemporary
{
  CellDataTemporary() :
    data_chunks{}, mutex{}, fit_task_submissions_count{ 0 }
  {
    data_chunks.reserve(8);
  }

  CellDataTemporary(const CellDataTemporary &) = delete;
  CellDataTemporary& operator=(const CellDataTemporary &) = delete;

  CellDataTemporary(CellDataTemporary &&) = delete;
  CellDataTemporary& operator=(CellDataTemporary &&) = delete;

  ToyVector<Span<IncidentRadiance>> data_chunks;
  tbb::spin_mutex mutex; // To protected the incident radiance buffer
  int fit_task_submissions_count;
};


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
        using Record = IncidentRadiance;
        using RecordBuffer = ToyVector<Record>;
        using RadianceEstimate = CellData::CurrentEstimate;

        struct ThreadLocal 
        {
            ToyVector<RecordBuffer> records_by_cells;
        };

        PathGuiding(const Box &region, double cellwidth, const RenderingParameters &params, tbb::task_arena &the_task_arena);

        void BeginRound(Span<ThreadLocal*> thread_locals);
        void WriteDebugData(const std::string name_prefix);

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

        static Record ComputeStochasticFilterPosition(const Record & rec, const CellData &cd, Sampler &sampler);
        
        void ProcessSamples(int cell_idx);
        void LearnIncidentRadianceIn(CellData &cell, Span<IncidentRadiance> buffer);
        void Enqueue(int cell_idx, ToyVector<Record> &sample_buffer);
        CellData& LookupCellData(const Double3 &p);

        Box region;
        kdtree::Tree recording_tree;
        ToyVector<CellData, AlignedAllocator<CellData, CACHE_LINE_SIZE>> cell_data;
        std::unique_ptr<CellDataTemporary[]> cell_data_temp;
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
        std::unique_ptr<CellDebug[]> cell_data_debug;
#endif
        tbb::atomic<int64_t> num_recorded_samples = 0;
        
        int param_num_initial_samples;
        int param_em_every;
        double param_prior_strength;

        int round = 0;

        tbb::task_arena *the_task_arena;
        tbb::task_group the_task_group;
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