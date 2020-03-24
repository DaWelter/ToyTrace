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




template<class T>
inline Vec<T,2> StereographicProjection(const Vec<T,3> &p) noexcept
{
    // https://en.wikipedia.org/wiki/Stereographic_projection
    // except I flip the z direction
    Vec<T,2> xy{ p[0], p[1] };
    xy /= T(1)+p[2];
    return xy;
}


template<class T>
inline Vec<T,3> InvStereographicProjection(const Vec<T,2> &p) noexcept
{
    // https://en.wikipedia.org/wiki/Stereographic_projection
    // except I flip the z direction
    T rec_denom = T(1)/(T(1) + Sqr(p[0]) + Sqr(p[1]));
    return Vec<T,3>{
      T(2)*rec_denom*p[0],
      T(2)*rec_denom*p[1],
      -rec_denom*(T(-1) + Sqr(p[0]) + Sqr(p[1]))
    };
}

// See https://en.wikipedia.org/wiki/Stereographic_projection
// This function returns 
//                         dx dy
//                         ------ = Det(J JT) (??)
//                           dA
// So I can compute p(v) = dx dy / dA p(x,y)
inline float StereoprojectionJacobianDet(const Float2 &p) noexcept
{
  float tmp = 0.5f*(1.f + Sqr(p[0]) + Sqr(p[1]));
  return tmp * tmp;
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
    
    int index = -1;

    alignas (CACHE_LINE_SIZE) struct Learned { 
      vmf_fitting::VonMisesFischerMixture radiance_distribution;
      vmf_fitting::incremental::Data fitdata;
      LeafStatistics leaf_stats;
    } learned;
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




class SurfacePathGuiding
{
    public:
        using Record = IncidentRadiance;
        using RecordBuffer = ToyVector<Record>;

        struct ThreadLocal 
        {
            ToyVector<RecordBuffer> records_by_cells;
        };

        SurfacePathGuiding(const Box &region, double cellwidth, const RenderingParameters &params, tbb::task_arena &the_task_arena);

        void BeginRound(Span<ThreadLocal*> thread_locals);
        void WriteDebugData();

        void AddSample(
          ThreadLocal& tl, const SurfaceInteraction &surface, 
          Sampler &sampler, const Double3 &reverse_incident_dir, 
          const Spectral3 &radiance);

        const vmf_fitting::VonMisesFischerMixture* FindSamplingMixture(const Double3 &p) const;

        void FinalizeRound(Span<ThreadLocal*> thread_locals);

        void PrepareAdaptedStructures();

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


} // namespace guiding