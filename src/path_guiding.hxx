#pragma once

#include "util.hxx"
#include "vec3f.hxx"
#include "spectral.hxx"
#include "box.hxx"
#include "span.hxx"
#include "distribution_mixture_models.hxx"

#include <unordered_map>
#include <random>
#include <fstream>
#include <tuple>

#include <boost/pool/object_pool.hpp>
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


#ifdef HAVE_JSON
namespace rapidjson
{

class CrtAllocator;

template <typename BaseAllocator>
class MemoryPoolAllocator;

template <typename Encoding, typename Allocator>
class GenericValue;

template <typename Encoding, typename Allocator, typename StackAllocator>
class GenericDocument;

template<typename CharType>
struct UTF8;


typedef GenericDocument<UTF8<char>, MemoryPoolAllocator<CrtAllocator>, CrtAllocator> Document;
typedef GenericValue<UTF8<char>, MemoryPoolAllocator<CrtAllocator>> Value;
} // rapidjson
#endif


namespace guiding
{

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


struct CellData
{
    CellData() = default;
    
    vmf_fitting::VonMisesFischerMixture mixture_sampled;
    Double3 cell_size;
    int num = 0;

    vmf_fitting::VonMisesFischerMixture mixture_learned;
    vmf_fitting::incremental::Data fitdata;
    LeafStatistics leaf_stats;
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



// old tree:
//    keep for guiding
//    leaf-payload is gmm
// adapted tree:
//    record samples and learn gmm.
//    leaf-payload is gmm + sample statistics / incremental learning data.


namespace kdtree
{

struct Node
{
  bool is_leaf;
};


struct Leaf : Node, CellData
{
};


struct Branch : Node
{
  double split_pos = std::numeric_limits<double>::quiet_NaN();
  Node* left = nullptr;
  Node* right = nullptr;
  //int branch_index = -1;
  char split_axis = -1;
};


struct AdaptParams
{
  std::uint64_t max_num_points = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t min_num_points = 0;
};


struct user_allocator_aligned
{
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  static char * malloc(const size_type bytes)
  { return reinterpret_cast<char*>(boost::alignment::aligned_alloc(32, bytes)); }
  static void free(char * const block)
  { boost::alignment::aligned_free(block); }
};


template<class T>
using aligned_pool = boost::object_pool<T, user_allocator_aligned>;


class Tree
{
  friend class TreeAdaptor;
  // Pointers for movability
  std::unique_ptr<aligned_pool<Leaf>> leaf_storage;
  std::unique_ptr<aligned_pool<Branch>> branch_storage;
  Node* root = nullptr;
  ToyVector<CellData*> payloads;

  Branch* AllocateBranch()
  {
    auto* node = branch_storage->construct();
    node->is_leaf = false;
    return node;
  }

  Leaf* AllocateLeaf()
  {
    Leaf* tmp = leaf_storage->construct();
    tmp->is_leaf = true;
    payloads.push_back(tmp);
    return tmp;
  }

  Leaf* AllocateCopy(const Leaf &other)
  {
    auto* ret = leaf_storage->construct(other);
    payloads.push_back(ret);
    return ret;
  }

  struct TagUninitialized {};

  Tree(TagUninitialized) :
    leaf_storage{ std::make_unique<aligned_pool<Leaf>>() },
    branch_storage{ std::make_unique<aligned_pool<Branch>>() }
  {
  }

public:
  Tree() :
    Tree(TagUninitialized{})
  {
    root = AllocateLeaf();
  }

  Tree(const Tree &) = delete;
  Tree& operator=(const Tree&) = delete;

  Tree(Tree &&) = default;
  Tree& operator=(Tree &&) = default;

  int NumLeafs() const { return isize(payloads); }

  Span<CellData* const> GetData() { return AsSpan(payloads); }

  Span<CellData * const> GetData() const { return AsSpan(payloads); }

  const Node* GetRoot() const
  {
    return root;
  }

  CellData* Lookup(const Double3 &p)
  {
    Node* current = root;
    while (true)
    {
      assert(current);
      if (current->is_leaf)
      {
        return static_cast<Leaf*>(current);
      }
      else
      {
        const auto* b = static_cast<const Branch*>(current);
        const bool is_left = p[b->split_axis] < b->split_pos;
        current = is_left ? b->left : b->right;
      }
    }
  }

  const CellData* Lookup(const Double3 &p) const
  {
    return const_cast<Tree*>(this)->Lookup(p);
  }

#ifdef HAVE_JSON
  void DumpTo(rapidjson::Document &doc, rapidjson::Value & parent) const;
#endif
};


class TreeAdaptor
{
public:
  TreeAdaptor(const AdaptParams &params) :
    params{ params }
  {
  }

  void AdaptInplace(Tree &tree)
  {
    this->tree = &tree;
    tree.root = AdaptRecursive(*tree.root);
    int num = 0;
    for (auto &cell : tree.GetData())
    {
      cell->num = num++;
    }
  }

private:
  AdaptParams params;
  Tree *tree = nullptr;
  
  inline CellData& GetPayload(Leaf &p) { return static_cast<CellData&>(p); }
  inline const CellData& GetPayload(const Leaf &p) { return static_cast<const CellData&>(p); }

  // Axis and location
  std::pair<int, double> DetermineSplit(const Leaf &p)
  {
    int axis = -1;
    double pos = NaN;
    const auto& stats = GetPayload(p).leaf_stats;
    if (stats.Count())
    {
      const auto mean = stats.Mean();
      const auto var = stats.Var();
      var.maxCoeff(&axis);
      pos = mean[axis];
    }
    return { axis, pos };
  }

  Node* AdaptRecursive(Leaf &node)
  {
    if (GetPayload(node).leaf_stats.Count() > params.max_num_points)
    {
      // Split
      auto* as_branch = tree->AllocateBranch();
      auto* right = tree->AllocateCopy(node);
      *right = node;
      as_branch->left = &node;
      as_branch->right = right;
      std::tie(as_branch->split_axis, as_branch->split_pos) = DetermineSplit(node);
      return as_branch;
    }
    else
    {
      return &node;
    }
  }

  Node* AdaptRecursive(Branch &node)
  {
    // Always take branch nodes as they are. Never contract them to single leaf?!!
    // Update the points speculatively because a leaf could have been replaced by a branch.
    node.left  = AdaptRecursive(*node.left);
    node.right = AdaptRecursive(*node.right);
    return &node;
  }

  Node* AdaptRecursive(Node &node)
  {
    return node.is_leaf ? AdaptRecursive(static_cast<Leaf&>(node)) : AdaptRecursive(static_cast<Branch &>(node));
  }
};


} // kdtree


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

        void AddSample(ThreadLocal& tl, const SurfaceInteraction &surface, Sampler &sampler, const Double3 &reverse_incident_dir, const Spectral3 &radiance);
        //void TransferSamplesToGuidingSystem(ThreadLocal& tl); // Transfer samples from render threads to guiding algorithm

        const vmf_fitting::VonMisesFischerMixture* FindSamplingMixture(const Double3 &p) const;

        void FinalizeRound(Span<ThreadLocal*> thread_locals);

        void PrepareAdaptedStructures();

    private:

        static Record ComputeStochasticFilterPosition(const Record & rec, const CellData &cd, Sampler &sampler);
        
        void ProcessSamples(int cell_idx);
        void LearnIncidentRadianceIn(CellData &cell, Span<IncidentRadiance> buffer);
        void Enqueue(int cell_idx, ToyVector<Record> &sample_buffer);

        Box region;
        kdtree::Tree recording_tree;
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