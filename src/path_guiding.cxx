#include "path_guiding.hxx"
#include "scene.hxx"

#ifdef HAVE_JSON
#include "json.hxx"
#include "rapidjson/document.h"
#endif
#include <fstream>

#include <tbb/parallel_for_each.h>

#ifdef _MSC_VER
static const char* DEBUG_FILE_PREFIX = "D:\\tmp2\\";
#else
static const char* DEBUG_FILE_PREFIX = "/tmp/";
#endif




#ifdef HAVE_JSON
namespace rapidjson_util {

rapidjson::Value ToJSON(const gmm_fitting::GaussianMixture2d &mixture, Alloc &a)
{
  using namespace rapidjson_util;
  rj::Value jgmm(rj::kObjectType);
  jgmm.AddMember("weights", ToJSON(mixture.weights, a), a);
  jgmm.AddMember("means", ToJSON(mixture.means, a), a);
  jgmm.AddMember("precisions", ToJSON(mixture.precisions, a), a);
  return jgmm;
}

rapidjson::Value ToJSON(const vmf_fitting::VonMisesFischerMixture &mixture, Alloc &a)
{
  using namespace rapidjson_util;
  rj::Value jgmm(rj::kObjectType);
  jgmm.AddMember("weights", ToJSON(mixture.weights, a), a);
  jgmm.AddMember("means", ToJSON(mixture.means, a), a);
  jgmm.AddMember("concentrations", ToJSON(mixture.concentrations, a), a);
  return jgmm;
}


} // rapidjson_util namespace
#endif


namespace guiding
{

boost::filesystem::path GetDebugFilePrefix()
{
  return DEBUG_FILE_PREFIX;
}

static constexpr size_t CELL_BUFFER_SIZE = 1024;

#if 0
HashGridNaiveDynamic::HashGridNaiveDynamic(const Box &, double cellwidth) :
    inv_cellsize{1./cellwidth}
{
    celldata.reserve(1024);
}


CellData& HashGridNaiveDynamic::Lookup(const Double3& p)
{
    CellIndex key = (p * inv_cellsize).cast<int>();
    return celldata[key];
}
#endif

SurfacePathGuiding::SurfacePathGuiding(const Box &region, double cellwidth, const RenderingParameters &params, tbb::task_arena &the_task_arena) :
    region{ region },
    recording_tree{},
    param_num_initial_samples{ params.guiding_tree_subdivision_factor },
    param_em_every{ params.guiding_em_every },
    param_prior_strength{ params.guiding_prior_strength },
    the_task_arena{ &the_task_arena }
    //graph{},
    //transmission_node{
    //    graph,
    //    tbb::flow::unlimited,
    //    [this](std::tuple<int, Span<Record>> x) 
    //    {  
    //        this->ProcessSamples(std::get<0>(x), std::get<1>(x));
    //        FreeSpan(std::get<1>(x));
    //    }
    //}
{
  auto &cell = *recording_tree.GetData()[0];
  vmf_fitting::InitializeForUnitSphere(cell.mixture_learned);
  vmf_fitting::InitializeForUnitSphere(cell.mixture_sampled);

}


const vmf_fitting::VonMisesFischerMixture* SurfacePathGuiding::FindSamplingMixture(const Double3 &p) const
{
    return &recording_tree.Lookup(p)->mixture_sampled;
}


namespace {
SurfacePathGuiding::Record MakeSampleRecord(const SurfaceInteraction &surface, const Double3 &reverse_incident_dir, const Spectral3 &radiance)
{
  return {
        surface.pos,
        reverse_incident_dir.cast<float>(),
        radiance.cast<float>().sum()
  };
}
}


void  SurfacePathGuiding::Enqueue(int cell_num, ToyVector<Record> &sample_buffer)
{  
  auto buffer = CopyToSpan(sample_buffer);
  sample_buffer.clear();

  bool do_submit = false;
  auto &tcd = cell_data_temp[cell_num];
  {
    decltype(tcd.mutex)::scoped_lock l(tcd.mutex);
    if (tcd.fit_task_submissions_count <= 0)
    {
      tcd.fit_task_submissions_count++;
      do_submit = true;
    }
    tcd.data_chunks.push_back(buffer);
  }

  if (!do_submit)
    return;

  the_task_arena->execute(
    [this, cell_num]() {
      the_task_group.run(
        [this, cell_num]()
        {
          this->ProcessSamples(cell_num);
      });
  });
}


void SurfacePathGuiding::AddSample(
  ThreadLocal& tl, const SurfaceInteraction &surface,
  Sampler &sampler, const Double3 &reverse_incident_dir, const Spectral3 &radiance)
{
    auto BufferMaybeSendOffSample = [&tl,this](const CellData &cell, const Record &rec)
    {
      auto& sample_buffer = tl.records_by_cells[cell.num];
      sample_buffer.push_back(rec);

      if (sample_buffer.size() >= CELL_BUFFER_SIZE)
      {
        Enqueue(cell.num, sample_buffer);
      }
    };

    auto rec = MakeSampleRecord(surface, reverse_incident_dir, radiance);

    auto& cell = *recording_tree.Lookup(surface.pos);
    BufferMaybeSendOffSample(cell, rec);

    for (int i = 0; i < 1; ++i)
    {
      const auto new_rec = ComputeStochasticFilterPosition(rec, cell, sampler);

      auto& other_cell = *recording_tree.Lookup(new_rec.pos);
      if (&other_cell != &cell)
        BufferMaybeSendOffSample(other_cell, new_rec);
    }
}


SurfacePathGuiding::Record SurfacePathGuiding::ComputeStochasticFilterPosition(const Record & rec, const CellData &cd, Sampler &sampler)
{
  Record new_rec{rec};
  for (int axis = 0; axis < 3; ++axis)
  {
    new_rec.pos[axis] += 2.*Lerp(-cd.cell_size[axis], cd.cell_size[axis], sampler.Uniform01());
  }

  {
    const double prob = 0.1;
    const double pdf = vmf_fitting::Pdf(cd.mixture_sampled, rec.reverse_incident_dir);
    const double a = -prob/(pdf*2.*Pi + 1.e-9) + 1.;
    const double scatter_open_angle = std::min(std::max(a, -1.), 1.);
    Double3 scatter_offset = SampleTrafo::ToUniformSphereSection(scatter_open_angle, sampler.UniformUnitSquare());
    new_rec.reverse_incident_dir = OrthogonalSystemZAligned(rec.reverse_incident_dir) * scatter_offset.cast<float>();
  }

  new_rec.is_original = false;
  return new_rec;
}


void SurfacePathGuiding::ProcessSamples(int cell_idx)
{
  CellDataTemporary& cdtmp = cell_data_temp[cell_idx];
  CellData& cd = *recording_tree.GetData()[cell_idx];

  while (true)
  {    
    bool has_buffer = false;
    Span<IncidentRadiance> buffer;
    {
      tbb::spin_mutex::scoped_lock l{ cdtmp.mutex };
      if (!cdtmp.data_chunks.empty())
      {
        buffer = cdtmp.data_chunks.back();
        cdtmp.data_chunks.pop_back();
        has_buffer = true;
      }
      else
      {
        // Make known that this task will not process any more samples.
        cdtmp.fit_task_submissions_count--;
      }
    }

    if (!has_buffer)
      break;

    num_recorded_samples.fetch_and_add(buffer.size());

    LearnIncidentRadianceIn(cd, buffer);

    FreeSpan(buffer);
  }
}


void SurfacePathGuiding::LearnIncidentRadianceIn(CellData &cell, Span<IncidentRadiance> buffer)
{
  using namespace vmf_fitting;

  incremental::Params params;
  params.maximization_step_every = param_em_every;
  params.prior_alpha = param_prior_strength;
  params.prior_nu = param_prior_strength;
  params.prior_tau = 0.1*param_prior_strength;
  params.prior_mode = &cell.mixture_sampled;

  for (const auto &in : buffer)
  {
    //if (in.is_original)
    cell.leaf_stats += in.pos;
    //else
    //if (in.is_original)
    {
      float weight = in.weight;
      const auto xs = Span<const Eigen::Vector3f>(&in.reverse_incident_dir, 1);
      const auto ws = Span<const float>(&weight, 1);
      // TODO: Careful with false sharing here! 
      incremental::Fit(cell.mixture_learned, cell.fitdata, params, xs, ws);
    }
  }
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug[cell.num].Write(buffer);
#endif
}



void SurfacePathGuiding::BeginRound(Span<ThreadLocal*> thread_locals)
{
  const auto n = recording_tree.GetData().size();

  for (auto* tl : thread_locals)
  {
    tl->records_by_cells.resize(n);
    for (auto &sample_buffer : tl->records_by_cells)
    {
      assert(sample_buffer.size() == 0);
      sample_buffer.reserve(CELL_BUFFER_SIZE);
    }
  }
  cell_data_temp.reset(new CellDataTemporary[n]);
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug.reset(new CellDebug[n]);
  for (std::ptrdiff_t i = 0; i < n; ++i)
  {
    cell_data_debug[i].Open(
      strconcat(DEBUG_FILE_PREFIX, "round_", std::to_string(round), "_cell_", std::to_string(i), ".json")
    );
  }
#endif
}


void SurfacePathGuiding::FinalizeRound(Span<ThreadLocal*> thread_locals)
{
  for (auto* tl : thread_locals)
  {
    int cell_idx = 0;
    for (auto &sample_buffer : tl->records_by_cells)
    {
      Enqueue(cell_idx++, sample_buffer);
    }
  }
  
  the_task_arena->execute([&]() { the_task_group.wait(); } );

  auto celldatas = recording_tree.GetData();

  cell_data_temp.reset();

#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  std::for_each(cell_data_debug.get(), cell_data_debug.get() + celldatas.size(), [](auto& dbg) {
    dbg.Close();
  });
#endif
}


namespace 
{


void UpdateCellSizeFromBox(const kdtree::Node *node, Span<CellData* const> celldata, const Box &box)
{
  if (node->is_leaf)
  {
    const auto* leaf = static_cast<const kdtree::Leaf*>(node);
    auto cd = celldata[leaf->num];
    cd->cell_size = box.max - box.min;
  }
  else
  {
    const auto* branch = static_cast<const kdtree::Branch*>(node);
    {
      Box b{ box };
      b.max[branch->split_axis] = branch->split_pos;
      UpdateCellSizeFromBox(branch->left, celldata, b);
    }
    {
      Box b{ box };
      b.min[branch->split_axis] = branch->split_pos;
      UpdateCellSizeFromBox(branch->right, celldata, b);
    }
  }
}

}

void SurfacePathGuiding::PrepareAdaptedStructures()
{
  std::cout << strconcat("round ", round, ": num_samples=", num_recorded_samples, ", num_cells=", recording_tree.GetData().size()) << std::endl;

  {
    const std::uint64_t max_samples_per_cell = std::sqrt(num_recorded_samples.load()) * std::sqrt(param_num_initial_samples);
    kdtree::AdaptParams params{
      max_samples_per_cell, 0
    };
    kdtree::TreeAdaptor(params).AdaptInplace(recording_tree);
  }

  for (auto* cd : recording_tree.GetData())
  {
    cd->leaf_stats = LeafStatistics{};
    cd->fitdata = vmf_fitting::incremental::Data{};
    cd->mixture_sampled = cd->mixture_learned;
  }

  UpdateCellSizeFromBox(recording_tree.GetRoot(), recording_tree.GetData(), region);

  num_recorded_samples = 0;
  round++;
}

#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
CellDebug::~CellDebug()
{
  Close();
}

void CellDebug::Open(std::string filename_)
{
  this->filename = filename_;
  assert(!file.is_open());
  file.open(filename);
  assert(file.is_open());
  file << "[";
}

void CellDebug::Write(Span<const IncidentRadiance> records)
{
  using namespace rapidjson_util;
  assert(file.is_open());

  if (!records.size())
    return;

  rj::Document doc;
  doc.SetArray();
  auto &a = doc.GetAllocator();
  for (const auto &ir : records)
  {
    rj::Value jrec(rj::kObjectType);
    jrec.AddMember("dir", ToJSON(ir.reverse_incident_dir, a), a);
    jrec.AddMember("val", ToJSON(ir.weight, a), a);
    jrec.AddMember("proj", ToJSON(ir.xy, a), a);
    doc.PushBack(jrec.Move(), a);
  }
  auto datastr = ToString(doc);
  /*
  [
    {
      bla bla
    }
  ]
  */
  datastr = datastr.substr(2, datastr.size() - 4);
  if (!first)
    file << ',';
  else
    first = false;
  file << '\n' << datastr;
  file.flush();
}

void CellDebug::Close()
{
  if (file.is_open())
  {
    file << '\n' << ']';
    file.close();
  }
}

#endif


void SurfacePathGuiding::WriteDebugData()
{
#if (defined WRITE_DEBUG_OUT & defined HAVE_JSON & !defined NDEBUG)
  const auto filename = strconcat(DEBUG_FILE_PREFIX, "radiance_records_", round, ".json");
  std::ofstream file(filename);

  using namespace rapidjson_util;
  rj::Document doc;
  doc.SetObject();

  recording_tree.DumpTo(doc, doc);

  rj::Value records(rj::kArrayType);

  auto &a = doc.GetAllocator();

  for (const auto *cd : recording_tree.GetData())
  {
    //std::cout << "cell " << idx << " contains" << celldata.incident_radiance.size() << " records " << std::endl;
    rj::Value jcell(rj::kObjectType);
    jcell.AddMember("id", reinterpret_cast<std::size_t>(cd), a);
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
    jcell.AddMember("filename", ToJSON(cell_data_debug[cd->num].GetFilename(), a), a);
#endif
    auto[mean, stddev] = [&]() {
      if (cd->leaf_stats.Count())
      {
        return std::make_pair(
          cd->leaf_stats.Mean(),
          cd->leaf_stats.Stddev());
      }
      else
      {
        return std::make_pair(Eigen::Array3d::Zero().eval(), Eigen::Array3d::Zero().eval());
      }
    }();
    jcell.AddMember("point_distribution_statistics_mean", ToJSON(mean, a), a);
    jcell.AddMember("point_distribution_statistics_stddev", ToJSON(stddev, a), a);
    jcell.AddMember("size", ToJSON(cd->cell_size, a), a);
    jcell.AddMember("num_points", cd->leaf_stats.Count(), a);
    jcell.AddMember("average_weight", cd->fitdata.avg_weights, a);

    jcell.AddMember("mixture_learned", ToJSON(cd->mixture_learned, a), a);
    jcell.AddMember("mixture_sampled", ToJSON(cd->mixture_sampled, a), a);

    records.PushBack(jcell.Move(), a);
  }

  doc.AddMember("records", records.Move(), a);

  auto datastr = ToString(doc);
  file << datastr;

#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug.reset();
#endif
#endif
}


namespace kdtree
{

#ifdef HAVE_JSON

namespace dump_rj_tree
{
using namespace rapidjson_util;
using Alloc = rj::Document::AllocatorType;

rj::Value Build(const Branch &branch, Alloc &alloc);
rj::Value Build(const Leaf &leaf, Alloc &alloc);

rj::Value Build(const Node &node, Alloc &alloc)
{
  if (node.is_leaf)
    return Build(static_cast<const Leaf&>(node), alloc);
  else
    return Build(static_cast<const Branch&>(node), alloc);
}

rj::Value Build(const Branch &branch, Alloc &alloc)
{
  rj::Value v_left = Build(*branch.left, alloc);
  rj::Value v_right = Build(*branch.right, alloc);
  rj::Value v{ rj::kObjectType };
  v.AddMember("kind", "branch", alloc);
  v.AddMember("split_axis", branch.split_axis, alloc);
  v.AddMember("split_pos", branch.split_pos, alloc);
  v.AddMember("left", v_left.Move(), alloc);
  v.AddMember("right", v_right.Move(), alloc);
  return v;
}

rj::Value Build(const Leaf &leaf, Alloc &alloc)
{
  rj::Value v(rj::kObjectType);
  v.AddMember("kind", "leaf", alloc);
  v.AddMember("id", reinterpret_cast<std::size_t>(&static_cast<const CellData&>(leaf)), alloc);
  return v;
}

}


void Tree::DumpTo(rapidjson::Document &doc, rapidjson::Value & parent) const
{
  using namespace rapidjson_util;
  auto &alloc = doc.GetAllocator();
  parent.AddMember("tree", dump_rj_tree::Build(*root, alloc), alloc);
}
#endif

} // namespace kdtree




} // namespace guiding