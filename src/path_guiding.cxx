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


SurfacePathGuiding::SurfacePathGuiding(const Box &region, double cellwidth, const RenderingParameters &params, tbb::task_arena &the_task_arena) :
    region{ region },
    recording_tree{},
    param_num_initial_samples{ params.guiding_tree_subdivision_factor },
    param_em_every{ params.guiding_em_every },
    param_prior_strength{ params.guiding_prior_strength },
    the_task_arena{ &the_task_arena }
{
  auto& cell = cell_data.emplace_back();
  vmf_fitting::InitializeForUnitSphere(cell.mixture_learned);
  vmf_fitting::InitializeForUnitSphere(cell.mixture_sampled);
}


CellData& SurfacePathGuiding::LookupCellData(const Double3 &p)
{
  return cell_data[recording_tree.Lookup(p)];
}


const vmf_fitting::VonMisesFischerMixture* SurfacePathGuiding::FindSamplingMixture(const Double3 &p) const
{
    return &cell_data[recording_tree.Lookup(p)].mixture_sampled;
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

    auto& cell = LookupCellData(surface.pos);
    BufferMaybeSendOffSample(cell, rec);

    for (int i = 0; i < 1; ++i)
    {
      const auto new_rec = ComputeStochasticFilterPosition(rec, cell, sampler);

      auto& other_cell = LookupCellData(new_rec.pos);
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
  CellData& cd = cell_data[cell_idx];

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
      incremental::Fit(cell.mixture_learned, cell.fitdata, params, xs, ws);
    }
  }
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug[cell.num].Write(buffer);
#endif
}



void SurfacePathGuiding::BeginRound(Span<ThreadLocal*> thread_locals)
{
  const auto n = recording_tree.NumLeafs();

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

  cell_data_temp.reset();

#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  std::for_each(cell_data_debug.get(), cell_data_debug.get() + celldatas.size(), [](auto& dbg) {
    dbg.Close();
  });
#endif
}

namespace 
{

using namespace ::guiding::kdtree;

std::pair<Box, Box> ChildrenBoxes(const Tree &tree, Handle node, const Box &node_box)
{
  auto [axis, pos] = tree.Split(node);

  Box leftbox{ node_box };
  leftbox.max[axis] = pos;
  

  Box rightbox{ node_box };
  rightbox.min[axis] = pos;
  
  return std::make_pair(leftbox, rightbox);
}

void ComputeLeafBoxes(const Tree &tree, Handle node, const Box &node_box, Span<CellData> out)
{
  if (node.is_leaf)
  {
    // Assign lateral extents.
    out[node.idx].cell_size = (node_box.max - node_box.min);
  }
  else
  {
    auto [left, right] = tree.Children(node);
    auto [bl, br] = ChildrenBoxes(tree, node, node_box);
    ComputeLeafBoxes(tree, left, bl, out);
    ComputeLeafBoxes(tree, right, br, out);
  }
}

} // namespace


void SurfacePathGuiding::PrepareAdaptedStructures()
{
  std::cout << strconcat("round ", round, ": num_samples=", num_recorded_samples, ", num_cells=", recording_tree.NumLeafs()) << std::endl;

  { // Start tree adaptation
    const std::uint64_t max_samples_per_cell = std::sqrt(num_recorded_samples.load()) * std::sqrt(param_num_initial_samples);

    auto DetermineSplit = [this, max_samples_per_cell](int cell_idx) -> std::pair<int, double>
    {
      // If too many samples fell into this cell -> split it at the sample's centroid.
      const auto& stats = cell_data[cell_idx].leaf_stats;
      if (stats.Count() > max_samples_per_cell)
      {
        const auto mean = stats.Mean();
        const auto var = stats.Var();
        int axis = -1;
        var.maxCoeff(&axis);
        double pos = mean[axis];
        return { axis, pos };
      }
      else
      {
        return { -1, NaN };
      }
    };

    kdtree::TreeAdaptor adaptor(DetermineSplit);
    recording_tree = adaptor.Adapt(recording_tree);
  
    ToyVector<CellData> new_data(recording_tree.NumLeafs());

    {
      int i =0;  
      for (auto &cd : new_data)
        cd.num = i++;
    }

    // This should initialize all sampling mixtures with
    // the previously learned ones. If node is split, assignment
    // is done to both children.
    int i = 0;
    for (auto m : adaptor.GetNodeMappings())
    {
      if (m.new_first >=0)
      {
        new_data[m.new_first].mixture_sampled = cell_data[i].mixture_learned;
        new_data[m.new_first].mixture_learned = cell_data[i].mixture_learned;
      }
      if (m.new_second >= 0)
      {
        new_data[m.new_second].mixture_sampled = cell_data[i].mixture_learned;
        new_data[m.new_second].mixture_learned = cell_data[i].mixture_learned;
      }
      ++i;
    }

    cell_data = std::move(new_data);
  } // End tree adaption

  ComputeLeafBoxes(recording_tree, recording_tree.GetRoot(), region, AsSpan(cell_data));

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

  int cell_idx = 0;
  for (const auto cd : cell_data)
  {
    //std::cout << "cell " << idx << " contains" << celldata.incident_radiance.size() << " records " << std::endl;
    rj::Value jcell(rj::kObjectType);
    jcell.AddMember("id", cell_idx, a);
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
    jcell.AddMember("filename", ToJSON(cell_data_debug[cd->num].GetFilename(), a), a);
#endif
    auto[mean, stddev] = [&]() {
      if (cd.leaf_stats.Count())
      {
        return std::make_pair(
          cd.leaf_stats.Mean(),
          cd.leaf_stats.Stddev());
      }
      else
      {
        return std::make_pair(Eigen::Array3d::Zero().eval(), Eigen::Array3d::Zero().eval());
      }
    }();
    jcell.AddMember("point_distribution_statistics_mean", ToJSON(mean, a), a);
    jcell.AddMember("point_distribution_statistics_stddev", ToJSON(stddev, a), a);
    jcell.AddMember("size", ToJSON(cd.cell_size, a), a);
    jcell.AddMember("num_points", cd.leaf_stats.Count(), a);
    jcell.AddMember("average_weight", cd.fitdata.avg_weights, a);

    jcell.AddMember("mixture_learned", ToJSON(cd.mixture_learned, a), a);
    jcell.AddMember("mixture_sampled", ToJSON(cd.mixture_sampled, a), a);

    records.PushBack(jcell.Move(), a);
    ++cell_idx;
  }

  doc.AddMember("records", records.Move(), a);

  auto datastr = ToString(doc);
  file << datastr;

#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug.reset();
#endif
#endif
}

} // namespace guiding