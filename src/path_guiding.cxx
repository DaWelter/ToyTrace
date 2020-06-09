#include "path_guiding.hxx"
#include "scene.hxx"

#ifdef HAVE_JSON
#include "json.hxx"
#include "rapidjson/document.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/writer.h"
#endif
#include <fstream>

#include <tbb/parallel_for_each.h>
#include <Eigen/Eigenvalues>

//#include <range/v3/view/enumerate.hpp>

#ifdef _MSC_VER
static const char* DEBUG_FILE_PREFIX = "D:\\tmp2\\";
#else
static const char* DEBUG_FILE_PREFIX = "/tmp/";
#endif




#ifdef HAVE_JSON
namespace rapidjson_util {

rapidjson::Value ToJSON(const vmf_fitting::VonMisesFischerMixture<> &mixture, Alloc &a)
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

// The buffers use quite a lot of memory.
// Because there are num_threads*num_cells*buffer_size*sizeof(record),
// which adds up ...
static constexpr size_t CELL_BUFFER_SIZE = 32;


PathGuiding::PathGuiding(const Box &region, double cellwidth, const RenderingParameters &params, tbb::task_arena &the_task_arena, const char* name) :
    region{ region },
    recording_tree{},
    name{name},
    param_num_initial_samples{ params.guiding_tree_subdivision_factor },
    param_em_every{ params.guiding_em_every },
    param_prior_strength{ params.guiding_prior_strength },
    the_task_arena{ &the_task_arena }
{
  auto& cell = cell_data.emplace_back();
  vmf_fitting::InitializeForUnitSphere(cell.current_estimate.radiance_distribution);
  vmf_fitting::InitializeForUnitSphere(cell.learned.radiance_distribution);
  cell.index = 0;
  //cell.current_estimate.cell_size = region.max - region.min;
  cell.current_estimate.cell_bbox = region;
  cell.current_estimate.incident_flux_density = 1.;
}


CellData& PathGuiding::LookupCellData(const Double3 &p)
{
  return cell_data[recording_tree.Lookup(p)];
}


const PathGuiding::RadianceEstimate& PathGuiding::FindRadianceEstimate(const Double3 &p) const
{
    return cell_data[recording_tree.Lookup(p)].current_estimate;
}


void  PathGuiding::Enqueue(int cell_num, ToyVector<Record> &sample_buffer)
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


void PathGuiding::AddSample(
  ThreadLocal& tl, const Double3 &pos,
  Sampler &sampler, const Double3 &reverse_incident_dir, const Spectral3 &radiance)
{
    auto BufferMaybeSendOffSample = [&tl,this](const CellData &cell, const Record &rec)
    {
      auto& sample_buffer = tl.records_by_cells[cell.index];
      sample_buffer.push_back(rec);

      if (sample_buffer.size() >= CELL_BUFFER_SIZE)
      {
        Enqueue(cell.index, sample_buffer);
      }
    };

    auto rec = Record{
            pos,
            reverse_incident_dir.cast<float>(),
            radiance.cast<float>().mean(),
    };

    auto& cell = LookupCellData(pos);
    BufferMaybeSendOffSample(cell, rec);

    const auto new_rec = ComputeStochasticFilterPosition(rec, cell, sampler);
    auto& other_cell = LookupCellData(new_rec.pos);
    if (&other_cell.index != &cell.index)
      BufferMaybeSendOffSample(other_cell, new_rec);
}


PathGuiding::Record PathGuiding::ComputeStochasticFilterPosition(const Record & rec, const CellData &cd, Sampler &sampler)
{
  Record new_rec{rec};

  const Double3 rv{ sampler.Uniform01() , sampler.Uniform01() , sampler.Uniform01() };

  // Note: Stddev of uniform distribution over [0, 1] is ca 0.3. To scatter samples across this range 
  //       I multiply the size of the axes by 1./stddev * 1/2 which is the 3/2rd factor here. The 1/2
  //       is there because the frame is centered in the middle, so the axes lengths need to be the "radius"
  //       of the point cloud, not the diameter.
  //       Other factors are simply ad hoc tuning parameters
  static constexpr double MAGIC = 2.;
  new_rec.pos += cd.current_estimate.points_cov_frame * (rv*2. - Double3::Ones()) * (MAGIC * 3. / 2.);

  {
    const float prob = 0.1f;
    const float pdf = vmf_fitting::Pdf(cd.current_estimate.radiance_distribution, rec.reverse_incident_dir);
    const float a = -prob/(pdf*float(2.*Pi) + 1.e-9f) + 1.f;
    const float scatter_open_angle = std::min(std::max(a, -1.f), 1.f);
    Double3 scatter_offset = SampleTrafo::ToUniformSphereSection(scatter_open_angle, sampler.UniformUnitSquare());
    new_rec.reverse_incident_dir = OrthogonalSystemZAligned(rec.reverse_incident_dir) * scatter_offset.cast<float>();
  }

  new_rec.is_original = false;
  return new_rec;
}


void PathGuiding::ProcessSamples(int cell_idx)
{
  if (round <= 1)
    return;

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

    LearnIncidentRadianceIn(cd, buffer);
    FreeSpan(buffer);
  }
}


void PathGuiding::LearnIncidentRadianceIn(CellData &cell, Span<IncidentRadiance> buffer)
{
  using namespace vmf_fitting;
  // TODO: should be the max over all previous samples counts to prevent
  // overfitting in case the sample count decreases.
  double prior_strength = std::max(1., 0.001*cell.max_num_samples);
  incremental::Params<> params;
  params.prior_alpha = prior_strength;
  params.prior_nu = prior_strength;
  params.prior_tau = prior_strength;
  params.maximization_step_every = std::max(1, static_cast<int>(prior_strength)*50); //param_em_every;
  //params.prior_mode = &cell.current_estimate.radiance_distribution;

  // Below, I provide a fixed, more or less uniform prior. This seems to yield
  // higher variance than taking the previouly learned distribution as prior.
  vmf_fitting::VonMisesFischerMixture<> prior_mode;
  vmf_fitting::InitializeForUnitSphere(prior_mode);
  params.prior_mode = &prior_mode;

  for (const auto &in : buffer)
  {
    cell.learned.incident_flux_density_accum += in.weight;

    if (in.weight <= 0.)
      continue;

    if (in.is_original)
      cell.learned.leaf_stats += in.pos;
    //else
    //if (in.is_original)
    {
      float weight = in.weight;
      const auto xs = Span<const Eigen::Vector3f>(&in.reverse_incident_dir, 1);
      const auto ws = Span<const float>(&weight, 1);
      incremental::Fit(cell.learned.radiance_distribution, cell.learned.fitdata, params, xs, ws);
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
      cell_data_debug[cell.index].Write(in.pos, in.reverse_incident_dir, in.weight);
#endif
    }
  }

#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug[cell.index].params = params;
#endif
}


void PathGuiding::BeginRound(Span<ThreadLocal*> thread_locals)
{
  round++;

  const auto n = recording_tree.NumLeafs();
  assert(n == cell_data.size());

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
      strconcat(DEBUG_FILE_PREFIX, name, "_samples_round_", std::to_string(round), "_cell_", std::to_string(i), ".csv")
    );
  }
#endif

  long mem_thread_local_buffers = thread_locals.size()*(sizeof(ThreadLocal)+n*(sizeof(RecordBuffer)+CELL_BUFFER_SIZE*sizeof(Record)));
  long mem_cell_data_temp = n*sizeof(CellDataTemporary);
  long mem_cell_data = n*sizeof(CellData);
  std::cout << "--- expected mem use [MB] ----" << std::endl;
  std::cout << "Thread local buffer " << mem_thread_local_buffers/(1024*1024) << std::endl;
  std::cout << "Cell data temp " << mem_cell_data_temp/(1024*1024) << std::endl;
  std::cout << "Cell data " << mem_cell_data/(1024*1024) << std::endl;
}


void PathGuiding::FinalizeRound(Span<ThreadLocal*> thread_locals)
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

  // Release memory
  for (auto* tl : thread_locals)
    decltype(tl->records_by_cells){}.swap(tl->records_by_cells);

  if (round <= 1)
  {
    the_task_arena->execute([this]() {
      the_task_group.run_and_wait([this]() {
        AdaptInitial();
      });
    });
  }

  WriteDebugData();

  cell_data_temp.reset();

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
    //out[node.idx].current_estimate.cell_size = (node_box.max - node_box.min);
    out[node.idx].current_estimate.cell_bbox = node_box;
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

namespace
{

void InitializePcaFrame(CellData::CurrentEstimate &ce, const CellData::Learned &learned)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver{ learned.leaf_stats.Cov() };
  // Note: Eigenvalues are the variances w.r.t. the eigenvector frame.
  //       Take sqrt to obtain stddev.
  //       Then  scale axes of the frame so that computation of points for the stochastic filtering only needs the matrix multiplication.
  ce.points_cov_frame = eigensolver.eigenvectors() * eigensolver.eigenvalues().cwiseSqrt().asDiagonal();
  ce.points_mean = learned.leaf_stats.Mean();
  ce.points_stddev = learned.leaf_stats.Var().cwiseSqrt();

  assert(learned.leaf_stats.Count() > 1);
  assert(learned.leaf_stats.Cov().allFinite());
  assert(ce.points_cov_frame.allFinite());
  assert(ce.points_mean.allFinite());
}

}



void PathGuiding::PrepareAdaptedStructures()
{
  if (round > 1)
    AdaptIncremental();
}


void PathGuiding::AdaptInitial()
{
  // There is only one cell, and all the samples are stored in its CellDataTemporary struct.
  assert(cell_data_temp.get() && cell_data.size() == 1);

  ToyVector<Span<IncidentRadiance>> sample_chunks = std::move(cell_data_temp[0].data_chunks);
  
  // For easier access. Also freeing the memory, which I have to eventually.
  ToyVector<IncidentRadiance> samples; samples.reserve(sample_chunks.size() * CELL_BUFFER_SIZE);
  for (auto &span : sample_chunks)
  {
    samples.insert(samples.end(), span.begin(), span.end());
    FreeSpan(span);
  }

  // Probably this is taking samples from media interaction, in a scene without media.
  if (samples.empty())
    return;

  ToyVector<Span<IncidentRadiance>>{}.swap(sample_chunks);

  auto builder = kdtree::MakeBuilder<IncidentRadiance>(/*max_depth*/ MAX_DEPTH, /*min_num_points*/ param_num_initial_samples, [](const IncidentRadiance &s) { return s.pos; });
  recording_tree = builder.Build(AsSpan(samples));

  cell_data.resize(recording_tree.NumLeafs());

  std::cout << "Fitting to " << samples.size() << " initial samples in " << cell_data.size() << " cells ..." << std::endl;

  tbb::parallel_for(0, isize(cell_data), [this, &builder](int i)
  {
      CellData& cd = cell_data[i];
      Span<IncidentRadiance> cell_samples = builder.DataRangeOfLeaf(i);
      assert(cell_samples.size() > 0);

      cd.index = i;
      cd.last_num_samples = cell_samples.size();
      cd.max_num_samples = cell_samples.size();
      vmf_fitting::InitializeForUnitSphere(cd.current_estimate.radiance_distribution);
      vmf_fitting::InitializeForUnitSphere(cd.learned.radiance_distribution);

      LearnIncidentRadianceIn(cd, cell_samples);

      cd.current_estimate.radiance_distribution = cd.learned.radiance_distribution;
      cd.current_estimate.incident_flux_density = cd.learned.incident_flux_density_accum.Mean();
      cd.current_estimate.incident_flux_confidence_bounds = cd.learned.incident_flux_density_accum.Count() >= 10 ?
        std::sqrt(cd.learned.incident_flux_density_accum.Var() / static_cast<double>(cd.learned.incident_flux_density_accum.Count())) : LargeNumber;
      assert(std::isfinite(cd.current_estimate.incident_flux_confidence_bounds));

      InitializePcaFrame(cd.current_estimate, cd.learned);
  });

  ComputeLeafBoxes(recording_tree, recording_tree.GetRoot(), region, AsSpan(cell_data));

  previous_max_samples_per_cell = param_num_initial_samples;
  previous_total_samples = samples.size();

  std::cout << "round " << round << ", num cells = " << cell_data.size() << std::endl;
  const double avg_sample_count = static_cast<double>(std::accumulate(cell_data.begin(), cell_data.end(), 0l, [](const long &a, const CellData &b) { return a+b.last_num_samples; })) / cell_data.size();
  std::cout << "  avg sample count = " << avg_sample_count << std::endl;
}


void PathGuiding::AdaptIncremental()
{
  const std::int64_t num_fit_samples = std::accumulate(cell_data.begin(), cell_data.end(), 0l, [](std::int64_t n, const CellData &cd) {
    return n + cd.learned.leaf_stats.Count();
  });
  // Start off with 100 samples per cell = nc.
  // mc = sqrt(N)*c
  // mc_0 = sqrt(N0)*c = nc
  // => c = nc / sqrt(N0)

  // mc = sqrt(M)*c
  // nc = sqrt(N)*c
  // => nc = mc/sqrt(M)*sqrt(N)

  const std::uint64_t max_samples_per_cell = num_fit_samples
    ? std::sqrt(num_fit_samples)*previous_max_samples_per_cell/std::sqrt(previous_total_samples) 
    : previous_max_samples_per_cell;


  std::cout << strconcat(
    "round ", round,
    ": num_samples=", num_fit_samples,
    ", current num_cells=", recording_tree.NumLeafs(),
    ", target samples per cell=", max_samples_per_cell) << std::endl;

  { // Start tree adaptation
    auto DetermineSplit = [this, max_samples_per_cell](int cell_idx) -> std::pair<int, double>
    {
      // If too many samples fell into this cell -> split it at the sample's centroid.
      const auto& stats = cell_data[cell_idx].learned.leaf_stats;
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

    decltype(cell_data) new_data(recording_tree.NumLeafs());

    int num_cell_over_2x_limit = 0;
    int num_cell_over_1_5x_limit = 0;
    int num_cell_over1x_limit = 0;
    int num_cell_else = 0;
    int num_cell_sample_count_regressed = 0;
    double kl_divergence = 0.;
    const double logq = std::log2(1. / isize(cell_data));

    // This should initialize all sampling mixtures with
    // the previously learned ones. If node is split, assignment
    // is done to both children.
    int i = 0;
    for (const auto& m : adaptor.GetNodeMappings())
    {
      const bool is_split = (m.new_first >= 0) && (m.new_second >= 0);

      auto CopyCell = [&src = cell_data[i], new_data = AsSpan(new_data)](int dst_idx, bool is_split) mutable
      {
        CellData &dst = new_data[dst_idx];
        dst.index = dst_idx;

        dst.current_estimate.radiance_distribution = src.learned.radiance_distribution;        
        dst.current_estimate.incident_flux_density = src.learned.incident_flux_density_accum.Mean();
        dst.current_estimate.incident_flux_confidence_bounds = src.learned.incident_flux_density_accum.Count() >= 10 ?
          std::sqrt(src.learned.incident_flux_density_accum.Var() / static_cast<double>(src.learned.incident_flux_density_accum.Count())) : LargeNumber;
        assert(std::isfinite(dst.current_estimate.incident_flux_confidence_bounds));

        // Copy over point & radiance statistics from old cell.
        dst.learned.radiance_distribution = src.learned.radiance_distribution;
        dst.learned.incident_flux_density_accum = src.learned.incident_flux_density_accum;
        
        if (src.learned.leaf_stats.Count() >= 10)
        {
          InitializePcaFrame(dst.current_estimate, src.learned);
        }
        else
        {
          dst.current_estimate.points_cov_frame = src.current_estimate.points_cov_frame;
          dst.current_estimate.points_mean = src.current_estimate.points_mean;
          dst.current_estimate.points_stddev = src.current_estimate.points_stddev;
        }

        dst.last_num_samples = src.learned.leaf_stats.Count();
        dst.max_num_samples = std::max(dst.last_num_samples, dst.max_num_samples);
      };

      if (m.new_first >= 0)
      {
        CopyCell(m.new_first, is_split);
      }
      if (m.new_second >= 0)
      {
        CopyCell(m.new_second, is_split);
      }

      const int num_samples = cell_data[i].learned.leaf_stats.Count();
      if (num_samples > 2 * max_samples_per_cell)
        ++num_cell_over_2x_limit;
      else if (2 * num_samples > 3 * max_samples_per_cell)
        ++num_cell_over_1_5x_limit;
      else if (num_samples > max_samples_per_cell)
        ++num_cell_over1x_limit;
      else if (!is_split && num_samples * 3 < cell_data[i].last_num_samples * 4)
        ++num_cell_sample_count_regressed;
      else
        ++num_cell_else;
      const double p = static_cast<double>(num_samples) / num_fit_samples;
      kl_divergence += p > 0. ? (p*std::log2(p) - p * logq) : 0.;
      ++i;
    }

    cell_data = std::move(new_data);

    previous_max_samples_per_cell = max_samples_per_cell;
    previous_total_samples = num_fit_samples;

    std::cout << strconcat(
      "num_cell_over_2x_limit = ", num_cell_over_2x_limit, "\n",
      "num_cell_over_1_5x_limit = ", num_cell_over_1_5x_limit, "\n",
      "num_cell_over1x_limit = ", num_cell_over1x_limit, "\n",
      "num_cell_else = ", num_cell_else, "\n",
      "num_cell_regressed = ", num_cell_sample_count_regressed, "\n",
      "kl_divergence = ", kl_divergence) << std::endl;
    ++i;
  } // End tree adaption

  ComputeLeafBoxes(recording_tree, recording_tree.GetRoot(), region, AsSpan(cell_data));
}


#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
CellDebug::~CellDebug()
{
  Close();
}

void CellDebug::Open(std::string filename_)
{
  this->filename = filename_;
  std::cout << "opening " << this->filename << std::endl;
  assert(!file.is_open());
  file.open(filename);
  assert(file.is_open());
}

void CellDebug::Write(const Double3 &pos, const Float3 &dir, float weight)
{
  assert(file.is_open());
  file << pos[0] << "," << pos[1] << "," << pos[2] << ","<< dir[0] << ","<< dir[1] << ","<< dir[2] << ","<< weight << "\n";
}

void CellDebug::Close()
{
  std::cout << "closing " << this->filename << std::endl;
  file.close();
}

#endif


void PathGuiding::WriteDebugData()
{
#if (defined WRITE_DEBUG_OUT & defined HAVE_JSON & !defined NDEBUG)
  const auto filename = strconcat(DEBUG_FILE_PREFIX, name, "_radiance_records_", round, ".json");
  std::cout << "Writing " << filename << std::endl;

  using namespace rapidjson_util;
  rj::Document doc;
  doc.SetObject();

  recording_tree.DumpTo(doc, doc);

  rj::Value records(rj::kArrayType);

  auto &a = doc.GetAllocator();

  for (const auto &cd : cell_data)
  {
    //std::cout << "cell " << idx << " contains" << celldata.incident_radiance.size() << " records " << std::endl;
    rj::Value jcell(rj::kObjectType);
    jcell.AddMember("id", cd.index, a);

    if (cd.learned.leaf_stats.Count()>0)
    {
      assert(cd.learned.leaf_stats.Cov().allFinite());
      assert(cd.current_estimate.points_cov_frame.allFinite());
      assert(cd.current_estimate.points_mean.allFinite());
      jcell.AddMember("point_distribution_mean", ToJSON(cd.current_estimate.points_mean, a), a);
      jcell.AddMember("point_distribution_frame", ToJSON(cd.current_estimate.points_cov_frame, a), a);
      jcell.AddMember("point_distribution_stddev", ToJSON(cd.current_estimate.points_stddev, a), a);
    }
    else
    {
      jcell.AddMember("point_distribution_mean", ToJSON(Eigen::Vector3d::Zero().eval(), a), a);
      jcell.AddMember("point_distribution_frame", ToJSON(Eigen::Matrix3d::Zero().eval(), a), a);
      jcell.AddMember("point_distribution_stddev", ToJSON(Eigen::Vector3d::Zero().eval(), a), a);
    }
    jcell.AddMember("bbox_min", ToJSON(cd.current_estimate.cell_bbox.min, a), a);
    jcell.AddMember("bbox_max", ToJSON(cd.current_estimate.cell_bbox.max, a), a);
    jcell.AddMember("num_points", cd.learned.leaf_stats.Count(), a);
    jcell.AddMember("average_weight", cd.learned.fitdata.avg_weights, a);
    jcell.AddMember("incident_flux_learned", ToJSON(cd.learned.incident_flux_density_accum.Mean(), a), a);
    jcell.AddMember("incident_flux_sampled", ToJSON(cd.current_estimate.incident_flux_density, a), a);
    jcell.AddMember("mixture_learned", ToJSON(cd.learned.radiance_distribution, a), a);
    jcell.AddMember("mixture_sampled", ToJSON(cd.current_estimate.radiance_distribution, a), a);

    // Fit parameters 
#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
    jcell.AddMember("fitparam_prior_nu", cell_data_debug[cd.index].params.prior_nu, a);
    jcell.AddMember("fitparam_prior_tau", cell_data_debug[cd.index].params.prior_tau, a);
    jcell.AddMember("fitparam_prior_alpha", cell_data_debug[cd.index].params.prior_alpha, a);
    jcell.AddMember("fitparam_maximization_step_every", cell_data_debug[cd.index].params.maximization_step_every, a);
#endif

    records.PushBack(jcell.Move(), a);
  }

  assert(records.Size() == cell_data.size());

  doc.AddMember("records", records.Move(), a);

  {
    std::ofstream file(filename);
    rapidjson::OStreamWrapper osw(file);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
    doc.Accept(writer);
  }

  //auto datastr = ToString(doc);
  //file << datastr;

#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug.reset();
#endif
#endif
}

} // namespace guiding