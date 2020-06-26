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
#include <tbb/combinable.h>
#include <tbb/enumerable_thread_specific.h>
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


void PathGuiding::AddSample(
  ThreadLocal& tl, const Double3 &pos,
  Sampler &sampler, const Double3 &reverse_incident_dir, const Spectral3 &radiance)
{
    auto rec = IncidentRadiance{
            pos,
            reverse_incident_dir.cast<float>(),
            radiance.cast<float>().mean(),
    };

    tl.samples.push_back(rec);
}


IncidentRadiance PathGuiding::ComputeStochasticFilterPosition(const IncidentRadiance & rec, const CellData &cd, Sampler &sampler)
{
  IncidentRadiance new_rec{rec};

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

  //new_rec.is_original = false;
  return new_rec;
}


ToyVector<int> PathGuiding::ComputeCellIndices(Span<const IncidentRadiance> samples) const
{
  ToyVector<int> cell_indices(samples.size(), -1);
  tbb::parallel_for(tbb::blocked_range<long>(0, samples.size(), 1000), [&,this](tbb::blocked_range<long> r)
  {
    for (long i=r.begin(); i<r.end(); ++i)
    {
      const int cell = recording_tree.Lookup(samples[i].pos);
      cell_indices[i] = cell;
    }
  });
  return cell_indices;
}


ToyVector<ToyVector<IncidentRadiance>> PathGuiding::SortSamplesIntoCells(Span<const int> cell_indices, Span<const IncidentRadiance> samples) const
{
  ToyVector<long> cell_sample_count(cell_data.size(), 0);
  for (int i : cell_indices)
      ++cell_sample_count[i];
  
  ToyVector<ToyVector<IncidentRadiance>> sorted_samples(cell_data.size());
  for (int i=0; i<cell_data.size(); ++i)
    sorted_samples[i].reserve(cell_sample_count[i]);
  
  for (long i=0; i<samples.size(); ++i)
  {
    sorted_samples[cell_indices[i]].push_back(samples[i]);
  }

  return sorted_samples;
}


void PathGuiding::GenerateStochasticFilteredSamplesInplace(Span<int> cell_indices, Span<IncidentRadiance> samples) const
{
  tbb::enumerable_thread_specific<Sampler> tls_samplers;
  tbb::parallel_for(tbb::blocked_range<long>(0, samples.size(), 1000), [&, this](tbb::blocked_range<long> r)
  {
    auto sampler = tls_samplers.local();
    for (long i = r.begin(); i<r.end(); ++i)
    {
      const auto s = ComputeStochasticFilterPosition(samples[i], cell_data[cell_indices[i]], sampler);
      int new_cell = recording_tree.Lookup(s.pos);
      samples[i] = s;
      cell_indices[i] = new_cell;
    }
  });
}


// Also frees the tls sample buffer
// Updates the fit data in the CellData structs.
void PathGuiding::FitTheSamples(Span<ThreadLocal*> thread_locals)
{
  ToyVector<IncidentRadiance> samples;
  for (auto *tl : thread_locals)
  {
    samples.insert(samples.end(), tl->samples.begin(), tl->samples.end());
    ToyVector<IncidentRadiance>{}.swap(tl->samples);
  }

  auto cell_indices = ComputeCellIndices(AsSpan(samples));
  
  const auto noriginal = samples.size();
  samples.reserve(noriginal*2);
  cell_indices.reserve(noriginal*2);

  // This is only safe because the memory was reserved in advance!
  std::copy(samples.begin(), samples.begin()+noriginal, std::back_inserter(samples));
  std::copy(cell_indices.begin(), cell_indices.begin()+noriginal, std::back_inserter(cell_indices));
  
  GenerateStochasticFilteredSamplesInplace(
    Subspan(AsSpan(cell_indices), noriginal, noriginal), 
    Subspan(AsSpan(samples), noriginal, noriginal));

  auto sorted_samples = SortSamplesIntoCells(AsSpan(cell_indices), AsSpan(samples));

  tbb::enumerable_thread_specific<Sampler> samplers;

  tbb::parallel_for<int>(0, isize(cell_data), [&, this](int cell_idx) 
  {
    auto samples = AsSpan(sorted_samples[cell_idx]);
    if (samples.size())
    {    
      RandomShuffle(samples.begin(), samples.end(), samplers.local());
      FitTheSamples(cell_data[cell_idx], samples);
    }
  });
}


void PathGuiding::FitTheSamples(CellData &cell, Span<IncidentRadiance> buffer, bool is_original) const
{
  using namespace vmf_fitting;
  // TODO: should be the max over all previous samples counts to prevent
  // overfitting in case the sample count decreases.
  double prior_strength = std::max(1., 0.01*cell.max_num_samples);
  incremental::Params<> params;
  params.prior_alpha = prior_strength;
  params.prior_nu = prior_strength;
  params.prior_tau = prior_strength;
  params.maximization_step_every = std::max(1, static_cast<int>(prior_strength)*10); //param_em_every;
  //params.prior_mode = &cell.current_estimate.radiance_distribution;

  // Below, I provide a fixed, more or less uniform prior. This seems to yield
  // higher variance than taking the previouly learned distribution as prior.
  vmf_fitting::VonMisesFischerMixture<> prior_mode;
  vmf_fitting::InitializeForUnitSphere(prior_mode);
  params.prior_mode = &prior_mode;

  for (const auto &in : buffer)
  {
    cell.learned.incident_flux_density_accum += in.weight;

    if (is_original)
      cell.learned.leaf_stats += in.pos;

    if (in.weight <= 0.)
      continue;

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
    tl->samples.reserve(1024*1024);
  }


#ifdef PATH_GUIDING_WRITE_SAMPLES_ACTUALLY_ENABLED
  cell_data_debug.reset(new CellDebug[n]);
  for (std::ptrdiff_t i = 0; i < n; ++i)
  {
    cell_data_debug[i].Open(
      fmt::format("{}{}_samples_round_{}_cell_{}.csv", DEBUG_FILE_PREFIX, name, std::to_string(round), std::to_string(i))
    );
  }
#endif
}


void PathGuiding::FinalizeRound(Span<ThreadLocal*> thread_locals)
{
  the_task_arena->execute([&]() { the_task_group.wait(); } );

  long mem_cell_data = cell_data.size()*sizeof(CellData);
  long mem_tls_buffers = 0;
  for (auto* tl : thread_locals)
    mem_tls_buffers += tl->samples.size() + sizeof(decltype(*tl));

  if (round <= 1)
  {
    the_task_arena->execute([&, this]() {
      the_task_group.run_and_wait([&, this]() {
        AdaptInitial(thread_locals);
      });
    });
  }
  else
  {
    the_task_arena->execute([&, this]() {
      the_task_group.run_and_wait([&, this]() {
        FitTheSamples(thread_locals);
      });
    });
  }

  // std::cout << "--- expected mem use [MB] ----" << std::endl;
  // std::cout << "Thread local buffer " << mem_cell_data/(1024*1024) << std::endl;
  // std::cout << "Cell data " << mem_cell_data/(1024*1024) << std::endl;
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
  const Eigen::Vector3d evs = eigensolver.eigenvalues().cwiseMax(Epsilon);
  ce.points_cov_frame = eigensolver.eigenvectors() * evs.cwiseSqrt().asDiagonal();
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
  {
    WriteDebugData();
    AdaptIncremental();
  }
}


void PathGuiding::AdaptInitial(Span<ThreadLocal*> thread_locals)
{
  // There is only one cell, and all the samples are stored in its CellDataTemporary struct.
  assert(cell_data.size() == 1);

  // For easier access. Also freeing the memory, which I have to eventually.
  ToyVector<IncidentRadiance> samples; samples.reserve(1024*1024);
  for (auto* tls : thread_locals)
  {
    samples.insert(samples.end(), tls->samples.begin(), tls->samples.end());
    ToyVector<IncidentRadiance>{}.swap(tls->samples);
  }

  // Probably this is taking samples from media interaction, in a scene without media.
  if (samples.empty())
    return;

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

      FitTheSamples(cd, cell_samples);

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


  fmt::print(
    "round {}: num_samples={},\n current num_cells={},\n target samples per cell={}\n",
     round, num_fit_samples, recording_tree.NumLeafs(), max_samples_per_cell);

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
        dst.learned.fitdata = src.learned.fitdata;
        
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

    fmt::print(
      "num_cell_over_2x_limit = {}\n"
      "num_cell_over_1_5x_limit = {}\n"
      "num_cell_over1x_limit = {}\n"
      "num_cell_else = {}\n"
      "num_cell_regressed = {}\n"
      "kl_divergence = ", 
      num_cell_over_2x_limit, 
      num_cell_over_1_5x_limit, 
      num_cell_over1x_limit, 
      num_cell_else, 
      num_cell_sample_count_regressed, 
      kl_divergence);
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
  const auto filename = fmt::format("{}{}_radiance_records_{}.json",DEBUG_FILE_PREFIX, name, round);
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
    jcell.AddMember("average_weight", cd.learned.fitdata.avg_weights(), a);
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