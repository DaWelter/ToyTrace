#pragma once
#include "renderingalgorithms.hxx"

#include <unordered_map>
#include <boost/functional/hash.hpp>

#pragma GCC diagnostic warning "-Wunused-parameter" 
#pragma GCC diagnostic warning "-Wunused-variable"

namespace RandomWalk
{

// Ref: Veach's Thesis. Chpt. 10.
namespace BdptDetail
{
  
using NodeContainer = ToyVector<RW::PathNode>;
using PdfContainer  = ToyVector<Pdf>;
using SegmentContainer = ToyVector<RaySegment>;
using PathDensityContainer = ToyVector<double>;
using ConversionFactorContainer = ToyVector<double>;
using SpecularFlagContainer = ToyVector<bool>;
using WeightContainer = ToyVector<Spectral3>;

static constexpr int INDEX_OF_PICKING_INITIAL_POINT = -1;

class SubpathHistory
{
public:
  SubpathHistory(RadianceEstimatorBase& _randomwalk_functions)
    : randomwalk_functions(_randomwalk_functions)
  {
  }
  
  void Reset()
  {
    // Note: Allocation only on first use. clear() keeps the capacity up. However, incurs dtor calls.
    fwd_conversion_factors.clear();
    bwd_conversion_factors.clear();
    pdfs.clear();
    nodes.clear();
    betas.clear();
    segments.clear();
  }
  
  void Start(const RW::PathNode &start, const Spectral3 &_weight, Pdf _start_pdf)
  {
    // Bah ... copying :-(
    nodes.push_back(start);
    betas.push_back(_weight);
    pdfs.push_back(_start_pdf);
    segments.push_back(RaySegment{}); // For convenience
  }
  
  void AddSegment(const RW::PathNode &end_node, const Spectral3 &_weight, Pdf pdf_start_scatter, const RaySegment &segment)
  {   
    nodes.push_back(end_node);
    Spectral3  beta_at_node = betas.back()*_weight;
    betas.push_back(beta_at_node);
    pdfs.push_back(pdf_start_scatter);
    segments.push_back(segment);
  }
  
  void Finish()
  {
    fwd_conversion_factors.push_back(1.);
    bwd_conversion_factors.push_back(1.);
    // Fortunately I don't have to propagate this flag across connections because
    // connections transmit light/importance only through non-specular interactions.
     // Set to true/false at pdfs[1] at the emission event.
    bool is_parallel_beam = true;
    for (int i=1; i<nodes.size(); ++i)
    {
      is_parallel_beam &= pdfs[i].IsFromDelta();
      double fwd_factor = randomwalk_functions.PdfConversionFactorForTarget(Node(i-1), Node(i), segments[i], is_parallel_beam);
      fwd_conversion_factors.push_back(fwd_factor);
    }
    for (int i=1; i<nodes.size()-1; ++i) // Cannot handle last node because scatter pdf is unknown.
    {
      double bwd_factor = randomwalk_functions.PdfConversionFactorForTarget(Node(i), Node(i-1), segments[i], pdfs[i].IsFromDelta());
      bwd_conversion_factors.push_back(bwd_factor);
    }
    if (nodes.size() >= 2)
      bwd_conversion_factors.push_back(NaN); 
    fwd_conversion_factors.push_back(NaN); // Space for data of connection segment.
    pdfs.push_back(Pdf{});  // Dito.
    
    assert (fwd_conversion_factors.size() == nodes.size() + 1);
    assert (bwd_conversion_factors.size() == nodes.size());
    assert (pdfs.size() == nodes.size() + 1);
  }
  
  void Pop()
  {
    assert (!nodes.empty());
    nodes.pop_back();
    pdfs.pop_back();
    betas.pop_back();
    fwd_conversion_factors.pop_back();
    bwd_conversion_factors.pop_back();
    segments.pop_back();
  }
  
  int NumNodes() const
  {
    return nodes.size();
  }

  const RW::PathNode& Node(int i) const
  {
    return nodes[i];
  }
  
  const RW::PathNode& NodeFromBack(int i) const
  {
    return nodes[nodes.size()-i-1];
  }
  
  const Spectral3& Beta(int i) const
  {
    return betas[i];
  }
  
  bool IsSpecular(int i) const
  {
    return pdfs[i+1].IsFromDelta();
  }
  
  friend class BdptMis;
  friend class BackupAndReplace;
private:
  RadianceEstimatorBase& randomwalk_functions;
  ConversionFactorContainer fwd_conversion_factors;
  ConversionFactorContainer bwd_conversion_factors;
  ToyVector<RaySegment> segments;
  PdfContainer pdfs;
  NodeContainer nodes;
  WeightContainer betas;
};


struct Connection
{
  Pdf eye_pdf;
  Pdf light_pdf;
  int eye_index;
  int light_index;
  RaySegment segment;
};


// Think this is stupid? It is. But what should I do? There no hash support for pairs in the STL.
struct IntPairHash
{
  std::size_t operator()(const std::pair<int,int> &v) const
  {
    std::size_t seed = boost::hash_value(v.first);
    boost::hash_combine(seed, boost::hash_value(v.second));
    return seed;
  }
};
using DebugBuffers = std::unordered_map<std::pair<int,int>, Spectral3ImageBuffer, IntPairHash>;


class BackupAndReplace
{
  SubpathHistory &h;
  int node_index; 
  Pdf pdf;
  double fwd_conversion_factor;
  double bwd_conversion_factor;
  
  void InitBothPathsNonEmpty(const RW::PathNode &other_end_node, Pdf pdf_node_scatter, const RaySegment &segment)
  {
    assert (node_index >= 0 && other_end_node.node_type != RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK);
    pdf = h.pdfs[node_index+1];
    h.pdfs[node_index+1] = pdf_node_scatter; 

    fwd_conversion_factor = h.fwd_conversion_factors[node_index+1];
    bwd_conversion_factor = h.bwd_conversion_factors[node_index];
    h.fwd_conversion_factors[node_index+1] = h.randomwalk_functions.PdfConversionFactorForTarget(
      h.Node(node_index), other_end_node, segment, pdf_node_scatter.IsFromDelta());
    if (node_index > 0)
    {
      h.bwd_conversion_factors[node_index] = h.randomwalk_functions.PdfConversionFactorForTarget(
        h.Node(node_index-1), h.Node(node_index), h.segments[node_index], pdf_node_scatter.IsFromDelta());
    }
    else
    {
      assert (h.bwd_conversion_factors[node_index] == 1.);
    }
  }
  
  void InitOtherSideEmpty(double pdf_reverse_emission)
  {
    assert (node_index >= 1); // Because you cannot have a start node and simultaneously hit an emitter with it. Need at least two nodes.
    pdf = h.pdfs[node_index+1];
    fwd_conversion_factor = h.fwd_conversion_factors[node_index+1];
    bwd_conversion_factor = h.bwd_conversion_factors[node_index];
    
    h.pdfs[node_index+1] = pdf_reverse_emission;
    h.fwd_conversion_factors[node_index+1] = 1.;
    h.bwd_conversion_factors[node_index] = h.randomwalk_functions.PdfConversionFactorForTarget(
      h.Node(node_index), h.Node(node_index-1), h.segments[node_index], false);
  }
  
  void InitMySideEmpty(double other_node_creation_density)
  {
    assert (node_index == -1);
    h.pdfs[node_index+1] = other_node_creation_density;
    h.fwd_conversion_factors[node_index+1] = 1.;
  }
  
public:
  BackupAndReplace(const BackupAndReplace &) = delete;
  BackupAndReplace& operator=(const BackupAndReplace &) = delete;
  
  BackupAndReplace(SubpathHistory &_h, int _node_index, SubpathHistory &other_h, int other_node_index, Pdf pdf, const RaySegment &segment) 
    : h{_h}, node_index{_node_index} 
  {
    if (_node_index >= 0 && other_node_index >= 0)
      InitBothPathsNonEmpty(other_h.Node(other_node_index), pdf, segment);
    else if (_node_index >= 0)
      InitOtherSideEmpty(pdf);
    else
      InitMySideEmpty(pdf);
  }
  
  ~BackupAndReplace() 
  {
    h.pdfs[node_index+1] = pdf;
    h.fwd_conversion_factors[node_index+1] = fwd_conversion_factor;
    if (node_index >= 0)
      h.bwd_conversion_factors[node_index] = bwd_conversion_factor;
  }
};

  
class BdptMis
{
public:
  BdptMis(const Scene& scene)
  {
    one_over_number_of_splat_attempts_per_pixel = 1./(scene.GetCamera().xres*scene.GetCamera().yres);
  }
  
  double Compute(const SubpathHistory &eye, const SubpathHistory &light, const Connection &connection)
  {
    Reset(connection);
    ComputeSpecularFlags(eye, light, connection);
    ComputeForwardSubpathDensities(eye, light, connection.eye_index, connection.light_index, eye_densities);
    ComputeForwardSubpathDensities(light, eye, connection.light_index, connection.eye_index, light_densities);
    FactorInNumberWeight();
    double result = ComputePowerHeuristicWeight(connection);
    //double result = WeightByNumberOfAdmissibleTechniques();
    return result;
  }
private:
  SpecularFlagContainer specular_flags;
  PathDensityContainer eye_densities;
  PathDensityContainer light_densities;
  int total_node_count;
  double one_over_number_of_splat_attempts_per_pixel;
  static constexpr auto IDX_INITIAL = BdptDetail::INDEX_OF_PICKING_INITIAL_POINT;
    
  void Reset(const Connection &connection)
  {
    total_node_count = connection.eye_index+1+connection.light_index+1;
    specular_flags.clear();
    eye_densities.clear();
    light_densities.clear();
  }

  void ComputeSpecularFlags(const SubpathHistory &eye, const SubpathHistory &light, const Connection &connection)
  {   
    for (int i=0; i<=connection.eye_index+1; ++i)
      specular_flags.push_back(eye.pdfs[i].IsFromDelta());
    for (int i=connection.light_index+1; i>=0; --i)
      specular_flags.push_back(light.pdfs[i].IsFromDelta());
    assert(specular_flags.size() == total_node_count + 2);
  }
  
  void ComputeForwardSubpathDensities(const SubpathHistory &fwd_path, const SubpathHistory &reverse_path, 
                                      int idx_fwd, int idx_rev, PathDensityContainer &destination)
  {
    destination.push_back(Pdf{1.}); // When there are zero nodes on this subpath.
    for (int i=0; i<=idx_fwd; ++i)
    {
      destination.push_back(destination.back()*
        fwd_path.fwd_conversion_factors[i] * fwd_path.pdfs[i]);
    }
    if (idx_rev >= 0) // Handling the connection segment.
    {
      destination.push_back(destination.back()*
        fwd_path.fwd_conversion_factors[idx_fwd+1] * fwd_path.pdfs[idx_fwd+1]);
    }
    for (int i=idx_rev; i>=1; --i)
    {
      // Multiplying the scatter density at the node i with the conversion factor
      // leading in the reverse direction from node i.
      destination.push_back(destination.back()*
        reverse_path.bwd_conversion_factors[i] * reverse_path.pdfs[i+1]);
    }
    assert(destination.size() == total_node_count + 1);
  }

  double ComputePowerHeuristicWeight(const Connection &connection) const
  {
    /* Simple way to understand the number of admissible techniques in Veach's sense:
     * Imagine there is a very glossy vertex. Sampling via bsdf is low variance because the bsdf and the pdf cancel each other in the nominator/denominator.
     * Now, sampling something else and connecting to the vertex has high variance because only very few time we get a contribution, and if so, the bsdf has very large value because it is strongly peaked.
     * Now make the bsdf sharper, narrower, take limit to Dirac-Delta function. 
     * Variance goes to infinity. We cannot do this. And in fact, my 'Evaluate' & 'Pdf' functions return 0 for specular components, no matter what coordinates.
     * What is left are the sampling techniques where only non-specular vertices are connected. 
     * So I simply take the segments that have no adjacent specular vertices.
     */
    double nominator = eye_densities[connection.eye_index+1]*
                       light_densities[connection.light_index+1];
    nominator = Sqr(nominator);
    double denom = 0.;
    for (int s = 0; s <= total_node_count; ++s)
    {
      bool flag = !(specular_flags[s] || specular_flags[s+1]);
      double summand = flag ? (eye_densities[s]*light_densities[total_node_count-s]) : 0.;
      denom += Sqr(summand);
    }
    return nominator/denom;
  }
  
  void FactorInNumberWeight()
  {
    for (int i=2; i<=total_node_count; ++i)
    {
      eye_densities[i] *= one_over_number_of_splat_attempts_per_pixel;
    }
  }
  
  double WeightByNumberOfAdmissibleTechniques() const
  {
    int num_tech = 0;
    for (int i = 0; i <= total_node_count; ++i)
    {
      num_tech += !(specular_flags[i] || specular_flags[i+1]);
    }
    assert (num_tech >= 1); // Actually there can be purely specular paths that cannot be sampled.
    return 1./(num_tech + Epsilon);
  }
};


} // namespace BdptDetail

class Bdpt : public RadianceEstimatorBase, public IRenderingAlgo
{
  LambdaSelectionFactory lambda_selection_factory;
  ToyVector<IRenderingAlgo::SensorResponse> rgb_responses;

  struct Splat : public ROI::PointEmitterArray::Response
  {
    Splat(int unit_index, const Spectral3 &weight, int _path_length) :
      ROI::PointEmitterArray::Response{unit_index, weight}, path_length{_path_length} {}
    int path_length = {};
  };
  
  BdptDetail::SubpathHistory eye_history;
  BdptDetail::SubpathHistory direct_light_history; // The alternate history ;-) .Where a random light is sampled for s=*,t=1 paths, instead of the original one. Contains only a single node.
  BdptDetail::SubpathHistory light_history;
  BdptDetail::BdptMis bdptmis;
  Spectral3 total_eye_measurement_contributions;
  ToyVector<Splat> splats;
  Index3 lambda_idx;
  int pixel_index;
  
  bool enable_debug_buffers;
  BdptDetail::DebugBuffers debug_buffers;
  BdptDetail::DebugBuffers debug_buffers_mis;
  
public:
  Bdpt(const Scene &_scene, const AlgorithmParameters &algo_params) 
  : RadianceEstimatorBase{_scene, algo_params},
    lambda_selection_factory{},
    eye_history{*this},
    direct_light_history{*this},
    light_history{*this},
    bdptmis{_scene},
    enable_debug_buffers{algo_params.num_threads <= 0}
  {
    rgb_responses.reserve(1024);
  }

  
  RGB MakePrettyPixel(int _pixel_index) override
  {
    eye_history.Reset();
    light_history.Reset();
    total_eye_measurement_contributions = Spectral3{0.};
    splats.clear();
    
    pixel_index = _pixel_index;
    Spectral3 lambda_weights;
    std::tie(this->lambda_idx, lambda_weights) = lambda_selection_factory.WithWeights(sampler);
    PathContext context{this->lambda_idx};

    Pdf eye_node_pdf;
    RW::PathNode eye_node = SampleSensorEnd(pixel_index, context, eye_node_pdf);   
    MediumTracker eye_medium_tracker{scene};
    InitializeMediumTracker(eye_node, eye_medium_tracker);    
    eye_history.Start(eye_node, lambda_weights/eye_node_pdf, eye_node_pdf);
    BdptForwardTrace(eye_history, eye_medium_tracker, context);
    eye_history.Finish();
    
    Pdf light_node_pdf;
    RW::PathNode light_node = SampleEmissiveEnd(context, light_node_pdf);    
    MediumTracker light_medium_tracker{scene};
    InitializeMediumTracker(light_node, light_medium_tracker);
    light_history.Start(light_node, Spectral3{1./light_node_pdf}, light_node_pdf);
    BdptForwardTrace(light_history, light_medium_tracker, context);
    light_history.Finish();
    
    assert (light_history.NumNodes() <= max_path_node_count);
    assert (eye_history.NumNodes() <= max_path_node_count);
    
    if (light_history.NumNodes()>1 && light_history.NodeFromBack(0).node_type != RW::NodeType::SCATTER)
    {
      light_history.Pop();
    }

    for (int eye_idx=0; eye_idx<eye_history.NumNodes(); ++eye_idx)
    {
      if (!IsHitableEmissiveNode(eye_history.Node(eye_idx)))
      {
        // FIXME: Keep track of media. This media tracker is only valid for the last light/eye node!!!
        if (eye_idx+1 < max_path_node_count)
          DirectLighting(eye_idx, eye_medium_tracker, context);
        for (int light_idx=1; light_idx<std::min(light_history.NumNodes(), max_path_node_count-eye_idx-1); ++light_idx)
        {
          ConnectWithLightPath(eye_idx, light_idx, light_medium_tracker, context);
        }
      }
      else
      {
        assert (eye_idx == eye_history.NumNodes()-1); // Hitting a light should terminate the walk.
        HandleLightHit(context);
      }
    }
 
    for (const auto &splat : splats)
    {
      assert(splat.path_length>=2);
      rgb_responses.push_back(IRenderingAlgo::SensorResponse{
        splat.unit_index,
        Color::SpectralSelectionToRGB(splat.weight, lambda_idx)
      });
    }
    
    return Color::SpectralSelectionToRGB(total_eye_measurement_contributions, lambda_idx);
  }

  
  void DirectLighting(int eye_idx, const MediumTracker &medium_tracker, const PathContext &context)
  {
    Pdf pdf_sample_light;
    RW::PathNode light_node = SampleEmissiveEnd(context, pdf_sample_light); 
    direct_light_history.Reset();
    direct_light_history.Start(light_node, Spectral3{1./pdf_sample_light}, pdf_sample_light);
    direct_light_history.Finish();
    
    double pdf_scatter_eye, pdf_scatter_light;
    WeightedSegment to_eye = Connection(
      direct_light_history.Node(0), medium_tracker, eye_history.Node(eye_idx),
      context, &pdf_scatter_light, &pdf_scatter_eye);
    
    if (to_eye.weight.isZero())
      return;
    
    BdptDetail::Connection connection{
      Pdf{pdf_scatter_eye},
      Pdf{pdf_scatter_light},
      eye_idx,
      0,
      to_eye.segment.Reversed()
    };
    
    double mis_weight = MisWeight(eye_history, direct_light_history, connection);
    Spectral3 path_weight = eye_history.Beta(eye_idx)*direct_light_history.Beta(0)*to_eye.weight;
    AddPathWeight(eye_idx, 0, mis_weight, path_weight);
  }
  
  
  void ConnectWithLightPath(int eye_idx, int light_idx, const MediumTracker &light_medium_tracker, const PathContext &context)
  {
    double eye_pdf, light_pdf;
    WeightedSegment to_eye = Connection(
      light_history.Node(light_idx), light_medium_tracker, eye_history.Node(eye_idx),
      context, &light_pdf, &eye_pdf);
    
    if (to_eye.weight.isZero())
      return;
    
    BdptDetail::Connection connection{
      Pdf{eye_pdf},
      Pdf{light_pdf},
      eye_idx,
      light_idx,
      to_eye.segment.Reversed()
    };
    

    double mis_weight = MisWeight(eye_history, light_history, connection);
    Spectral3 path_weight = eye_history.Beta(eye_idx)*light_history.Beta(light_idx)*to_eye.weight;
    AddPathWeight(eye_idx, light_idx, mis_weight, path_weight);
  }
  
  
  void HandleLightHit(const PathContext &context)
  {
    assert(eye_history.NumNodes()>=2);
    const auto &end_node = eye_history.NodeFromBack(0);
    
    double reverse_scatter_pdf = 0.;
    Spectral3 end_weight = EvaluateScatterCoordinate(
      end_node, -end_node.incident_dir, context, &reverse_scatter_pdf);
    
    BdptDetail::Connection connection{
      Pdf{reverse_scatter_pdf},
      GetPdfOfGeneratingSampleOnEmitter(end_node, context),
      eye_history.NumNodes()-1,
      -1,
      RaySegment{}
    };
    
    double mis_weight = MisWeight(eye_history, light_history, connection);
    Spectral3 path_weight = end_weight*eye_history.Beta(eye_history.NumNodes()-1);
    AddPathWeight(eye_history.NumNodes()-1, -1, mis_weight, path_weight);
  }
  
  
  double MisWeight(BdptDetail::SubpathHistory &eye, BdptDetail::SubpathHistory &light, const BdptDetail::Connection &connection)
  {
    using B = BdptDetail::BackupAndReplace;
    B eye_backup{eye, connection.eye_index, light, connection.light_index, connection.eye_pdf, connection.segment};
    B light_backup(light, connection.light_index, eye, connection.eye_index, connection.light_pdf, connection.segment);
    return bdptmis.Compute(eye, light, connection);
  }
  
  
  void AddPathWeight(int s, int t, double mis_weight, const Spectral3 &path_weight)
  {
    int path_length = s+t+2; // +2 Because s and t are 0-based indices.
    assert(path_length >= 2);
    if (s == 0) // Initial eye vertex.
    {
      if (sensor_connection_unit >= 0)
      {
        int num_attempted_paths_per_unit = scene.GetCamera().xres*scene.GetCamera().yres;
        Spectral3 new_weight = path_weight/num_attempted_paths_per_unit;
        splats.emplace_back(
          sensor_connection_unit,
          mis_weight*new_weight,
          path_length
        );
        AddToDebugBuffer(sensor_connection_unit, s, t, mis_weight, new_weight);
        sensor_connection_unit = -1;
      }
    }
    else
    {
      total_eye_measurement_contributions += mis_weight*path_weight;
      AddToDebugBuffer(pixel_index, s, t, mis_weight, path_weight);
    }
  }
  
    
  void BdptForwardTrace(BdptDetail::SubpathHistory &path, MediumTracker &medium_tracker, const PathContext &context)
  {
    if (this->max_path_node_count <= 1)
      return;
    
    int path_node_count = 1;    
    while (true)
    {
      StepResult step = TakeRandomWalkStep(
        path.NodeFromBack(0), medium_tracker, context);
      
      if (step.node.node_type == RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK) // Hit a light or particle escaped the scene or aborting.
        break;   
      
      path.AddSegment(step.node, step.beta_factor, step.scatter_pdf, step.segment);
      
      ++path_node_count;

      bool survive = SurvivalAtNthScatterNode(step.beta_factor, path_node_count); 
      if (!survive)
        break;
      
      if (step.node.node_type != RW::NodeType::SCATTER)
        break;
    }    
  }

  // TODO: Duplicated code. Same as in PathTracing. Bad! But interface classes should not have state. In particular not with multi-inheritance. -> Need better interface!
  ToyVector<IRenderingAlgo::SensorResponse>& GetSensorResponses() override
  {
    return rgb_responses;
  }
  
  void AddToDebugBuffer(int unit_index, int s, int t, double mis_weight, const Spectral3 &path_weight)
  {
    if (!enable_debug_buffers)
      return;

    auto key = std::make_pair(s, t);
    auto it = debug_buffers.find(key);
    if (it == debug_buffers.end())
    {
      bool _;
      std::tie(it, _) = debug_buffers.insert(std::make_pair(
        key, Spectral3ImageBuffer(scene.GetCamera().xres, scene.GetCamera().yres)));

    }
    it->second.Insert(unit_index, Color::SpectralSelectionToRGB(path_weight, lambda_idx));
    
    if (!path_weight.isZero())
    {
      it = debug_buffers_mis.find(key);
      if (it == debug_buffers_mis.end())
      {
        bool _;
        std::tie(it, _) = debug_buffers_mis.insert(std::make_pair(
          key, Spectral3ImageBuffer(scene.GetCamera().xres, scene.GetCamera().yres)));

      }
      it->second.Insert(unit_index, RGB{Color::RGBScalar{mis_weight}});
    }
  }
  
  
  void NotifyPassesFinished(int pass_count) override;
};






class PathTracing : public IRenderingAlgo, public RandomWalk::RadianceEstimatorBase
{ 
  LambdaSelectionFactory lambda_selection_factory;
  bool do_sample_brdf;
  bool do_sample_lights;
  ToyVector<IRenderingAlgo::SensorResponse> rgb_responses;
public:
  PathTracing(const Scene &_scene, const AlgorithmParameters &algo_params) 
  : RadianceEstimatorBase{_scene, algo_params},
    lambda_selection_factory{},
    do_sample_brdf{true},
    do_sample_lights{true}
  {
    // Which style of sampling for the last vertex of the path. 
    // Defaults to both light and brdf sampling.
    if (algo_params.pt_sample_mode == "bsdf")
    {
      do_sample_lights = false;
    }
    else if (algo_params.pt_sample_mode == "lights")
    {
      do_sample_brdf = false;
    }
    rgb_responses.reserve(1024);
  }

  
  RGB MakePrettyPixel(int pixel_index) override
  {
    sensor_connection_unit = -1;
    
    auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    PathContext context{lambda_selection.first};
    MediumTracker medium_tracker{scene};
    
    Spectral3 path_sample_values{0.};

    RW::PathNode prev_node;
    
    StepResult step;
    step.node = SampleSensorEnd(pixel_index, context, step.scatter_pdf); // Note: abuse of scatter_pdf. This is not about scattering but picking the initial location.
    
    Spectral3 beta{lambda_selection.second/step.scatter_pdf};
    
    InitializeMediumTracker(step.node, medium_tracker);

    int path_node_count = 1;
    while (true)
    {
      if (this->do_sample_lights) 
      {
        // Next vertex. Sample a point on a light source. Compute geometry term. Add path weight to total weights.
        Pdf pdf_light;
        RW::PathNode light_node = SampleEmissiveEnd(context, pdf_light);
        double pdf_scatter = NaN;
        WeightedSegment to_light = Connection(step.node, medium_tracker, light_node, context, &pdf_scatter, nullptr);
        double pdf_of_light_due_to_scatter = pdf_scatter*PdfConversionFactorForTarget(step.node, light_node, to_light.segment, false);
        double mis_weight = MisWeight(
          pdf_light, // Maybe this needs multiplication with pmf to select the unit_index.
          pdf_of_light_due_to_scatter);
        Spectral3 path_weight = (mis_weight/pdf_light)*to_light.weight*beta;
        if (step.node.node_type == NodeType::CAMERA)
        {
          if (sensor_connection_unit >= 0)
          {
            int num_attempted_paths = scene.GetCamera().xres*scene.GetCamera().yres;
            path_weight /= num_attempted_paths;
            rgb_responses.push_back({
              sensor_connection_unit,
              Color::SpectralSelectionToRGB(path_weight, lambda_selection.first)
            }); // TODO: emplace_back wants a ctor. This part works via the auto-generated curly braces ctor.
          }
        }
        else
        {
          path_sample_values += path_weight;
        }
      }

      prev_node = step.node;
      step = TakeRandomWalkStep(prev_node, medium_tracker, context);
      
      beta *= step.beta_factor;
      
      ++path_node_count;
      
      if (step.node.node_type != RW::NodeType::SCATTER) // Hit a light or particle escaped the scene or aborting.
        break;   
      
      bool survive = SurvivalAtNthScatterNode(beta, path_node_count); 
      if (!survive)
        break;
      
      assert(beta.allFinite());
    }

    if (this->do_sample_brdf && IsHitableEmissiveNode(step.node))
    {
      double pdf_end_pos = 0.;
      Spectral3 end_weight = EndPathWithZeroVerticesOnOtherSide(prev_node, step.node, step.segment, context, &pdf_end_pos);
      
      Pdf step_pdf_of_end = PdfConversionFactorForTarget(prev_node, step.node, step.segment, step.scatter_pdf.IsFromDelta())*step.scatter_pdf;
      
      double mis_weight = MisWeight(step_pdf_of_end, pdf_end_pos);
      
      path_sample_values += mis_weight*end_weight*beta;
    }
 
    return Color::SpectralSelectionToRGB(path_sample_values, lambda_selection.first);
  }

  
  // TODO: remove parameter for start node?!
  Spectral3 EndPathWithZeroVerticesOnOtherSide(const RW::PathNode &, const RW::PathNode &end_node, const RaySegment &segment_to_end,  const PathContext &context, double *pdf_reverse_target)
  {
    assert(IsHitableEmissiveNode(end_node));
    Spectral3 scatter_value = EvaluateScatterCoordinate(end_node, -segment_to_end.ray.dir, context, nullptr);
    if (pdf_reverse_target)
    {
      // Does not need conversion, because is already either w.r.t. area or angle (env).
      *pdf_reverse_target = GetPdfOfGeneratingSampleOnEmitter(end_node, context);
    }
    return scatter_value;
  }
  
  
  ToyVector<IRenderingAlgo::SensorResponse>& GetSensorResponses() override
  {
    return rgb_responses;
  }
  

  double MisWeight(Pdf pdf_or_pmf_taken, double pdf_other) const
  {
    double mis_weight = 1.;
    if (!pdf_or_pmf_taken.IsFromDelta() && this->do_sample_brdf && this->do_sample_lights)
    {
      mis_weight = PowerHeuristic(pdf_or_pmf_taken, {pdf_other});
    }
    return mis_weight;
  }
};

} // namespace

using PathTracing = RandomWalk::PathTracing;
using Bdpt = RandomWalk::Bdpt;
using RadianceEstimatorBase = RandomWalk::RadianceEstimatorBase;

#pragma GCC diagnostic pop // Restore command line options
