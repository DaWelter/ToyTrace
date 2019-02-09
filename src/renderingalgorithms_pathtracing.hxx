#pragma once
#include "rendering_randomwalk_impl.hxx"
#include "renderingalgorithms_simplebase.hxx"

#include <unordered_map>
#include <boost/functional/hash.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wunused-parameter" 
#pragma GCC diagnostic warning "-Wunused-variable"

namespace RandomWalk
{
using namespace SimplePixelByPixelRenderingDetails;
  
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
    pdfs_rev.clear();
    nodes.clear();
    betas.clear();
    segments.clear();
    is_parallel_beam = true;
  }
  
  void Start(const RW::PathNode &start, const Spectral3 &_weight, Pdf _start_pdf)
  {
    // Bah ... copying :-(
    nodes.push_back(start);
    betas.push_back(_weight);
    pdfs.push_back(_start_pdf);  // The probability density of the starting location.
    pdfs_rev.push_back(1.);
    segments.push_back(RaySegment{}); // For convenience
    fwd_conversion_factors.push_back(1.);
    bwd_conversion_factors.push_back(1.);
    assert(_start_pdf > 0 && std::isfinite(_start_pdf)); // Because it is the sample. So the probability to generate it must be positive!
  }
  
  void AddSegment(const RW::PathNode &end_node, const Spectral3 &_weight, Pdf pdf_prev_scatter, Pdf pdf_prev_rev_scatter, const VolumePdfCoefficients &volume_pdf_coeff, const RaySegment &segment)
  {
    nodes.push_back(end_node);
    const int idx = nodes.size()-1;
    Spectral3  beta_at_node = betas.back()*_weight;
    betas.push_back(beta_at_node);
    pdfs.push_back(pdf_prev_scatter);
    pdfs_rev.push_back(pdf_prev_rev_scatter);
    segments.push_back(segment);
    is_parallel_beam &= pdf_prev_scatter.IsFromDelta();
    double geom_coeff = randomwalk_functions.PdfConversionFactor(nodes[idx-1], nodes[idx], segments[idx], FwdCoeffs(volume_pdf_coeff), is_parallel_beam);
    fwd_conversion_factors.push_back(geom_coeff);
    geom_coeff = randomwalk_functions.PdfConversionFactor(nodes[idx], nodes[idx-1], segments[idx].Reversed(), BwdCoeffs(volume_pdf_coeff), false);
    bwd_conversion_factors.push_back(geom_coeff);      
    assert(pdf_prev_scatter > 0 && std::isfinite(pdf_prev_scatter)); // Because it is the sample. So the probability to generate it must be positive!
    assert(geom_coeff > 0);
  }
  
  void Finish()
  {
    fwd_conversion_factors.push_back(NaN); // Space for data of connection segment.
    pdfs.push_back(Pdf{});  // Dito.
    pdfs_rev.push_back(Pdf{});
    assert (fwd_conversion_factors.size() == nodes.size() + 1);
    assert (bwd_conversion_factors.size() == nodes.size());
    assert (pdfs.size() == nodes.size() + 1);
    assert (pdfs_rev.size() == pdfs.size());
  }
  
  void Pop()
  {
    assert (!nodes.empty());
    nodes.pop_back();
    pdfs.pop_back();
    pdfs_rev.pop_back();
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

  RW::PathNode& Node(int i)
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

  friend class BdptMis;
  friend class BackupAndReplace;
private:
  RadianceEstimatorBase& randomwalk_functions;
  ConversionFactorContainer fwd_conversion_factors;
  ConversionFactorContainer bwd_conversion_factors;
  ToyVector<RaySegment> segments; // TODO: Might be able to get rid of this.
  PdfContainer pdfs; // Initial node probability, then scatter pdf at *previous* node.
  PdfContainer pdfs_rev; // First 1, then scatter pdf with swapped arguments at *previous* node.
  NodeContainer nodes;
  WeightContainer betas;
  bool is_parallel_beam;
};


struct Connection
{
  Pdf eye_pdf;
  Pdf eye_pdf_rev;
  Pdf light_pdf;
  Pdf light_pdf_rev;
  VolumePdfCoefficients volume_pdf_coeff;
  int eye_index;
  int light_index;
  RaySegment segment;
};


using DebugBuffers = std::unordered_map<std::pair<int,int>, Spectral3ImageBuffer, pair_hash<int,int>>;


class BackupAndReplace
{
  SubpathHistory &h;
  int node_index; 
  Pdf pdf;
  Pdf pdf_rev;
  double fwd_conversion_factor;
  
  void InitBothPathsNonEmpty(const RW::PathNode &other_end_node, Pdf pdf_node_scatter, Pdf pdf_node_scatter_rev, const RaySegment &segment, const VolumePdfCoefficients &vol_pdf_coeff)
  {
    assert (node_index >= 0 && other_end_node.node_type != RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK);
    // The pdf arrays contain the pdf at the previous node. So bump the index by one.
    pdf = h.pdfs[node_index+1];
    pdf_rev = h.pdfs_rev[node_index+1];
    // The conversion factor "pointing" to the next node.
    fwd_conversion_factor = h.fwd_conversion_factors[node_index+1];
    h.pdfs[node_index+1] = pdf_node_scatter; 
    h.pdfs_rev[node_index+1] = pdf_node_scatter_rev;
    // Replace by conversion factor for going from solid angle at this node to position at the opposite node.
    h.fwd_conversion_factors[node_index+1] = h.randomwalk_functions.PdfConversionFactor(
      h.Node(node_index), other_end_node, segment, FwdCoeffs(vol_pdf_coeff), pdf_node_scatter.IsFromDelta());
  }
  
  void InitEitherSideEmpty(double one_or_start_position_pdf, double one_or_angular_emission_pdf)
  {
    // If this side has zero vertices, then it is a light (or not-implemented sensor), which was randomly hit, 
    // and the pdf should have the node position density in the forward pdf and 1 in the reverse pdf.
    // On the other hand,
    // If the other side has zero vertices, then this is the path that did the random walk to hit the light.
    // It's forward pdf should be set to 1, and the reverse pdf to the angular emission density.
    //assert ((one_or_angular_emission_pdf==1.) ^ (one_or_start_position_pdf==1.));
    pdf = h.pdfs[node_index+1];
    pdf_rev = h.pdfs[node_index+1];
    fwd_conversion_factor = h.fwd_conversion_factors[node_index+1];
    h.pdfs[node_index+1] = one_or_start_position_pdf;
    h.pdfs_rev[node_index+1] = one_or_angular_emission_pdf;
    h.fwd_conversion_factors[node_index+1] = 1.;
  }
   
public:
  BackupAndReplace(const BackupAndReplace &) = delete;
  BackupAndReplace& operator=(const BackupAndReplace &) = delete;
  
  BackupAndReplace(SubpathHistory &_h, int _node_index, SubpathHistory &other_h, int other_node_index, Pdf pdf, Pdf pdf_rev, const RaySegment &segment, const VolumePdfCoefficients &vol_pdf_coeff)
    : h{_h}, node_index{_node_index} 
  {
    if (_node_index >= 0 && other_node_index >= 0)
      InitBothPathsNonEmpty(other_h.Node(other_node_index), pdf, pdf_rev, segment, vol_pdf_coeff);
    else
      InitEitherSideEmpty(pdf, pdf_rev);
  }
  
  ~BackupAndReplace() 
  {
    h.pdfs[node_index+1] = pdf;
    h.fwd_conversion_factors[node_index+1] = fwd_conversion_factor;
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
    //double result = WeightByNumberOfAdmissibleTechniques(); // For debugging!
    return result;
  }
private:
  SpecularFlagContainer specular_flags;
  PathDensityContainer eye_densities;
  PathDensityContainer light_densities;
  int total_node_count;
  double one_over_number_of_splat_attempts_per_pixel;
    
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
    assert (fwd_path.fwd_conversion_factors[0] == 1.);
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
        reverse_path.bwd_conversion_factors[i] * reverse_path.pdfs_rev[i+1]);
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
    assert (std::isfinite(nominator));
    double denom = 0.;
    for (int s = 0; s <= total_node_count; ++s)
    {
      bool flag = !(specular_flags[s] || specular_flags[s+1]);
      double summand = flag ? (eye_densities[s]*light_densities[total_node_count-s]) : 0.;
      denom += Sqr(summand);
      assert (std::isfinite(denom));
    }
    assert(denom > 0.);
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


class BdptWorker;


class BdptAlgo : public SimplePixelByPixelRenderingDetails::SimplePixelByPixelRenderingAlgo
{
  friend class BdptWorker;
  bool enable_debug_buffers = false;
  BdptDetail::DebugBuffers debug_buffers;
  BdptDetail::DebugBuffers debug_buffers_mis;
  tbb::spin_mutex debug_buffer_mutex;
public:
  BdptAlgo(const Scene &scene_,const RenderingParameters &render_params_)
    : SimplePixelByPixelRenderingAlgo{render_params_,scene_}
  {}
protected:
  std::unique_ptr<SimplePixelByPixelRenderingDetails::Worker> AllocateWorker(int i) override;
  void PassCompleted() override;
};


class BdptWorker : public RadianceEstimatorBase, public Worker
{
  LambdaSelectionFactory lambda_selection_factory;
  ToyVector<SensorResponse> rgb_responses;

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
  ToyVector<MediumTracker> eye_medium_tracker_before_node;
  Spectral3 total_eye_measurement_contributions;
  ToyVector<Splat> splats;
  Index3 lambda_idx;
  int pixel_index;
  PathContext eye_context;
  PathContext light_context;
  BdptAlgo* master;
  
public:
  BdptWorker(const Scene &_scene, const AlgorithmParameters &algo_params, BdptAlgo* master) 
  : RadianceEstimatorBase{_scene, algo_params},
    lambda_selection_factory{},
    eye_history{*this},
    direct_light_history{*this},
    light_history{*this},
    bdptmis{_scene},
    eye_medium_tracker_before_node{},
    master{master}
  {
    rgb_responses.reserve(1024);
    eye_medium_tracker_before_node.reserve(32);
  }

  
  RGB RenderPixel(int _pixel_index) override
  {
    eye_history.Reset();
    light_history.Reset();
    total_eye_measurement_contributions = Spectral3{0.};
    splats.clear();
    eye_medium_tracker_before_node.clear();
    
    pixel_index = _pixel_index;
    Spectral3 lambda_weights;
    std::tie(this->lambda_idx, lambda_weights) = lambda_selection_factory.WithWeights(sampler);

    eye_context = PathContext{this->lambda_idx, TransportType::RADIANCE};
    light_context = PathContext{this->lambda_idx, TransportType::IMPORTANCE};

    {
      Pdf eye_node_pdf;
      RW::PathNode eye_node = SampleSensorEnd(pixel_index, eye_context, eye_node_pdf);   
      MediumTracker eye_medium_tracker{scene};
      InitializeMediumTracker(eye_node, eye_medium_tracker);    
      eye_history.Start(eye_node, lambda_weights/eye_node_pdf, eye_node_pdf);
      BdptForwardTrace(
        eye_history, eye_medium_tracker,                
        [&eye_medium_tracker, this](int ) { this->eye_medium_tracker_before_node.emplace_back(eye_medium_tracker); }, 
        eye_context);
      eye_history.Finish();
    }
    
    {
      Pdf light_node_pdf;
      RW::PathNode light_node = SampleEmissiveEnd(light_context, light_node_pdf);    
      MediumTracker light_medium_tracker{scene};
      InitializeMediumTracker(light_node, light_medium_tracker);
      light_history.Start(light_node, Spectral3{1./light_node_pdf}, light_node_pdf);
      BdptForwardTrace(
        light_history, light_medium_tracker, 
        [](int ) {}, 
        light_context);
      light_history.Finish();
    }
   
    
    assert (light_history.NumNodes() <= max_path_node_count);
    assert (eye_history.NumNodes() <= max_path_node_count);

    for (int eye_idx=0; eye_idx<eye_history.NumNodes(); ++eye_idx)
    {
      if (eye_idx==0 || eye_history.Node(eye_idx).IsScatterNode())
      {
        // Only camera and scatter nodes are allowed to connect to the light path. Sensor nodes only if they are the first node.
        // In particular we must not connect light sources on the eye path to the light path. 
        // Surface or volume nodes default to being scatter nodes, so I don't have to consider these explicitly in the above conditional.
        if (eye_idx+1 < max_path_node_count)
          DirectLighting(eye_idx, eye_medium_tracker_before_node[eye_idx]);
        for (int light_idx=1; light_idx<std::min(light_history.NumNodes(), max_path_node_count-eye_idx-1); ++light_idx)
        {
          ConnectWithLightPath(eye_idx, light_idx,  eye_medium_tracker_before_node[eye_idx]);
        }
      }
      if (eye_idx>0) // Would cause error on starting node
        HandleLightHit(eye_idx);
    }
 
    for (const auto &splat : splats)
    {
      assert(splat.path_length>=2);
      rgb_responses.push_back(SensorResponse{
        splat.unit_index,
        Color::SpectralSelectionToRGB(splat.weight, lambda_idx)
      });
    }
    
    return Color::SpectralSelectionToRGB(total_eye_measurement_contributions, lambda_idx);
  }
  
  void DirectLighting(int eye_idx, const MediumTracker &eye_medium_tracker)
  {
    Pdf pdf_sample_light;
    RW::PathNode light_node = SampleEmissiveEnd(light_context, pdf_sample_light); 
    direct_light_history.Reset();
    direct_light_history.Start(light_node, Spectral3{1./pdf_sample_light}, pdf_sample_light);
    direct_light_history.Finish();
    
    ConnectEyeWithOtherPath(eye_idx, 0, direct_light_history, eye_medium_tracker);
  }
  
  
  void ConnectWithLightPath(int eye_idx, int light_idx, const MediumTracker &eye_medium_tracker)
  {
    ConnectEyeWithOtherPath(eye_idx, light_idx, light_history, eye_medium_tracker);
  }
  
  
  void ConnectEyeWithOtherPath(int eye_idx, int other_idx, BdptDetail::SubpathHistory &other, const MediumTracker &eye_medium_tracker)
  {
    VolumePdfCoefficients volume_pdf_coeff{};
    double eye_pdf, light_pdf;
    ConnectionSegment to_light = CalculateConnection(
      eye_history.Node(eye_idx), eye_medium_tracker, other.Node(other_idx),
      eye_context, light_context, &eye_pdf, &light_pdf, &volume_pdf_coeff);
    
    if (to_light.weight.isZero())
      return;

    double eye_rev_pdf = ReverseScatterPdf(eye_history.Node(eye_idx), eye_pdf, to_light.segment.ray.dir, this->eye_context);
    double light_rev_pdf = ReverseScatterPdf(light_history.Node(other_idx), light_pdf, -to_light.segment.ray.dir, this->light_context);
    
    double mis_weight = MisWeight(eye_history, other, 
      BdptDetail::Connection{
        Pdf{eye_pdf},
        Pdf{eye_rev_pdf},
        Pdf{light_pdf},
        Pdf{light_rev_pdf},
        volume_pdf_coeff,
        eye_idx,
        other_idx,
        to_light.segment
      }
    );
    
    Spectral3 path_weight = eye_history.Beta(eye_idx)*other.Beta(other_idx)*to_light.weight;
    AddPathContribution(eye_idx, other_idx, mis_weight, path_weight);
  }
  
  
  // Eye node lies on a light source. Compute path contribution.
  void HandleLightHit(int eye_idx)
  {
    assert(eye_history.NumNodes()>=2);
    auto &end_node = eye_history.Node(eye_idx);
    
    if (!end_node.TryEnableOperationsOnEmitter(scene))
      return;
    
    double reverse_scatter_pdf = 0.;
    Spectral3 end_weight = Evaluate(
      end_node, -end_node.incident_dir, light_context, &reverse_scatter_pdf);
    
    VolumePdfCoefficients volume_pdf_coeff{}; // Already covered by the random walk. The SubpathHistory has the coefficients for the last segment.
    
    double mis_weight = MisWeight(eye_history, light_history, 
      BdptDetail::Connection{
        Pdf{1.},
        Pdf{reverse_scatter_pdf},
        GetPdfOfGeneratingSampleOnEmitter(end_node, light_context),
        Pdf{1.},
        volume_pdf_coeff,
        eye_idx,
        -1,
        RaySegment{}
      }
    );
    
    Spectral3 path_weight = end_weight*eye_history.Beta(eye_idx);
    AddPathContribution(eye_idx, -1, mis_weight, path_weight);
    
    end_node.EnableOperationOnScattererIfFeasible();
  }
  
  
  double MisWeight(BdptDetail::SubpathHistory &eye, BdptDetail::SubpathHistory &light, BdptDetail::Connection &&connection)
  {
    using B = BdptDetail::BackupAndReplace;
    B eye_backup{eye, connection.eye_index, light, connection.light_index, connection.eye_pdf, connection.eye_pdf_rev, connection.segment, connection.volume_pdf_coeff};
    std::swap(connection.volume_pdf_coeff.pdf_scatter_bwd, connection.volume_pdf_coeff.pdf_scatter_fwd);
    B light_backup{light, connection.light_index, eye, connection.eye_index, connection.light_pdf, connection.light_pdf_rev, connection.segment, connection.volume_pdf_coeff};
    return bdptmis.Compute(eye, light, connection);
  }
  
  
  void AddPathContribution(int s, int t, double mis_weight, const Spectral3 &path_weight)
  {
    assert(path_weight.allFinite() && std::isfinite(mis_weight));
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
   
  
  template<class StepCallback>
  void BdptForwardTrace(BdptDetail::SubpathHistory &path, MediumTracker &medium_tracker, StepCallback step_callback, const PathContext &context)
  {
    step_callback(0);
    
    if (this->max_path_node_count <= 1)
      return;
    
    int path_node_count = 1;    
    while (true)
    { 
      VolumePdfCoefficients volume_pdf_coeff;
      auto &current_node = path.NodeFromBack(0);
      
      StepResult step = TakeRandomWalkStep(
        current_node, medium_tracker, context, &volume_pdf_coeff);
      
      if (step.node.node_type == RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK) // Hit a light or particle escaped the scene or aborting.
        break;

      // If this is a light path we must not generate paths with light sources in between the ends.
      if (context.transport==IMPORTANCE && !step.node.IsScatterNode())
        break;
      
      Pdf reverse_pdf = ReverseScatterPdf(current_node, step.scatter_pdf, step.segment.ray.dir, context);
      
      // Determine survival of russian roulette early because it modifies beta_factor!
      bool survive = SurvivalAtNthScatterNode(step.beta_factor, path_node_count+1); 
      
      path.AddSegment(step.node, step.beta_factor, step.scatter_pdf, reverse_pdf, volume_pdf_coeff, step.segment);
      
      step_callback(path_node_count);
            
      // The new node is used definitely. However now the walk can terminate.
      // It would make more sense maybe to do RR right after evaluating the 
      // scatter function. On the other hand, it should not matter how and where 
      // I compute the termination (or survival) probability.
      if (!survive)
        break;
      
      if (step.node.node_type == RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK ||
          step.node.node_type == RW::NodeType::ENV)
        break;
      
      ++path_node_count;
    }
  }
  
  
  Pdf ReverseScatterPdf(const PathNode &node, Pdf forward_pdf, const Double3 &reverse_incident_dir, const PathContext &context) const
  {
    // Only surface (BSDF) sampling pdf's are allowed to be unsymmetric. There is no need for phase functions to be asymmetric.
    // And emission directions of light sources and sensors don't have a reverse path at all.
    if (node.node_type != NodeType::SURFACE_SCATTER)
      return forward_pdf;
    // Components proportional to Dirac-Delta should be symmetrical because it importance samples the symmetrical bsdf!
    // And then there is really no other place to put the peak other than the symmetrical directions.
    // In that case, the "pdf" value here represents the probability to select the Dirac-Delta component of the material.
    if (forward_pdf.IsFromDelta()) 
      return forward_pdf;

    const SurfaceInteraction &intersection = node.interaction.surface;
    return GetShaderOf(intersection, scene).Pdf(reverse_incident_dir, intersection, -node.incident_dir, context);
  }


  ToyVector<SensorResponse>& GetSensorResponses() override
  {
    return rgb_responses;
  }
  
  
  void AddToDebugBuffer(int unit_index, int s, int t, double mis_weight, const Spectral3 &path_weight)
  {
    if (!master->enable_debug_buffers)
      return;

    tbb::spin_mutex::scoped_lock lock(master->debug_buffer_mutex);
    auto key = std::make_pair(s, t);
    auto it = master->debug_buffers.find(key);
    if (it == master->debug_buffers.end())
    {
      bool _;
      std::tie(it, _) = master->debug_buffers.insert(std::make_pair(
        key, Spectral3ImageBuffer(scene.GetCamera().xres, scene.GetCamera().yres)));

    }
    it->second.Insert(unit_index, Color::SpectralSelectionToRGB(path_weight, lambda_idx));
    
    if (!path_weight.isZero())
    {
      it = master->debug_buffers_mis.find(key);
      if (it == master->debug_buffers_mis.end())
      {
        bool _;
        std::tie(it, _) = master->debug_buffers_mis.insert(std::make_pair(
          key, Spectral3ImageBuffer(scene.GetCamera().xres, scene.GetCamera().yres)));

      }
      it->second.Insert(unit_index, RGB{Color::RGBScalar{mis_weight}});
    }
  }
};



class PathTracingWorker : public Worker, public RandomWalk::RadianceEstimatorBase
{ 
  LambdaSelectionFactory lambda_selection_factory;
  bool do_sample_brdf;
  bool do_sample_lights;
  ToyVector<SensorResponse> rgb_responses;
public:
  PathTracingWorker(const Scene &_scene, const AlgorithmParameters &algo_params) 
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

  
  RGB RenderPixel(int pixel_index) override
  {
    sensor_connection_unit = -1;
    
    auto lambda_selection = lambda_selection_factory.WithWeights(sampler);
    PathContext context{lambda_selection.first, TransportType::RADIANCE};
    std::tie(context.pixel_x, context.pixel_y) = scene.GetCamera().UnitToPixel(pixel_index);
    PathContext light_context{context}; light_context.transport = IMPORTANCE;
    MediumTracker medium_tracker{scene};
    VolumePdfCoefficients volume_pdf_coeff;
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
        volume_pdf_coeff = VolumePdfCoefficients{};
        ConnectionSegment to_light = CalculateConnection(step.node, medium_tracker, light_node, context, light_context, &pdf_scatter, nullptr, &volume_pdf_coeff);
        double pdf_of_light_due_to_scatter = pdf_scatter*PdfConversionFactor(step.node, light_node, to_light.segment, FwdCoeffs(volume_pdf_coeff), false);
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
            }); // This part works via the auto-generated initializer list ctor.
          }
        }
        else
        {
          path_sample_values += path_weight;
        }
      }

      prev_node = step.node;
      volume_pdf_coeff = VolumePdfCoefficients{};
      step = TakeRandomWalkStep(prev_node, medium_tracker, context, &volume_pdf_coeff);
      
      beta *= step.beta_factor;
      
      ++path_node_count;
      
      if (this->do_sample_brdf && step.node.TryEnableOperationsOnEmitter(scene))
      {
        Spectral3 end_weight = Evaluate(step.node, -step.segment.ray.dir, context, nullptr);
        double pdf_direct_sample = GetPdfOfGeneratingSampleOnEmitter(step.node, context);
        Pdf pdf_due_to_scatter = PdfConversionFactor(prev_node, step.node, step.segment, FwdCoeffs(volume_pdf_coeff), step.scatter_pdf.IsFromDelta())*step.scatter_pdf;
        double mis_weight = MisWeight(pdf_due_to_scatter, pdf_direct_sample);
        path_sample_values += mis_weight*end_weight*beta;
        step.node.EnableOperationOnScattererIfFeasible();
      }
      
      if (step.node.node_type == RW::NodeType::ZERO_CONTRIBUTION_ABORT_WALK ||
          step.node.node_type == RW::NodeType::ENV)
        break;
      
      bool survive = SurvivalAtNthScatterNode(beta, path_node_count); 
      if (!survive)
        break;
      
      assert(beta.allFinite());
    }

 
    return Color::SpectralSelectionToRGB(path_sample_values, lambda_selection.first);
  }
  
  
  ToyVector<SensorResponse>& GetSensorResponses() override
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


using RadianceEstimatorBase = RandomWalk::RadianceEstimatorBase;

#pragma GCC diagnostic pop // Restore command line options
