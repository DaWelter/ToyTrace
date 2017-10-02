#include "scene.hxx"


/* This thing tracks overlapping media volumes. Since it is a bit complicated
 * to physically correctly handle mixtures of media that would occur
 * in overlapping volumes, I take a simpler approach. This hands over the
 * medium with the highest associated priority. In case multiple volumes with the
 * same medium material overlap it will be as if there was the union of all of
 * those volumes.
 */
class MediumTracker
{
  static constexpr int MAX_INTERSECTING_MEDIA = 4;
  const Medium* current;
  std::array<const Medium*, MAX_INTERSECTING_MEDIA> media;
  const Scene &scene;
  void enterVolume(const Medium *medium);
  void leaveVolume(const Medium *medium);
public:
  explicit MediumTracker(const Scene &_scene);
  MediumTracker(const MediumTracker &_other) = default;
  void initializePosition(const Double3 &pos);
  void goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection &intersection);
  const Medium& getCurrentMedium() const;
};


MediumTracker::MediumTracker(const Scene& _scene)
  : scene{_scene}, current{nullptr}, media{}
  // Note: Zero-initializes media array.
{
  
}


const Medium& MediumTracker::getCurrentMedium() const
{
  assert(current);
  return *current;
}


void MediumTracker::initializePosition(const Double3& pos)
{
  // TODO: Find the actual stack of media the position is contained within.
  current = &scene.GetEmptySpaceMedium();
  std::fill(media.begin(), media.end(), nullptr);
}


void MediumTracker::goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection& intersection)
{
  if (Dot(dir_of_travel, intersection.volume_normal) < 0)
    enterVolume(intersection.primitive().medium);
  else
    leaveVolume(intersection.primitive().medium);
}


void MediumTracker::enterVolume(const Medium* medium)
{
  // Set one of the array entries that is currently nullptr to the new medium pointer.
  // But only if there is room left. Otherwise the new medium is ignored.
  bool is_null = false;
  for (int i = 0; (i < media.size()) && !is_null; ++i)
  {
    is_null = media[i]==nullptr;
    media[i] = is_null ? medium : media[i];
  }
  current = (is_null && medium->priority > current->priority) ? medium : current;
}


void MediumTracker::leaveVolume(const Medium* medium)
{
  // If the medium is in the media stack, remove it.
  // And also make the medium of highest prio the current one.
  bool is_eq = false;
  for (int i = 0; (i < media.size()) && !is_eq; ++i)
  {
    is_eq = media[i] == medium;
    media[i] = is_eq ? nullptr : media[i];
  }
  if (medium == current)
  {
    const Medium* medium_max_prio = &scene.GetEmptySpaceMedium();
    for (int i = 0; i < media.size(); ++i)
    {
      medium_max_prio = (media[i] &&  medium_max_prio->priority < media[i]->priority) ?
                          media[i] : medium_max_prio;
    }
    current = medium_max_prio;
  }
}



class BaseAlgo
{
protected:
  const Scene &scene;
  Sampler sampler;
  MediumTracker medium_tracker_root;

public:
  BaseAlgo(const Scene &_scene)
    : scene{_scene},
      medium_tracker_root{_scene}
    {}
    
  std::tuple<const Light*, double> PickLightUniform()
  {
    const auto &light = scene.GetLight(sampler.UniformInt(0, scene.GetNumLights()-1));
    double pmf_of_light = 1./scene.GetNumLights();
    return std::make_tuple(&light, pmf_of_light);
  }
  
  virtual Spectral MakePrettyPixel() = 0;
};



class Raytracing : public BaseAlgo
{
  int max_level = 5;
public:
  Raytracing(const Scene &_scene) : BaseAlgo(_scene) {}
  
  RaySegment MakeSegmentToLight(const Double3 &pos_to_be_lit, const RadianceOrImportance::Sample &light_sample)
  {
    if (!light_sample.is_direction)
      return RaySegment::FromTo(pos_to_be_lit, light_sample.pos);
    else
    {
      return RaySegment{{pos_to_be_lit, -light_sample.pos}, LargeNumber};
    }
  }
  
  
  Spectral TransmittanceEstimate(RaySegment seg, HitId last_hit, MediumTracker medium_tracker)
  {
    // TODO: Russian roulette.
    Spectral result{1.};
    double length_to_go = seg.length;
    while (length_to_go > Epsilon && result.abs().sum()>1.e-6)
    {
      
      // The last_hit given by the argument is ignored.
      last_hit = scene.Intersect(seg.ray, seg.length, last_hit);
      if (last_hit)
      {
        RaySurfaceIntersection intersection{last_hit, seg};
        Spectral bsdf = intersection.shader().EvaluateBSDF(-seg.ray.dir, intersection, seg.ray.dir, nullptr);
        result *= bsdf * medium_tracker.getCurrentMedium().EvaluateTransmission(seg);
        medium_tracker.goingThroughSurface(seg.ray.dir, intersection);
        length_to_go -= seg.length;
        seg.ray.org += seg.length * seg.ray.dir;
        seg.length = length_to_go;
      }
      else
      {
        length_to_go = 0.;
        result *= medium_tracker.getCurrentMedium().EvaluateTransmission(seg);
      }
    }
    return result;
  }
  
  
  std::pair<Spectral, Double3> LightEmissionWithTransmittanceTo(const Double3 &pos, const HitId &last_hit, const MediumTracker &medium_tracker_parent)
  {
    if (scene.GetNumLights() <= 0)
      return std::make_pair(Spectral{0.}, Double3{0.});

    const Light* light; double pmf_of_light; std::tie(light, pmf_of_light) = PickLightUniform();

    RadianceOrImportance::Sample light_sample = light->TakePositionSample(sampler);
    RaySegment segment_to_light = MakeSegmentToLight(pos, light_sample);
    
    // TODO: consider case of infinitely far away light!!
    auto transmittance = TransmittanceEstimate(segment_to_light, last_hit, medium_tracker_parent);
    auto sample_value = (transmittance * light_sample.measurement_contribution)
                         / (light_sample.pdf_of_pos * Sqr(segment_to_light.length));
    
    return std::make_pair(sample_value, segment_to_light.ray.dir);
  }
  
  
  Spectral LightConnection(const Double3 &incident_dir, const RaySurfaceIntersection &intersection, const MediumTracker &medium_tracker_parent)
  {
    Spectral light_value;
    Double3 dir_to_light;
    std::tie(light_value, dir_to_light) = LightEmissionWithTransmittanceTo(intersection.pos, intersection.hitid, medium_tracker_parent);
    
    if (light_value.abs().sum() <= 1.e-6)
      return Spectral{0.};
    

    const auto &shader = intersection.shader();
    Spectral bsdf_value = shader.EvaluateBSDF(-incident_dir, intersection, dir_to_light, nullptr);
    double d_factor = std::max(0., Dot(intersection.normal, dir_to_light));
    
    auto ret = d_factor * bsdf_value * light_value;
    return ret;
  }
  
  
  Spectral LightConnection(const Double3 &pos, const Double3 &incident_dir, const MediumTracker &medium_tracker_parent)
  {
    Spectral light_value;
    Double3 dir_to_light;
    std::tie(light_value, dir_to_light) = LightEmissionWithTransmittanceTo(pos, HitId(), medium_tracker_parent);
    
    if (light_value.abs().sum() <= 1.e-6)
      return Spectral{0.};
    
    
    const auto &medium = medium_tracker_parent.getCurrentMedium();
    auto phase_func = medium.EvaluatePhaseFunction(-incident_dir, pos, dir_to_light, nullptr);
    auto ret = phase_func * light_value;
    return ret;
  }
  
  
  Spectral PropagationAtIntersection(const Double3 &incident_dir, const RaySurfaceIntersection &intersection, int level, const MediumTracker &medium_tracker_parent)
  {
    if (level > max_level)
      return Spectral{0.};
    
    auto surface_sample  = intersection.shader().SampleBSDF(-incident_dir, intersection, sampler);
    // Preserve the information about current media. However, as we propagate the path further,
    // the current medium can change. Therefore we need to clone the medium tracker.
    MediumTracker medium_tracker(medium_tracker_parent);
    // By definition, intersection.normal points to where the intersection ray is comming from.
    // Thus we can determine if the sampled direction goes through the surface by looking
    // if the direction goes in the opposite direction of the normal.
    if (Dot(surface_sample.dir, intersection.normal) < 0.)
    {
      medium_tracker.goingThroughSurface(surface_sample.dir, intersection);
    }
    auto ret = Trace({intersection.pos, surface_sample.dir}, level+1,  medium_tracker);
    
    double d_factor = std::abs(Dot(intersection.normal, surface_sample.dir));
    ret *= d_factor / surface_sample.pdf * surface_sample.scatter_function;
    return ret;
  }
  
  
  Spectral PropagationAtVolumeInteraction(const Double3 &pos, const Double3 &incident_dir, int level, const MediumTracker &medium_tracker_parent)
  {
    if (level > max_level)
      return Spectral{0.};
    const Medium& medium = medium_tracker_parent.getCurrentMedium();
    
    auto scatter_smpl = medium.SamplePhaseFunction(-incident_dir, pos, sampler);
    auto result = Trace({pos, scatter_smpl.dir}, level+1, medium_tracker_parent);
    
    result *= scatter_smpl.phase_function / scatter_smpl.pdf;
    return result;
  }
  
  
  Spectral Trace(const Ray &ray, int level, const MediumTracker &medium_tracker_parent)
  {
    const Medium& medium = medium_tracker_parent.getCurrentMedium();
    auto segment = RaySegment{ray, LargeNumber};

    HitId hit = scene.Intersect(segment.ray, segment.length);  
    auto medium_smpl = medium.SampleInteractionPoint(segment, sampler);
    
    if (medium_smpl.t < segment.length)
    {
      auto pos = ray.org + medium_smpl.t * ray.dir;
      auto result = PropagationAtVolumeInteraction(
        pos,
        ray.dir,
        level,
        medium_tracker_parent
      );
      
      result += LightConnection(pos, ray.dir, medium_tracker_parent);
      result *= medium_smpl.sigma_s / medium_smpl.pdf;
      return result;
    }
    else
    {
      if (hit)
      {
        RaySurfaceIntersection intersection{hit, segment};
        // Compute specular and translucency path components.
        auto result = PropagationAtIntersection(ray.dir, intersection, level, medium_tracker_parent);
        // Direct lighting.
        result += LightConnection(ray.dir, intersection, medium_tracker_parent);
        return result;
      }
      else
      {
        return scene.bgColor;
      }
    }
  }
  
  
  Spectral MakePrettyPixel() override
  {
    RadianceOrImportance::Sample start_pos_sample = scene.GetCamera().TakePositionSample(sampler);
    RadianceOrImportance::DirectionalSample start = scene.GetCamera().TakeDirectionSampleFrom(start_pos_sample.pos, sampler);
    
    Spectral camera_factor = 
      (start_pos_sample.measurement_contribution*start.measurement_contribution) 
      / start_pos_sample.pdf_of_pos /  start.pdf;
    
    this->medium_tracker_root.initializePosition(start_pos_sample.pos);
    Spectral eye_tree_factor = Trace(start.ray_out, 1, this->medium_tracker_root);

    return camera_factor * eye_tree_factor;
  };
};


#if 1
// TODO: Turn this abomination into a polymorphic type with custom allocator??
//       Use factory to hide allocation in some given buffer (stack or heap).
//       Something like this https://stackoverflow.com/questions/13340664/polymorphism-without-new
//       
struct PathNode
{
  using DirectionalSample = RadianceOrImportance::DirectionalSample;

  enum PathNodeType : char 
  {
    VOLUME,
    NODE1,
    SURFACE,
    NODE1_DIRECTIONAL,
    ESCAPED,
  };
  
  PathNodeType type;
  char index;
  
  // TODO: Use proper variant type. But I don't want boost yet. Don't have c++17 either which has a variant type.
  struct Surface
  {
    RaySurfaceIntersection intersection;
    Double3 inbound_dir;
  };
  struct One
  {
    const RadianceOrImportance::EmitterSensor *emitter;
    Double3 pos_or_dir_to;
    //One() : emitter{nullptr} {}
  };
  struct Volume
  {
    Double3 pos;
    Double3 inbound_dir;
  };
  static auto constexpr BUFFER_SIZE = std::max({sizeof(Surface), sizeof(One), sizeof(Volume)});
  char buffer[BUFFER_SIZE];
  Surface& surface() 
  { 
    assert(type==SURFACE); 
    return *reinterpret_cast<Surface*>(buffer); 
  }
  Volume& volume()
  { 
    assert(type==VOLUME); 
    return *reinterpret_cast<Volume*>(buffer); 
  }
  One& one() 
  { 
    assert(type==NODE1 || type == NODE1_DIRECTIONAL); 
    return *reinterpret_cast<One*>(buffer); 
  }
  const Surface& surface() const { return const_cast<PathNode*>(this)->surface(); }
  const Volume& volume() const { return const_cast<PathNode*>(this)->volume(); }
  const One& one() const { return const_cast<PathNode*>(this)->one(); }
  
  PathNode(PathNodeType _type, char _index) : type{_type}, index{_index} {}
  PathNode(const PathNode &);
  PathNode& operator=(const PathNode &);
//   PathNode(PathNode &&);
//   PathNode& operator=(const PathNode &&);
  DirectionalSample SampleDirection(Sampler &sampler) const;
  Spectral EvaluateDirection(const Double3 &outbound_dir) const;
  Double3 Position() const;
  Double3 DirectionToEmitter() const;
  Double3 InboundDir() const;
  HitId hitId() const;

  static PathNode MakeTypeOne(
    const RadianceOrImportance::EmitterSensor &emitter, 
    const RadianceOrImportance::Sample &sample);
  static PathNode MakeSurface(
        char index, const Double3& inbound_dir, const RaySurfaceIntersection& intersection);
  static PathNode MakeEscaped(char index);
};


Double3 PathNode::Position() const
{
  const auto &node = *this;
  assert (node.type != NODE1_DIRECTIONAL);
  return node.type == NODE1 ? 
    node.one().pos_or_dir_to : 
      (node.type == VOLUME ? 
        node.volume().pos : node.surface().intersection.pos);
}


Double3 PathNode::DirectionToEmitter() const
{
  const auto &node = *this;
  assert (node.type == NODE1_DIRECTIONAL);
  return node.one().pos_or_dir_to;
}


Double3 PathNode::InboundDir() const
{
  const auto &node = *this;
  assert (node.type == SURFACE || node.type == VOLUME);
  return node.type == SURFACE ? 
    node.surface().inbound_dir : 
    node.volume().inbound_dir;
}


HitId PathNode::hitId() const
{
  return (type == SURFACE) ? surface().intersection.hitid : HitId{};
}


PathNode::PathNode(const PathNode& other) :
  type{other.type},
  index{other.index}
{
  switch(type)
  {
    case NODE1:
    case NODE1_DIRECTIONAL:
      new (buffer) One(other.one());
      break;
    case SURFACE:
      new (buffer) Surface(other.surface());
      break;
    case VOLUME:
      new (buffer) Volume(other.volume());
      break;
  }
}


PathNode& PathNode::operator=(const PathNode& other)
{
  this->~PathNode();
  new (this) PathNode(other);
  return *this;
}


Spectral PathNode::EvaluateDirection(const Double3& outbound_dir) const
{
  const auto &node = *this;
  if (node.type ==NODE1 || node.type == NODE1_DIRECTIONAL)
  {
    return node.one().emitter->EvaluateDirectionComponent(node.one().pos_or_dir_to, outbound_dir, nullptr);
  }
  else if (node.type == SURFACE)
  {
    return node.surface().intersection.shader().EvaluateBSDF(-node.surface().inbound_dir, node.surface().intersection, outbound_dir, nullptr);
  }
  assert(false && "Not implemented");
}


PathNode::DirectionalSample PathNode::SampleDirection(Sampler &sampler) const
{
  const auto &node = *this;
  if (node.type == NODE1 || node.type == NODE1_DIRECTIONAL)
  {
    return node.one().emitter->TakeDirectionSampleFrom(node.one().pos_or_dir_to, sampler);
  }
  else if (node.type == SURFACE)
  {
    auto sample = node.surface().intersection.shader().SampleBSDF(-node.surface().inbound_dir, node.surface().intersection, sampler);
    double d_factor = Dot(node.surface().intersection.normal, sample.dir);
    return RadianceOrImportance::DirectionalSample{
      {node.surface().intersection.pos, sample.dir}, 
      sample.pdf, 
      d_factor * sample.scatter_function};
  }
  assert(false && "Not implemented");
}


PathNode PathNode::MakeSurface(char index, const Double3& inbound_dir, const RaySurfaceIntersection& intersection)
{
  PathNode result{SURFACE, index};
  new (result.buffer) Surface{intersection, inbound_dir};
  return result;
}


PathNode PathNode::MakeTypeOne(const RadianceOrImportance::EmitterSensor& emitter, const RadianceOrImportance::Sample& sample)
{
  PathNode result{sample.is_direction ? NODE1_DIRECTIONAL : NODE1, 1};
  new (result.buffer) One{&emitter, sample.pos};
  return result;
}


PathNode PathNode::MakeEscaped(char index)
{
  return PathNode{ESCAPED, index};
}




class Bdpt : public BaseAlgo
{
  static constexpr int MAX_SUBPATH_LENGH = 5;
  static constexpr int MAX_PATH_LENGTH = MAX_SUBPATH_LENGH*2 + 2; 
  
  struct BdptHistory
  {
    Spectral measurement_contribution[MAX_PATH_LENGTH];
    double  pdf_product_up_to_node[MAX_PATH_LENGTH];
    BdptHistory()
    {
      pdf_product_up_to_node[0] = 1.;
      measurement_contribution[0] = Spectral{1.};
      for (int i=1; i<MAX_PATH_LENGTH; ++i)
      {
        pdf_product_up_to_node[i] = NaN;
        measurement_contribution[i] = Spectral{NaN};
      }
    }
    void Copy(int src, int dst)
    {
      measurement_contribution[dst] = measurement_contribution[src];
      pdf_product_up_to_node[dst] = pdf_product_up_to_node[src];
    }
    void MultiplyNodeValues(int idx, double pdf_factor, const Spectral &contrib_factor)
    {
      measurement_contribution[idx] *= contrib_factor;
      pdf_product_up_to_node[idx] *= pdf_factor;
    }
  };
  
public:  
  Bdpt(const Scene &_scene) : BaseAlgo(_scene)
  {
  }
  

  Spectral MakePrettyPixel() override
  {
    Spectral ret{0.};
    ret += BiDirectionPathTraceOfLength(2, 1);
    ret += BiDirectionPathTraceOfLength(2, 2);
    ret += BiDirectionPathTraceOfLength(2, 3);
    ret += BiDirectionPathTraceOfLength(3, 1);
    //ret += BiDirectionPathTraceOfLength(4, 1);
    return ret;
  }
  
  
  Spectral BiDirectionPathTraceOfLength(int eye_path_length, int light_path_length)
  {
    BdptHistory history_from_eye;
    BdptHistory history_from_light;

    if (eye_path_length <= 0)
      return Spectral(0.);
    if (light_path_length <= 0)
      return Spectral(0.);
    
    auto eye_node = TracePath(scene.GetCamera(), eye_path_length, history_from_eye);
    if (eye_node.type == PathNode::ESCAPED) return Spectral{0};
    
    const Light* the_light; double pmf_of_light; 
    std::tie(the_light, pmf_of_light) = PickLightUniform();
    auto light_node = TracePath(*the_light, light_path_length, history_from_light);
    if (light_node.type == PathNode::ESCAPED) return Spectral{0};
    
    Spectral measurement_contribution = 
      GeometryTermAndScattering(light_node, eye_node) *
      history_from_eye.measurement_contribution[eye_node.index] *
      history_from_light.measurement_contribution[light_node.index];
    double  path_pdf = history_from_eye.pdf_product_up_to_node[eye_node.index] *
                       history_from_light.pdf_product_up_to_node[light_node.index];
    Spectral path_sample_value = measurement_contribution / path_pdf;
    return path_sample_value;
  };
    
  
  PathNode TracePath(const RadianceOrImportance::EmitterSensor &emitter, int length, BdptHistory &history)
  {
      PathNode lastnode = InitNodeOne(
        emitter,
        history);
      while(lastnode.index < length && lastnode.type != PathNode::ESCAPED)
      {
        MakeNextNode(
          lastnode,
          history);
      }
      return lastnode;
  }
  
  
  PathNode InitNodeOne(
    const RadianceOrImportance::EmitterSensor &emitter,
    BdptHistory &history)
  {
    RadianceOrImportance::Sample sample = emitter.TakePositionSample(sampler);
    history.MultiplyNodeValues(0, 
                               sample.pdf_of_pos, 
                               sample.measurement_contribution);
    history.Copy(0, 1);
    // Return value optimization???
    return PathNode::MakeTypeOne(emitter, sample);
  }
  
  
  void MakeNextNode(
    PathNode &node, // updated
    BdptHistory &history)
  {
    auto dir_sample = node.SampleDirection(sampler);
    history.MultiplyNodeValues(
      node.index,
      dir_sample.pdf,
      dir_sample.measurement_contribution);
    history.Copy(node.index, node.index+1); // It's the same value since we have not sampled at the node we are going to add now yet.
    auto segment = RaySegment{dir_sample.ray_out, LargeNumber};
    auto last_hit = node.hitId();
    HitId hit = scene.Intersect(segment.ray, segment.length, last_hit);
    if (hit)
    {
      auto intersection = RaySurfaceIntersection{hit, segment};
      node = PathNode::MakeSurface(node.index+1, dir_sample.ray_out.dir, intersection);
    }
    else
    {
      node = PathNode::MakeEscaped(node.index+1);
    }
  }


  double DFactor(const PathNode &node, const Double3 &outgoing_dir)
  {
    assert(node.type == PathNode::SURFACE);
    return std::max(0., Dot(node.surface().intersection.normal, outgoing_dir));
  }
  
  
  Spectral GeometryTermAndScattering(
    const PathNode &node1,
    const PathNode &node2
  )
  {
    // We sampled for each of the nodes a random direction,
    // reflecting light or importance arriving from an infinitely 
    // distance source. Only if both direction coincided would there
    // be a contribution (unlikely). Such paths are better handled
    // by s=0,t=2 or s=2,t=0 paths.
    if (node1.type == PathNode::NODE1_DIRECTIONAL && 
        node2.type == PathNode::NODE1_DIRECTIONAL)
      return Spectral{0.};
    
    RaySegment segment;
    HitId hits_ignored[2];
    if (node1.type == PathNode::NODE1_DIRECTIONAL)
    {
      segment = RaySegment{{node2.Position(), node1.DirectionToEmitter()}, LargeNumber};
      hits_ignored[0] = node2.hitId();
    }
    else if (node2.type == PathNode::NODE1_DIRECTIONAL)
    {
      segment = RaySegment{{node1.Position(), node2.DirectionToEmitter()}, LargeNumber};
      hits_ignored[0] = node1.hitId();
    }
    else
    {
      segment = RaySegment::FromTo(node1.Position(), node2.Position());
      hits_ignored[0] = node1.hitId();
      hits_ignored[1] = node2.hitId();
    }
    if (scene.Occluded(segment.ray, segment.length, hits_ignored[0], hits_ignored[1]))
      return Spectral{0.};
    bool isAreaMeasure = segment.length<LargeNumber; // i.e. nodes are not directional.
    double geom_term = isAreaMeasure ? Sqr(1./segment.length) : 1.;
    if (node1.type == PathNode::SURFACE)
      geom_term *= DFactor(node1, segment.ray.dir);
    if (node2.type == PathNode::SURFACE)
      geom_term *= DFactor(node2, -segment.ray.dir);    

    Spectral scatter1 = node1.EvaluateDirection(segment.ray.dir);
    Spectral scatter2 = node2.EvaluateDirection(-segment.ray.dir);
    return geom_term * scatter1 * scatter2;
  }
};
#endif