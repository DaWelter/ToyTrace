#pragma once

#include <fstream>

#include "scene.hxx"
#include "util.hxx"
#include "shader_util.hxx"

// WARNING: THIS IS NOT THREAD SAFE. DON'T WRITE TO THE SAME FILE FROM MULTIPLE THREADS!
class PathLogger
{
  std::ofstream file;
  int max_num_paths;
  int num_paths_written;
  int total_path_index;
  void PreventLogFromGrowingTooMuch();
public:
  enum SegmentType : int {
    EYE_SEGMENT = 'e',
    LIGHT_PATH = 'l',
    EYE_LIGHT_CONNECTION = 'c',
  };
  enum ScatterType : int {
    SCATTER_VOLUME = 'v',
    SCATTER_SURFACE = 's'
  };
  PathLogger();
  void AddSegment(const Double3 &x1, const Double3 &x2, const Spectral3 &beta_at_end_before_scatter, SegmentType type);
  void AddScatterEvent(const Double3 &pos, const Double3 &out_dir, const Spectral3 &beta_after, ScatterType type);
  void NewTrace(const Spectral3 &beta_init);
};


PathLogger::PathLogger()
  : file{"paths.log"},// {"paths2.log"}},
    max_num_paths{100},
    num_paths_written{0},
    total_path_index{0}
{
  file.precision(std::numeric_limits<double>::digits10);
}


void PathLogger::PreventLogFromGrowingTooMuch()
{
  if (num_paths_written > max_num_paths)
  {
    //files[0].swap(files[1]);
    file.seekp(std::ios_base::beg);
    num_paths_written = 0;
  }
}


void PathLogger::AddSegment(const Double3 &x1, const Double3 &x2, const Spectral3 &beta_at_end_before_scatter, SegmentType type)
{
  const auto &b = beta_at_end_before_scatter;
  file << static_cast<char>(type) << ", "
       << x1[0] << ", " << x1[1] << ", " << x1[2] << ", "
       << x2[0] << ", " << x2[1] << ", " << x2[2] << ", "
       <<  b[0] << ", " <<  b[1] << ", " <<  b[2] << "\n";
  file.flush();
}

void PathLogger::AddScatterEvent(const Double3 &pos, const Double3 &out_dir, const Spectral3 &beta_after, ScatterType type)
{
  const auto &b = beta_after;
  file << static_cast<char>(type) << ", "
       <<     pos[0] << ", " <<     pos[1] << ", " <<     pos[2] << ", "
       << out_dir[0] << ", " << out_dir[1] << ", " << out_dir[2] << ", "
       <<       b[0] << ", " <<       b[1] << ", " <<       b[2] << "\n";
  file.flush();
}


void PathLogger::NewTrace(const Spectral3 &beta_init)
{
  ++total_path_index;
  ++num_paths_written;
  PreventLogFromGrowingTooMuch();
  const auto &b = beta_init;
  file << "n, " << total_path_index << ", " << b[0] << ", " <<  b[1] << ", " <<  b[2] << "\n";
  file.flush();
}


#ifndef NDEBUG
#  define PATH_LOGGING(x)
#else
#  define PATH_LOGGING(x)
#endif


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
  const Medium* findMediumOfHighestPriority() const;
  bool remove(const Medium *medium);
  bool insert(const Medium *medium);
public:
  explicit MediumTracker(const Scene &_scene);
  MediumTracker(const MediumTracker &_other) = default;
  void initializePosition(const Double3 &pos, IntersectionCalculator &intersector);
  void initializePosition(const Double3 &pos, IntersectionCalculator &&intersector)
  { // Take control of the temporary that was passed in here. Now it has a name and
    // therefore repeated calling of initializePosition will invoke the one with the
    // lvalue in the parameter list.
    initializePosition(pos, intersector);
  }
  void goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection &intersection);
  const Medium& getCurrentMedium() const;
};


MediumTracker::MediumTracker(const Scene& _scene)
  : scene{_scene}, current{nullptr},
    media{} // Note: Ctor zero-initializes the media array.
{
}


const Medium& MediumTracker::getCurrentMedium() const
{
  assert(current);
  return *current;
}


void MediumTracker::initializePosition(const Double3& pos, IntersectionCalculator &intersector)
{
  std::fill(media.begin(), media.end(), nullptr);
  current = &scene.GetEmptySpaceMedium();
  {
    const Box bb = scene.GetBoundingBox();
    // The InBox check is important. Otherwise I would not know how long to make the ray.
    if (bb.InBox(pos))
    {
      double distance_to_go = 2. * (bb.max-bb.min).maxCoeff(); // Times to due to the diagonal.
      Double3 start = pos;
      start[0] += distance_to_go;
      RaySegment seg{{start, {-1., 0., 0.}}, distance_to_go};
      intersector.All(seg.ray, seg.length);
      for (const auto &hit : intersector.Hits())
      {
        RaySurfaceIntersection intersection(hit, seg);
        goingThroughSurface(seg.ray.dir, intersection);        
      }
    }
  }
}


void MediumTracker::goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection& intersection)
{
  if (Dot(dir_of_travel, intersection.volume_normal) < 0)
    enterVolume(intersection.primitive().medium);
  else
    leaveVolume(intersection.primitive().medium);
}


const Medium* MediumTracker::findMediumOfHighestPriority() const
{
  const Medium* medium_max_prio = &scene.GetEmptySpaceMedium();
  for (int i = 0; i < media.size(); ++i)
  {
    medium_max_prio = (media[i] &&  medium_max_prio->priority < media[i]->priority) ?
                        media[i] : medium_max_prio;
  }
  return medium_max_prio;
}


bool MediumTracker::remove(const Medium *medium)
{
  bool is_found = false;
  for (int i = 0; (i < media.size()) && !is_found; ++i)
  {
    is_found = media[i] == medium;
    media[i] = is_found ? nullptr : media[i];
  }
  return is_found;
}


bool MediumTracker::insert(const Medium *medium)
{
  bool is_place_empty = false;
  for (int i = 0; (i < media.size()) && !is_place_empty; ++i)
  {
    is_place_empty = media[i]==nullptr;
    media[i] = is_place_empty ? medium : media[i];
  }
  return is_place_empty;
}


void MediumTracker::enterVolume(const Medium* medium)
{
  // Set one of the array entries that is currently nullptr to the new medium pointer.
  // But only if there is room left. Otherwise the new medium is ignored.
  bool was_inserted = insert(medium);
  current = (was_inserted && medium->priority > current->priority) ? medium : current;
}


void MediumTracker::leaveVolume(const Medium* medium)
{
  // If the medium is in the media stack, remove it.
  // And also make the medium of highest prio the current one.
  remove(medium);
  if (medium == current)
  {
    current = findMediumOfHighestPriority();
  }
}


class Spectral3ImageBuffer
{
  std::vector<int> count;
  std::vector<RGB, boost::alignment::aligned_allocator<RGB, 128> >  accumulator;
  int xres, yres;
public:
  Spectral3ImageBuffer(int _xres, int _yres)
    : xres(_xres), yres(_yres)
  {
    int sz = _xres * _yres;
    assert(sz > 0);
    count.resize(sz, 0);
    accumulator.resize(sz, RGB::Zero());
  }
  
  void Insert(int pixel_index, const RGB &value)
  {
    assert (pixel_index >= 0 && pixel_index < count.size() && pixel_index < accumulator.size());
    ++count[pixel_index];
    accumulator[pixel_index] += value; 
  }
  
  void ToImage(Image &dest, int ystart, int yend) const
  {
    assert (ystart >= 0 && yend>= ystart && yend <= dest.height());
    for (int y=ystart; y<yend; ++y)
    for (int x=0; x<xres; ++x)
    {
      int pixel_index = xres * y + x;
      RGB average = accumulator[pixel_index]/Color::RGBScalar(count[pixel_index]);
      Image::uchar rgb[3];
      bool isfinite = average.isFinite().all();
      //bool iszero = (accumulator[pixel_index]==0.).all();
      average = average.max(0._rgb).min(1._rgb);
      if (isfinite)
      {
        for (int i=0; i<3; ++i)
          rgb[i] = value(Color::LinearToSRGB(average[i])*255.999_rgb);
        dest.set_pixel(x, dest.height() - 1 - y, rgb[0], rgb[1], rgb[2]);
      }
    }
  }
};



class RadianceEstimatorBase
{
protected:
  const Scene &scene;
  IntersectionCalculator intersector;
  Sampler sampler;

public:
  RadianceEstimatorBase(const Scene &_scene)
    : scene{_scene}, intersector{scene.MakeIntersectionCalculator()}
    {}

  virtual ~RadianceEstimatorBase() {}

  std::tuple<const Light*, double> PickLightUniform()
  {
    const auto &light = scene.GetLight(sampler.UniformInt(0, scene.GetNumLights()-1));
    double pmf_of_light = 1./scene.GetNumLights();
    return std::make_tuple(&light, pmf_of_light);
  }
  
  
  Spectral3 EvaluateEnvironmentalRadianceField(const Double3 &viewing_dir, const RadianceOrImportance::LightPathContext &context)
  {
    Spectral3 environmental_radiance{0.};
    for (int i=0; i<scene.GetNumLights(); ++i)
    {
      const auto &light = scene.GetLight(i);
      if (light.is_environmental_radiance_distribution) // What an aweful hack. Should there not be an env map class or similar?
      {
        environmental_radiance += light.EvaluatePositionComponent(
          -viewing_dir,
          context,
          nullptr);
      }
    }
    return environmental_radiance;
  }
  
  
  Spectral3 TransmittanceEstimate(RaySegment seg, MediumTracker &medium_tracker, const PathContext &context)
  {
    // TODO: Russian roulette.
    Spectral3 result{1.};
    auto SegmentContribution = [&seg, &medium_tracker, &context, this](double t1, double t2) -> Spectral3
    {
      RaySegment subseg{{seg.ray.PointAt(t1), seg.ray.dir}, t2-t1};
      return medium_tracker.getCurrentMedium().EvaluateTransmission(subseg, sampler, context);
    };

    intersector.All(seg.ray, seg.length);
    double t = 0;
    for (const auto &hit : intersector.Hits())
    {
      RaySurfaceIntersection intersection{hit, seg};
      result *= intersection.shader().EvaluateBSDF(-seg.ray.dir, intersection, seg.ray.dir, context, nullptr);
      if (result.isZero())
        return result;
      result *= SegmentContribution(t, hit.t);
      medium_tracker.goingThroughSurface(seg.ray.dir, intersection);
      t = hit.t;
    }
    result *= SegmentContribution(t, seg.length);
    return result;
  }

  
  struct CollisionData
  {
    CollisionData (const Ray &ray) :
      segment{ray, LargeNumber},
      hit{},
      smpl{}
    {}
    RaySegment segment;
    Medium::InteractionSample smpl;
    HitId hit;
  };
  

  void TrackToNextInteraction(
    CollisionData &collision,
    MediumTracker &medium_tracker,
    const PathContext &context)
  {
    Spectral3 total_weight{1.};
    auto &segment = collision.segment;
    while (true)
    {
      const Medium& medium = medium_tracker.getCurrentMedium();
      
      const auto &hit = collision.hit = intersector.First(segment.ray, segment.length);
      const auto &medium_smpl = collision.smpl = medium.SampleInteractionPoint(segment, sampler, context);
      total_weight *= medium_smpl.weight;
      
      if (medium_smpl.t >= segment.length && hit && hit.primitive->shader->IsPassthrough())
      {
        RaySurfaceIntersection intersection{hit, segment};
        medium_tracker.goingThroughSurface(segment.ray.dir, intersection);
        segment.ray.org = intersection.pos;
        segment.ray.org += AntiSelfIntersectionOffset(intersection, RAY_EPSILON, segment.ray.dir);
        segment.length = LargeNumber;
      }
      else
      {
        collision.smpl.weight = total_weight;
        return;
      }
    }
  }
};



class BaseAlgo : public RadianceEstimatorBase
{
public:
  BaseAlgo(const Scene &scene) : RadianceEstimatorBase{scene} {}
  virtual RGB MakePrettyPixel(int pixel_index) = 0;
};



class NormalVisualizer : public BaseAlgo
{
public:
  NormalVisualizer(const Scene &_scene) : BaseAlgo(_scene) 
  {
  }
  
  RGB MakePrettyPixel(int pixel_index) override
  {
    auto cam_sample = TakeRaySample(
      scene.GetCamera(), pixel_index, sampler,
      RadianceOrImportance::LightPathContext(Color::LambdaIdxClosestToRGBPrimaries())
    );
    
    RaySegment seg{cam_sample.ray_out, LargeNumber};
    HitId hit = intersector.First(seg.ray, seg.length);
    if (hit)
    {
      RaySurfaceIntersection intersection{hit, seg};
      RGB col = (intersection.normal.array() * 0.5 + 0.5).cast<Color::RGBScalar>();
      return col;
    }
    else
    {
      return RGB::Zero();
    }
  };
};


using  AlgorithmParameters = RenderingParameters;



#if 0
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
  Spectral3 EvaluateDirection(const Double3 &outbound_dir) const;
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


Spectral3 PathNode::EvaluateDirection(const Double3& outbound_dir) const
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
    Spectral3 measurement_contribution[MAX_PATH_LENGTH];
    double  pdf_product_up_to_node[MAX_PATH_LENGTH];
    BdptHistory()
    {
      pdf_product_up_to_node[0] = 1.;
      measurement_contribution[0] = Spectral3{1.};
      for (int i=1; i<MAX_PATH_LENGTH; ++i)
      {
        pdf_product_up_to_node[i] = NaN;
        measurement_contribution[i] = Spectral3{NaN};
      }
    }
    void Copy(int src, int dst)
    {
      measurement_contribution[dst] = measurement_contribution[src];
      pdf_product_up_to_node[dst] = pdf_product_up_to_node[src];
    }
    void MultiplyNodeValues(int idx, double pdf_factor, const Spectral3 &contrib_factor)
    {
      measurement_contribution[idx] *= contrib_factor;
      pdf_product_up_to_node[idx] *= pdf_factor;
    }
  };
  
public:  
  Bdpt(const Scene &_scene) : BaseAlgo(_scene)
  {
  }
  

  Spectral3 MakePrettyPixel() override
  {
    Spectral3 ret{0.};
    ret += BiDirectionPathTraceOfLength(2, 1);
    ret += BiDirectionPathTraceOfLength(2, 2);
    ret += BiDirectionPathTraceOfLength(2, 3);
    ret += BiDirectionPathTraceOfLength(3, 1);
    //ret += BiDirectionPathTraceOfLength(4, 1);
    return ret;
  }
  
  
  Spectral3 BiDirectionPathTraceOfLength(int eye_path_length, int light_path_length)
  {
    BdptHistory history_from_eye;
    BdptHistory history_from_light;

    if (eye_path_length <= 0)
      return Spectral3(0.);
    if (light_path_length <= 0)
      return Spectral3(0.);
    
    auto eye_node = TracePath(scene.GetCamera(), eye_path_length, history_from_eye);
    if (eye_node.type == PathNode::ESCAPED) return Spectral3{0};
    
    const Light* the_light; double pmf_of_light; 
    std::tie(the_light, pmf_of_light) = PickLightUniform();
    auto light_node = TracePath(*the_light, light_path_length, history_from_light);
    if (light_node.type == PathNode::ESCAPED) return Spectral3{0};
    
    Spectral3 measurement_contribution = 
      GeometryTermAndScattering(light_node, eye_node) *
      history_from_eye.measurement_contribution[eye_node.index] *
      history_from_light.measurement_contribution[light_node.index];
    double  path_pdf = history_from_eye.pdf_product_up_to_node[eye_node.index] *
                       history_from_light.pdf_product_up_to_node[light_node.index];
    Spectral3 path_sample_value = measurement_contribution / path_pdf;
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
                               sample.pdf, 
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
  
  
  Spectral3 GeometryTermAndScattering(
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
      return Spectral3{0.};
    
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
      return Spectral3{0.};
    bool isAreaMeasure = segment.length<LargeNumber; // i.e. nodes are not directional.
    double geom_term = isAreaMeasure ? Sqr(1./segment.length) : 1.;
    if (node1.type == PathNode::SURFACE)
      geom_term *= DFactor(node1, segment.ray.dir);
    if (node2.type == PathNode::SURFACE)
      geom_term *= DFactor(node2, -segment.ray.dir);    

    Spectral3 scatter1 = node1.EvaluateDirection(segment.ray.dir);
    Spectral3 scatter2 = node2.EvaluateDirection(-segment.ray.dir);
    return geom_term * scatter1 * scatter2;
  }
};
#endif
