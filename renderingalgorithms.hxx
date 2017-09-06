#include "scene.hxx"


class BaseAlgo
{
protected:
  const Scene &scene;
  Sampler sampler;

public:
  BaseAlgo(const Scene &_scene)
    : scene(_scene)
    {}
    
  std::tuple<const Light*, double> PickLightUniform()
  {
    const auto &light = scene.GetLight(sampler.UniformInt(0, scene.GetNumLights()-1));
    double pmf_of_light = 1./scene.GetNumLights();
    return std::make_tuple(&light, pmf_of_light);
  }
};



class Raytracing : public BaseAlgo
{
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
  
  
  Double3 LightingMeasurementOverPdf(const RaySurfaceIntersection &intersection)
  {
    Double3 ret{0., 0., 0.};
    if (scene.GetNumLights() > 0)
    {
      const Light* light; double pmf_of_light; std::tie(light, pmf_of_light) = PickLightUniform();
          
      RadianceOrImportance::Sample light_sample = light->TakePositionSample(sampler);
      
      RaySegment segment_to_light = MakeSegmentToLight(intersection.pos, light_sample);
      // TODO: Hand over reference to last hit.
      if (!scene.Occluded(segment_to_light.ray, segment_to_light.length))
      {
        const auto *shader = intersection.primitive->shader;
        Double3 brdf_value = shader->EvaluateBRDF(intersection, segment_to_light.ray.dir);
        double d_factor = std::max(0., Dot(intersection.normal, segment_to_light.ray.dir));
        
        ret = ((d_factor/light_sample.pdf_of_pos/segment_to_light.length) * 
          brdf_value.array() * light_sample.measurement_contribution.array()).matrix();
      }
    }
    return ret;
  }
  
  
  // trace the given ray and shade it and
  // return the color of the shaded ray
  Double3 MakePrettyPixel()
  {
    RadianceOrImportance::Sample start_pos_sample = scene.GetCamera().TakePositionSample(sampler);
    RadianceOrImportance::DirectionalSample start = scene.GetCamera().TakeDirectionalSampleFrom(start_pos_sample.pos, sampler);
    
    Double3 result = ((start_pos_sample.measurement_contribution.array() *
                      start.measurement_contribution.array()) / start_pos_sample.pdf_of_pos /  start.pdf_of_dir_given_pos).matrix();
    
    auto segment = RaySegment{start.ray_out, LargeNumber};
    HitId hit = scene.Intersect(segment.ray, segment.length);
    // TODO: give Intesect optionally a reference to the starting location as in 
    // the primitive that was hit and from which the ray now starts.
    if (hit)
    {
      RaySurfaceIntersection intersection{hit, segment};
      Double3 lights_factor = LightingMeasurementOverPdf(intersection);
      return (result.array() * lights_factor.array()).matrix();
    }
    else
      return scene.bgColor; // ray missed geometric primitives
  };
};



#if 0
enum PathNodeType : char 
{
  VOLUME,
  NODE1,
  SURFACE,
  NODE1_DIRECTIONAL
};

struct PathNode
{
  using RadianceOrImportance::DirectionalSample;
  
  PathNodeType type;
  char index;
  union
  {
    struct PathNodeSurface
    {
      RaySurfaceIntersection intersection;
    } surface_node;
    struct PathNodeOne
    {
      RadianceOrImportance::PathEndPoint &emitter;
    } end_node;
    // struct PathNodeVolume
    // ...
  };
  DirectionalSample SampleDirection(const Double3 &inbound_dir);
  DirectionalSample EvaluateDirection(const Double3 &inbound_dir, const Double3 &outbound_dir);
  
};



class Bdpt : public BaseAlgo
{

  double DFactor(const PathNode &node, const Double3 &outgoing_dir)
  {
    return (node.type == NODE_TYPE_SURFACE) ?
      std::max(0., Dot(node.surface_intersection.normal, outgoing_dir))
      :
      1.;
  }
  
  bool NeglectInverseDistance(const PathNode &node)
  {
    return node.type == NODE_TYPE_END && node.end_node_sample.is_extremely_far_away;
  }
  
  double GeometryTerm(const PathNode &org_node, 
                      const PathNode &end_node)
  {
    if(scene.Occluded(segment.ray, segment.length))
      return 0.;
    double d_factor_org = DFactor(org_node, segment.ray.dir);
    double d_factor_end = DFactor(end_node, -segment.ray.dir);
    double geom_term = d_factor_org * d_factor_end;
    if (!NeglectInverseDistance(org_node) && 
        !NeglectInverseDistance(end_node))
    {
      geom_term / segment.length;
    }
    return geom_term;
  }

  static constexpr int MAX_SUBPATH_LENGH = 5;
  static constexpr int MAX_PATH_LENGTH = MAX_SUBPATH_LENGH*2 + 2; 
  
  struct BdptHistory
  {
    Double3 measurment_contribution[MAX_PATH_LENGTH];
    double  pdf_product_up_to_node[MAX_PATH_LENGTH];
    BdptHistory()
    {
      memset(pdf_product_up_to_node, 0, sizeof(double)*MAX_PATH_LENGTH);
    }
    void Copy(int src, int dst)
    {
      measurment_contribution[dst] = measurment_contribution[src];
      //
    }
    void MultiplyNodeValues(int idx, double pdf_factor, const Double3 &contrib_factor)
    {
      //
    }
  };
  
  void InitNodeOne(
    const RadianceOrImportance::PathEndPoint &emitter,
    BdptHistory &history,
    PathNode &node)
  {
    RadianceOrImportance::Sample sample = emitter.TakePositionSample(sampler);
    history.MultiplyNodeValues(0, 
                               sample.pdf_of_pos, 
                               sample.measurement_contribution);
    history.Copy(0, 1);
    node = PathNode::MakeTypeOne(emitter, sample);
  }
  
  bool GoToNextNode(
    Double3 &dir_in_out,  // updated
    PathNode &node, // updated
    BdptHistory &history)
  {
    auto dir_sample = node.SampleDirection(dir_in_out);
    history.MultiplyNodeValues(
      node.index,
      dir_sample.pdf_of_dir_given_pos,
      dir_sample.measurement_contribution);
    history.Copy(node.index, node.index+1); // It's the same value since we have not sampled at the node we are going to add now yet.
    dir_in_out = dir_sample.ray_out.dir;
    auto segment = RaySegment{dir_sample.ray_out, LargeNumber};
    HitId hit = scene.Intersect(segment.ray, segment.length);
    if (hit)
    {
      SurfaceIntersection intersection = RaySurfaceIntersection(hit, segment.ray);
      node = PathNode::MakeSurface(node.index+1, intersection);
      return true;
    }
    return false;
  }
  
  Double3 BiDirectionPathTraceOfLength(int eye_path_length, int light_path_length)
  {
    BdptHistory history_from_eye;
    BdptHistory history_from_light;
    PathNode eye_node, light_node;
    
    if (eye_path_length <= 0)
      return Double3(0.);
    if (light_path_length <= 0)
      return Double3(0.);
    
    {
      bool keep_going = true;
      auto dir = Double3{0,0,0};
      InitNodeOne(
        scene.GetCamera(),
        history_from_eye,
        eye_node);
      while(eye_node.index < eye_path_length && keep_going)
      {
        keep_going = GoToNextNode(
          dir,
          eye_node,
          history_from_eye);
      }
    }
    
    {
      bool keep_going = true;
      auto dir = Double3{0,0,0};
      const Light* the_light; double pmf_of_light; 
      std::tie(the_light, pmf_of_light) = PickLightUniform(scene, sampler);
      
      InitNodeOne(
        *the_light,
        history_from_light,
        light_node);
      while(light_node.index < light_path_length && keep_going)
      {
        keep_going = GoToNextNode(
          dir,
          light_node,
          history_from_light);
      }
    }
    
    Double3 measurement_contribution = 
      GeometryTerm(light_node, eye_node) *
      history_from_eye.measurment_contribution[eye_node.index] *
      history_from_light.measurment_contribution[light_node.index];
    double  path_pdf = history_from_eye.pdf_product_up_to_node[eye_node.index] *
                       history_from_light.pdf_product_up_to_node[light_node.index];
    Double3 path_sample_value = measurement_contribution / path_pdf;
    return path_sample_value;
  };
};
#endif