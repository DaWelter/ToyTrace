#ifndef SCENE_HXX
#define SCENE_HXX

#include "perspectivecamera.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "bsptree.hxx"

#include <memory>

class Scene
{
  // parse an NFF file 'fileName', add all its primitives
  // to the specified group...
  void ParseNFF(FILE *file, char *fileName);
  void ParseMesh(char *filename);
  
public:
  BSPTree bsptree;
  // TODO: Manage memory ...
  std::vector<Primitive*> primitives;
  std::vector<Light *> lights;
  Camera *camera;
  Double3 bgColor;
  Box boundingBox;

  Scene()
    : bgColor(Double3(0,0,0)),
      camera(new PerspectiveCamera(Double3(0,0,0),Double3(0,0,-1),
      Double3(0,1,0),60,640,480))
  {
  };
	
  void AddLight(std::unique_ptr<Light> light) 
  {
    // TODO: Manage memory ...
	  lights.push_back(light.release());
  }
  
  void AddPrimitive(std::unique_ptr<Primitive> primitive)
  {
    // TODO: Manage memory ...
    primitives.push_back(primitive.release());
  }

  // parse an NFF file 'fileName', store all its primitives
  void ParseNFF(char *fileName);
  
  
  std::tuple<Light*, double> PickLight(Sampler &sampler)
  {
    auto* light = lights[sampler.UniformInt(0, lights.size()-1)];
    double pmf_of_light = 1./lights.size();
    return std::make_tuple(light, pmf_of_light);
  }
  
  // trace the given ray and shade it and
  // return the color of the shaded ray
  Double3 RayTrace(Sampler &sampler)
  {
    if (lights.empty()) return Double3();
    
    RadianceOrImportance::DirectionalSample start = this->camera->TakeDirectionalSample(sampler);
    auto segment = RaySegment{{start.sample_pos, start.emission_dir}, LargeNumber};
    SurfaceHit hit;
    // TODO: give Intesect optionally a reference to the starting location as in 
    // the primitive that was hit and from which the ray now starts.
    if (bsptree.Intersect(segment.ray, segment.length, hit))
    {
      // TODO: get hit ID, instantiate Hit info.
      // Hit info computes normal, according to inbound ray direction, so that normal faces inbound ray.
      if(hit.primitive && hit.primitive->shader)
      {
        auto *shader = hit.primitive->shader;
        Light* light; double pmf_of_light; std::tie(light, pmf_of_light) = PickLight(sampler);
        
        Double3 hit_pos = segment.EndPoint();
        RadianceOrImportance::Sample light_sample = light->TakePositionSampleTo(hit_pos, sampler);
        
        RaySegment segment_to_light = RaySegment::FromTo(hit_pos, light_sample.sample_pos);
        
        // TODO: Hand over reference to last hit.
        if (!Occluded(segment_to_light.ray, segment_to_light.length))
        {
          Double3 brdf_value = shader->EvaluateBRDF(hit, segment_to_light.ray.dir);
          Double3 normal = hit.Normal(-segment.ray.dir);
          double d_factor = std::max(0., Dot(normal, segment_to_light.ray.dir));
          return (d_factor/light_sample.pdf_of_pos/segment_to_light.length) * Product(light_sample.value, brdf_value);
        }
      }
      else 
        return Double3(1,0,0);
    }
    else
      return bgColor; // ray missed geometric primitives
  };

  bool Occluded(const Ray &ray, double ray_length)
  { 
    SurfaceHit hit;
    return bsptree.Intersect(ray, ray_length, hit);
  }

  void BuildAccelStructure()
  {   
	  this->boundingBox = CalcBounds();
	  bsptree.Build(primitives, boundingBox);
  }
  
  void PrintInfo()
  {
    std::cout << std::endl;
    std::cout << "bounding box min: "
              << boundingBox.min << std::endl;
    std::cout << "bounding box max: "
              << boundingBox.max << std::endl;
  }
  
  Box CalcBounds()
  {
    Box scenebox;
    for (int i=0; i<(int)primitives.size(); i++) 
    {
      Box box = primitives[i]->CalcBounds();
      scenebox.Extend(box);
    }
    
    // primitives might lie on box borders
    scenebox.min -= Double3(Epsilon,Epsilon,Epsilon);
    scenebox.max += Double3(Epsilon,Epsilon,Epsilon);
    return scenebox;
  };
};


#endif
