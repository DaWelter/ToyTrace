#ifndef SCENE_HXX
#define SCENE_HXX

#include "perspectivecamera.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "bsptree.hxx"
#include "util.hxx"

#include <memory>

namespace boost { namespace filesystem {
  class path;
}};

struct RenderingParameters
{
  int pixel_x = {-1};
  int pixel_y = {-1};
  int width = {-1};
  int height = {-1};
  int num_threads = { -1 };
  int max_ray_depth = 25;
  int max_samples_per_pixel = {-1};
  std::string pt_sample_mode = {};
  std::string algo_name = {};
  std::vector<std::string> search_paths = { "" };
};



class Scene
{
  // parse an NFF file 'fileName', add all its primitives
  // to the specified group...
  //void ParseMesh(char *filename);
  friend class NFFParser;
  using Light = RadianceOrImportance::PointEmitter;
  using EnvironmentalRadianceField = RadianceOrImportance::EnvironmentalRadianceField;
  using AreaLight = RadianceOrImportance::AreaEmitter;

  std::unique_ptr<TreeNode> bsptree;
  // TODO: Manage memory ...
  std::vector<Primitive*> primitives;
  std::vector<Primitive*> emissive_primitives;
  std::vector<std::unique_ptr<Light>> lights;
  std::vector<std::unique_ptr<EnvironmentalRadianceField>> envlights;
  std::unique_ptr<Camera> camera;
  std::unique_ptr<Medium> empty_space_medium;
  std::unique_ptr<Shader> invisible_shader;
  
  Box boundingBox;
  
public:
  Scene()
    : camera(nullptr),
      empty_space_medium{new VacuumMedium()},
      invisible_shader{new InvisibleShader{}}
  {
  }

  void ParseNFF(const boost::filesystem::path &filename, RenderingParameters *render_params = nullptr);
  void ParseNFFString(const std::string &scenestr, RenderingParameters *render_params = nullptr);
  void ParseNFF(std::istream &is, RenderingParameters *render_params = nullptr);
  
  template<class CameraType, class... Args>
  void SetCamera(Args&&... args)
  {
    std::unique_ptr<CameraType> new_cam(new CameraType(std::forward<Args>(args)...));
    camera = std::move(new_cam);
  }
  
  void AddLight(std::unique_ptr<Light> light) 
  {
    lights.push_back(std::move(light));
  }
  
  int GetNumLights() const
  {
    return lights.size();
  }
  
  const Light& GetLight(int i) const
  {
    return *lights[i];
  }
  
  int GetNumEnvLights() const
  {
    return envlights.size();
  }
  
  int GetNumAreaLights() const
  {
    return emissive_primitives.size();
  }
  
  std::pair<const Primitive*, const AreaLight*> GetAreaLight(int i) const
  {
    return std::make_pair(emissive_primitives[i], emissive_primitives[i]->emitter);
  }
  
  const RadianceOrImportance::EnvironmentalRadianceField& GetEnvLight(int i) const
  {
    return *envlights[i];
  }
  
  const Camera& GetCamera() const
  {
    return *camera;
  }
  
  Camera& GetCamera()
  {
    return *camera;
  }

  bool HasCamera() const
  {
    return camera!=nullptr;
  }
  
  const Medium& GetEmptySpaceMedium() const
  {
    return *empty_space_medium;
  }
  
  const Shader& GetInvisibleShader() const
  {
    return *invisible_shader;
  }
  
  template<class PrimitiveType, class... Args>
  Primitive& AddPrimitive(Args&&... args)
  {
    // TODO: Manage memory ...
    auto* prim = new PrimitiveType(std::forward<Args>(args)...);
    primitives.push_back(prim);
    return *prim;
  }
  
  void MakePrimitiveEmissive(Primitive &prim, RadianceOrImportance::AreaEmitter &emitter)
  {
    emissive_primitives.push_back(&prim);
    prim.emitter = &emitter;
  }

  const Primitive& GetPrimitive(int i) const
  {
    return *primitives[i];
  }
  
  int GetNumPrimitives() const 
  {
    return primitives.size();
  }
  
  IntersectionCalculator MakeIntersectionCalculator() const
  {
    assert(bsptree);
    return IntersectionCalculator(*bsptree);
  }
  
//   HitId Intersect(const Ray &ray, double &ray_length) const
//   {
//     HitId hit;
//     bsptree.Intersect(ray, ray_length, hit);
//     return hit;
//   }
// 
//   void IntersectAll(const Ray &ray, double ray_length, HitVector &all_hits) const
//   {
//     bsptree.IntersectAll(ray, ray_length, all_hits);
//   }

//   bool Occluded(const Ray &ray, double ray_length, HitVector &all_hits) const
//   { 
//     bsptree.IntersectAll(ray, ray_length, all_hits);
//     
//   }
  
  void BuildAccelStructure()
  {   
    this->boundingBox = CalcBounds();
    bsptree = BuildBspTree(primitives, boundingBox);
  }
  
  void PrintInfo() const
  {
    std::cout << "Number of primitives: " << primitives.size() << std::endl;
    std::cout << "Number of lights: " << lights.size() << std::endl;
    std::cout << std::endl;
    std::cout << "bounding box min: "
              << boundingBox.min << std::endl;
    std::cout << "bounding box max: "
              << boundingBox.max << std::endl;
  }
  
  Box GetBoundingBox() const
  {
    return this->boundingBox;
  }
  
  Box CalcBounds() const
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
  }
};


#endif
