#ifndef SCENE_HXX
#define SCENE_HXX

#include "perspectivecamera.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "bsptree.hxx"

#include <memory>


struct RenderingParameters
{
  int pixel_x = {-1};
  int pixel_y = {-1};
  int width = {-1};
  int height = {-1};
  int num_threads = { -1 };
  int max_ray_depth = 25;
  int max_samples_per_pixel = {-1};
};



class Scene
{
  // parse an NFF file 'fileName', add all its primitives
  // to the specified group...
  //void ParseMesh(char *filename);
  friend class NFFParser;

  BSPTree bsptree;
  // TODO: Manage memory ...
  std::vector<Primitive*> primitives;
  std::vector<Light *> lights;
  std::unique_ptr<Camera> camera;
  std::unique_ptr<Medium> empty_space_medium;
  
  Box boundingBox;
  
public:
  Spectral bgColor;

  Scene()
    : bgColor{0,0,0},
      camera(nullptr),
      empty_space_medium{new VacuumMedium()}
  {
  }

  void ParseNFF(const char *fileName, RenderingParameters *render_params = nullptr);
  
  template<class CameraType, class... Args>
  void SetCamera(Args&&... args)
  {
    std::unique_ptr<CameraType> new_cam(new CameraType(std::forward<Args>(args)...));
    camera = std::move(new_cam);
  }
  
  void AddLight(std::unique_ptr<Light> light) 
  {
    // TODO: Manage memory ...
    lights.push_back(light.release());
  }
  
  int GetNumLights() const
  {
    return lights.size();
  }
  
  const Light& GetLight(int i) const
  {
    return *lights[i];
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
    
  template<class PrimitiveType, class... Args>
  Primitive& AddPrimitive(Args&&... args)
  {
    // TODO: Manage memory ...
    auto* prim = new PrimitiveType(std::forward<Args>(args)...);
    primitives.push_back(prim);
    return *prim;
  }

  const Primitive& GetPrimitive(int i) const
  {
    return *primitives[i];
  }
  
  int GetNumPrimitives() const 
  {
    return primitives.size();
  }
  
  bool Occluded(const Ray &ray, double ray_length, const HitId &to_ignore1 = HitId(), const HitId &to_ignore2 = HitId()) const
  { 
    HitId hit;
    return bsptree.Intersect(ray, ray_length, hit, to_ignore1, to_ignore2);
  }

  HitId Intersect(const Ray &ray, double &ray_length, const HitId &to_ignore1 = HitId(), const HitId &to_ignore2 = HitId()) const
  {
    HitId hit;
    bsptree.Intersect(ray, ray_length, hit, to_ignore1, to_ignore2);
    return hit;
  }
  
  void BuildAccelStructure()
  {   
	  this->boundingBox = CalcBounds();
	  bsptree.Build(primitives, boundingBox);
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
