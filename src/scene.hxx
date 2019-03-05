#pragma once
#include "embreeaccelerator.hxx"
#include "util.hxx"
#include "types.hxx"
#include "box.hxx"
#include "primitive.hxx"

#include <memory>
#include <boost/functional/hash.hpp>


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
  double initial_photon_radius = 0.01;
};


struct Material
{
  Shader* shader = {nullptr};
  Medium* medium = {nullptr};
  RadianceOrImportance::AreaEmitter *emitter = {nullptr};
  
  struct Hash
  {
    inline std::size_t operator()(const Material &key) const
    {
      std::size_t h = boost::hash_value((key.shader));
      boost::hash_combine(h, boost::hash_value((key.medium)));
      boost::hash_combine(h, boost::hash_value((void*)key.emitter));
      return h;
    }
  };
  
  friend inline bool operator==(const Material &a, const Material &b)
  {
    return (a.shader == b.shader) &&
          (a.medium == b.medium) &&
          (a.emitter == b.emitter);
  }
};


class Scene
{
  friend class NFFParser;
  friend struct Scope;
  friend class EmbreeAccelerator;
  friend class PrimitiveIterator;
  using Light = RadianceOrImportance::PointEmitter;
  using EnvironmentalRadianceField = RadianceOrImportance::EnvironmentalRadianceField;
  using AreaLight = RadianceOrImportance::AreaEmitter;

  EmbreeAccelerator embreeaccelerator;
  std::unique_ptr<Camera> camera;
  std::vector<Material> materials;
  // TODO: seperate emissive geometries to make things a little simpler?
  std::unique_ptr<Mesh> triangles;
  std::unique_ptr<Spheres> spheres;
  Geometry* new_primitives[2];
  Medium* empty_space_medium;
  Shader* invisible_shader;
  Shader* default_shader;
  MaterialIndex default_material_index;
  MaterialIndex vacuum_material_index;
  std::vector<std::unique_ptr<Shader>> shaders;
  std::vector<std::unique_ptr<Medium>> media;
  std::vector<std::unique_ptr<EnvironmentalRadianceField>> envlights;
  std::vector<std::unique_ptr<Light>> lights; // point lights
  std::vector<std::shared_ptr<Texture>> textures;
  Box boundingBox;
  std::unique_ptr<EnvironmentalRadianceField> envlight;
  
public:
  Scene();
  ~Scene();
  void ParseNFF(const boost::filesystem::path &filename, RenderingParameters *render_params = nullptr);
  void ParseNFFString(const std::string &scenestr, RenderingParameters *render_params = nullptr);
  void ParseNFF(std::istream &is, RenderingParameters *render_params = nullptr);
   
  int GetNumLights() const
  {
    return lights.size();
  }
  
  const Light& GetLight(int i) const
  {
    return *lights[i];
  }
  
  bool HasLights() const;
 
  bool HasEnvLight() const { return envlights.size()>0; }

  const RadianceOrImportance::EnvironmentalRadianceField& GetTotalEnvLight() const
  {
    assert(envlight.get() != nullptr);
    return *envlight;
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
  
  const Material& GetMaterialOf(int geom_idx, int prim_idx) const;
  
  const Material& GetMaterialOf(const PrimRef ref) const;
    
  const Medium& GetEmptySpaceMedium() const
  {
    return *this->empty_space_medium;
  }
  
  const Shader& GetInvisibleShader() const
  {
    return *this->invisible_shader;
  }

  void Append(const Mesh &other_mesh);
  
  void Append(const Spheres &other_spheres);
  
  inline int GetNumGeometries() const
  {
    return 2;
  }
  
  inline const Geometry& GetGeometry(int i) const
  {
    return *new_primitives[i];
  }
  
  inline int GetNumMaterials() const
  {
    return materials.size();
  }
  
  inline const Material& GetMaterial(int i) const
  {
    return materials[i];
  }
  
  bool FirstIntersectionEmbree(const Ray &ray, double tnear, double &ray_length, SurfaceInteraction &intersection) const;
  
  void BuildAccelStructure();
  
  void PrintInfo() const;
  
  Box GetBoundingBox() const;
};

/*
PrimitiveIterator::PrimitiveIterator(const PrimitiveIterator& other)
{

}

PrimitiveIterator::PrimitiveIterator(const Scene& scene)
{

}


bool PrimitiveIterator::operator bool() const
{

}

PrimitiveIterator::reference PrimitiveIterator::operator*() const
{

}

PrimitiveIterator PrimitiveIterator::operator++(int)
{
  ++current.index;
  if (current.index >= current.geom->Size())
  {
    current.index = 0;
    current.geom 
  }
}

bool PrimitiveIterator::operator==(const PrimitiveIterator& other)
{
  return this->geoms==other.geoms && this->current == other.current;
}*/

