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
  Medium* medium = {nullptr};  // Within the geometry. I.e. on the other side of where the surface normal points.
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
public:
  using index_t = scene_index_t;
  static constexpr MaterialIndex DEFAULT_MATERIAL_INDEX{0};
  
private:
  friend class NFFParser;
  friend struct Scope;
  friend class EmbreeAccelerator;
  friend class PrimitiveIterator;
  using Light = RadianceOrImportance::PointEmitter;
  using EnvironmentalRadianceField = RadianceOrImportance::EnvironmentalRadianceField;
  using AreaLight = RadianceOrImportance::AreaEmitter;

  EmbreeAccelerator embreeaccelerator;
  EmbreeAccelerator embreevolumes;
  std::unique_ptr<Camera> camera;
  
  //ToyVector<Mesh> triangle_geometries;
  //ToyVector<Spheres> sphere_geometries;
  ToyVector<std::unique_ptr<Geometry>> geometries;
  ToyVector<Geometry*> emissive_geometries; // Indicies into geometries array

  //ToyVector<Mesh> triangle_volumes;
  //ToyVector<Spheres> sphere_volumes;
  ToyVector<std::unique_ptr<Geometry>> volumes;

  ToyVector<Material> materials;
  Medium* empty_space_medium;
  Shader* invisible_shader;
  Shader* default_shader;
  MaterialIndex default_material_index;
  MaterialIndex vacuum_material_index;

  ToyVector<std::unique_ptr<Shader>> shaders;
  ToyVector<std::unique_ptr<Medium>> media;
  ToyVector<std::unique_ptr<EnvironmentalRadianceField>> envlights;
  ToyVector<std::unique_ptr<Light>> lights; // point lights
  ToyVector<std::shared_ptr<Texture>> textures;
  Box boundingBox;
  std::unique_ptr<EnvironmentalRadianceField> envlight;
  
public:
  Scene();
  ~Scene();
  void ParseNFF(const boost::filesystem::path &filename, RenderingParameters *render_params = nullptr);
  void ParseNFFString(const std::string &scenestr, RenderingParameters *render_params = nullptr);
  void ParseNFF(std::istream &is, RenderingParameters *render_params = nullptr);
   

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
  

  bool HasLights() const;

  bool HasEnvLight() const { return envlights.size() > 0; }

  // TODO: Remove
  const RadianceOrImportance::EnvironmentalRadianceField& GetTotalEnvLight() const
  {
    assert(envlight.get() != nullptr);
    return *envlight;
  }

  index_t GetNumPointLights() const
  {
    return isize(lights);
  }

  const Light& GetPointLight(index_t i) const
  {
    return *lights[i];
  }

  index_t GetNumAreaLights() const;
  
  index_t GetAreaLightIndex(const PrimRef ref) const; // Return invalid index if has no light.

  PrimRef GetPrimitiveFromAreaLightIndex(index_t light) const;

  //index_t GetNumVolumeLights() const;

  //const Material& GetVolumeLight(index_t i) const;

  const Material& GetMaterialOf(index_t geom_idx, index_t prim_idx) const;
  
  const Material& GetMaterialOf(const PrimRef ref) const;
    
  const Medium& GetEmptySpaceMedium() const
  {
    return *this->empty_space_medium;
  }
  
  const Shader& GetInvisibleShader() const
  {
    return *this->invisible_shader;
  }


  inline index_t GetNumGeometries() const
  {
    return isize(geometries);
  }
  
  inline const Geometry& GetGeometry(index_t i) const
  {
    assert(i >= 0 && i < GetNumGeometries());
    return *geometries[i];
  }
  
  inline index_t GetNumMaterials() const
  {
    return (index_t)materials.size();
  }
  
  inline const Material& GetMaterial(index_t i) const
  {
    return materials[i];
  }
  
  bool FirstIntersectionEmbree(const Ray &ray, double tnear, double &ray_length, SurfaceInteraction &intersection) const;
  // TODO: Change this so I don't use raw pointers.
  //int IntersectionsWithVolumes(const Ray &ray, double tnear, double tfar, int *items, float *distances, const int buffer_size);

  void BuildAccelStructure();
  
  void PrintInfo() const;
  
  Box GetBoundingBox() const;

  void Append(const Geometry &geo, const Material &mat);

  void UpdateEmissiveIndexOffset();
};
