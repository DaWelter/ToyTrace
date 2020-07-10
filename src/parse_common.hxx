#pragma once
#include "scene.hxx"
#include "camera.hxx"
#include "infiniteplane.hxx"
#include "shader.hxx"
#include "util.hxx"
#include "atmosphere.hxx"
#include "light.hxx"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace scenereader
{

using RadianceOrImportance::AreaEmitter;

namespace fs = boost::filesystem;

using Transform = Eigen::Transform<double, 3, Eigen::Affine>;

// TODO: Rename this
template<class Thing>
class SymbolTable
{
  Thing currentThing;
  std::unordered_map<std::string, Thing> things;
  std::string name_of_this_table;
public:
  SymbolTable(const std::string &_name) :
    currentThing{}, name_of_this_table(_name)
  {}

  int size() const
  {
    return isize(things);
  }

  void activate(const std::string &name)
  {
    auto it = things.find(name);
    if (it != things.end())
    {
      currentThing = it->second;
    }
    else
    {
      char buffer[1024];
      std::snprintf(buffer, 1024, "Error: %s %s not defined. Define it in the NFF file prior to referencing it.", this->name_of_this_table.c_str(), name.c_str());
      throw std::runtime_error(buffer);
    }
  }

  void set_and_activate(const char* name, Thing thing)
  {
    currentThing = thing;
    things[name] = thing;
  }

  Thing operator()() const
  {
    return currentThing;
  }

  boost::optional<Thing> operator[](const char *name) const
  {
    auto it = things.find(name);
    if (it == things.end())
      return boost::none;
    else
      return it->second;
  }
};


class GlobalContext
{
  std::unordered_map<Material, MaterialIndex, Material::Hash> to_material_index;
  std::vector<fs::path> search_paths;
  Scene* scene;
  RenderingParameters* params;
  fs::path filename;

public:
  GlobalContext(Scene &scene_, RenderingParameters *params_, const fs::path &path_hint);
  auto& GetScene() const { return *scene; }
  auto* GetParams() const { return params; }
  auto GetFilename() const { return filename; }
  fs::path MakeFullPath(const fs::path &filename) const;
};


struct Scope
{
  SymbolTable<Shader*> shaders;
  SymbolTable<Medium*> mediums;
  SymbolTable<AreaEmitter*> areaemitters;
  std::unordered_map<std::string, Material> materials;
  Transform currentTransform;

  Scope() :
    shaders("Shader"),
    mediums("Medium"),
    areaemitters("AreaEmitter"),
    currentTransform{ Transform::Identity() }
  {
  }
};


void AddDefaultMaterials(Scope &scope, const Scene &scene);


inline auto NormalTrafo(const Transform &trafo)
{
  return trafo.linear().inverse().transpose().eval();
}



namespace assimp
{

using MaterialGetter = std::function<Material(const std::optional<std::string> &)>;

void Read(Scene &scene, Transform model_transform, MaterialGetter material_getter, const fs::path &filename_path);


} // namespace assimp

} // namespace scenereader
