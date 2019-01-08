#include "scene.hxx"
#include "ray.hxx"
#include "shader.hxx"
#include "camera.hxx"
#include "light.hxx"

Scene::Scene()
    : camera(nullptr),
      triangles{new Mesh{0, 0}},
      spheres{new Spheres{}}
{
    new_primitives[0] = triangles.get();
    new_primitives[1] = spheres.get();
  
    auto vac_medium = std::make_unique<VacuumMedium>();
    auto inv_shader = std::make_unique<InvisibleShader>();
    auto def_shader = std::make_unique<DiffuseShader>(Color::RGBToSpectrum({0.8_rgb, 0.8_rgb, 0.8_rgb}), nullptr);
    this->empty_space_medium = vac_medium.get();
    this->invisible_shader = inv_shader.get();
    this->default_shader = def_shader.get();
    
    materials.push_back(Material{def_shader.get(), vac_medium.get()});
    default_material_index = MaterialIndex(materials.size()-1);
    materials.push_back(Material{inv_shader.get(), vac_medium.get()});
    vacuum_material_index = MaterialIndex(materials.size()-1);

    media.push_back(std::move(vac_medium));
    shaders.push_back(std::move(inv_shader));      
    shaders.push_back(std::move(def_shader));
}


Scene::~Scene()
{}


bool Scene::FirstIntersectionEmbree(const Ray &ray, double tnear, double &ray_length, SurfaceInteraction &ia) const
{
  return embreeaccelerator.FirstIntersection(ray, tnear, ray_length, ia);
}


const Material& Scene::GetMaterialOf(int geom_idx, int prim_idx) const
{
  assert(geom_idx>=0 && geom_idx<GetNumGeometries());
  auto *geom = new_primitives[geom_idx];
  assert(prim_idx>=0 && prim_idx<geom->Size());
  return materials[value(geom->material_indices[prim_idx])];
}

const Material& Scene::GetMaterialOf(const PrimRef ref) const
{
  assert((bool)ref);
  return materials[value(ref.geom->material_indices[ref.index])];
}


void Scene::Append(const Mesh &other_mesh)
{
  triangles->Append(other_mesh);
}

void Scene::Append(const Spheres &other_spheres)
{
  spheres->Append(other_spheres);
}

void Scene::BuildAccelStructure()
{   
  embreeaccelerator.Add(*triangles);
  embreeaccelerator.Add(*spheres);
  embreeaccelerator.Build();
  this->boundingBox = embreeaccelerator.GetSceneBounds();
}

void Scene::PrintInfo() const
{
  std::cout << "Number of geometries: " << GetNumGeometries() << std::endl;
  std::cout << "Number of lights: " << lights.size() << std::endl;
  std::cout << std::endl;
  std::cout << "bounding box min: "
            << boundingBox.min << std::endl;
  std::cout << "bounding box max: "
            << boundingBox.max << std::endl;
}

Box Scene::GetBoundingBox() const
{
  return this->boundingBox;
}

bool Scene::HasLights() const
{
  if (envlights.size())
    return true;
  if (lights.size())
    return true;
  // This does not check if the medium or the emissive surface are actually used though.
  for (int i=0; i<materials.size(); ++i)
  {
    if (materials[i].emitter)
      return true;
    if (materials[i].medium && materials[i].medium->is_emissive)
      return true;
  }
  return false;
}
