#include "scene.hxx"
#include "ray.hxx"
#include "shader.hxx"
#include "camera.hxx"
#include "light.hxx"

Scene::Scene()
    : camera(nullptr)
{ 
    auto vac_medium = std::make_unique<VacuumMedium>();
    auto inv_shader = std::make_unique<InvisibleShader>();
    auto def_shader = std::make_unique<DiffuseShader>(Color::RGBToSpectrum({0.8_rgb, 0.8_rgb, 0.8_rgb}), nullptr);
    this->empty_space_medium = vac_medium.get();
    this->invisible_shader = inv_shader.get();
    this->default_shader = def_shader.get();
    
    materials.push_back(Material{def_shader.get(), vac_medium.get()});
    default_material_index = MaterialIndex(isize(materials)-1);
    materials.push_back(Material{inv_shader.get(), vac_medium.get()});
    vacuum_material_index = MaterialIndex(isize(materials)-1);

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
  return materials[value(GetGeometry(geom_idx).material_index)];
}

const Material& Scene::GetMaterialOf(const PrimRef ref) const
{
  assert((bool)ref);
  return materials[value(ref.geom->material_index)];
}

namespace {

Geometry* FindMatchingGeo(ToyVector<std::unique_ptr<Geometry>> &v, MaterialIndex searchIndex, Geometry::Type searchType)
{
  auto other_geo_it = std::find_if(v.begin(), v.end(), [&](const std::unique_ptr<Geometry> &g)
  {
    return g->material_index == searchIndex && g->type == searchType;
  });
  return other_geo_it == v.end() ? nullptr : other_geo_it->get();
}

}


void Scene::Append(const Geometry &geo, const Material &mat)
{
  const bool is_emissive = mat.emitter != nullptr;
  const bool is_volume = mat.shader == nullptr;
  assert(!(is_emissive && is_volume));

  Geometry* existing_geom_with_mat = nullptr;
  MaterialIndex material_index{ -1 };

  // First look if we have a matching material already.
  auto mat_it = std::find(materials.begin(), materials.end(), mat);
  if (mat_it == materials.end())
  {
    materials.push_back(mat);
    material_index = MaterialIndex(isize(materials) - 1);
  }
  else
  {
    material_index = MaterialIndex(mat_it - materials.begin());
    existing_geom_with_mat = FindMatchingGeo(is_volume ? volumes : geometries, material_index, geo.type);
  }

  // Add new geometry or append to existing.
  if (existing_geom_with_mat)
  {
    existing_geom_with_mat->Append(geo);
  }
  else
  {
    (is_volume ? volumes : geometries).push_back(geo.Clone());
    if (is_emissive)
      emissive_geometries.push_back(geometries.back().get());
  }

  if (is_emissive)
    UpdateEmissiveIndexOffset();
}


void Scene::BuildAccelStructure()
{
  for (int i = 0; i < isize(geometries); ++i)
    embreeaccelerator.Add(*geometries[i]);
  embreeaccelerator.Build();
  for (int i = 0; i < isize(volumes); ++i)
    embreevolumes.Add(*volumes[i]);
  embreevolumes.Build();
  this->boundingBox = embreeaccelerator.GetSceneBounds();
  this->boundingBox.Extend(embreevolumes.GetSceneBounds());
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


Scene::index_t Scene::GetNumAreaLights() const
{
  return isize(emissive_geometries);
}



Scene::index_t Scene::GetAreaLightIndex(const PrimRef ref) const
{
  assert(ref.geom != nullptr);
  assert(ref.index >= 0 && ref.geom->Size());
  return ref.geom->light_num_offset + ref.index;
}

PrimRef Scene::GetPrimitiveFromAreaLightIndex(Scene::index_t light) const
{
  // find first element that _Val is before
  auto iter = std::upper_bound(emissive_geometries.begin(), emissive_geometries.end(), light, [&](Scene::index_t light_idx, const Geometry* geo) {
    return light_idx < geo->light_num_offset;
  });
  assert(iter - emissive_geometries.begin() > 0);
  --iter;
  light -= (*iter)->light_num_offset;
  assert(0 <= light && light < (*iter)->Size());
  return PrimRef{ *iter, light };
}

void Scene::UpdateEmissiveIndexOffset()
{
  index_t value = 0;
  for (int i = 0; i<isize(emissive_geometries)-1; ++i)
  {
    auto &geo = *emissive_geometries[i];
    geo.light_num_offset = value;
    value += geo.Size();
  }
}


//Scene::index_t Scene::GetNumVolumeLights() const
//{
//  return index_t();
//}
//
//const Material & Scene::GetVolumeLight(index_t i) const
//{
//  // TODO: insert return statement here
//}
