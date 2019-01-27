#include "lightpicker_trivial.hxx"

TrivialLightPicker::TrivialLightPicker(const Scene& _scene)
    : scene{_scene}
{
  FindAreaLightGeometry();
  FindVolumeLightGeometry();
  const double nl = scene.GetNumLights();
  const double ne = scene.HasEnvLight() ? 1 : 0;
  const double na = arealight_refs.size();
  const double nv = volume_light_refs.size();
  // Why do I need to initialize like this to not get a negative number in IDX_PROB_POINT?
  // I mean when nl is zero and I assign an initializer list, the last entry is going to be like -something.e-42. Why???
  const double normalize_factor = 1. / (nl + ne + na + nv);
  emitter_type_selection_probabilities[IDX_PROB_ENV] = ne * normalize_factor;
  emitter_type_selection_probabilities[IDX_PROB_AREA] = na * normalize_factor;
  emitter_type_selection_probabilities[IDX_PROB_POINT] = nl * normalize_factor;
  emitter_type_selection_probabilities[IDX_PROB_VOLUME] = nv * normalize_factor;
}


void TrivialLightPicker::FindAreaLightGeometry()
{
  for (int geom_idx = 0; geom_idx < scene.GetNumGeometries(); ++geom_idx)
  {
    const auto &geom = scene.GetGeometry(geom_idx);
    for (int prim_idx = 0; prim_idx < geom.Size(); ++prim_idx)
    {
      auto &mat = scene.GetMaterialOf(geom_idx, prim_idx);
      if (mat.emitter)
      {
        arealight_refs.push_back(std::make_pair(geom_idx, prim_idx));
      }
    }
  }
}


void TrivialLightPicker::FindVolumeLightGeometry()
{
  for (int i = 0; i < scene.GetNumMaterials(); ++i)
  {
    const auto *medium = scene.GetMaterial(i).medium;
    if (medium && medium->is_emissive)
      volume_light_refs.push_back(i);
  }
}

