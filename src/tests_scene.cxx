#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

#ifdef HAVE_JSON
#include <rapidjson/document.h>
#endif

#include "sampler.hxx"
#include "scene.hxx"



namespace
{

void CheckSceneParsedWithScopes(const Scene &scene)
{
  ASSERT_EQ(scene.GetNumGeometries(), 3);
  ASSERT_TRUE(scene.GetGeometry(0).type == Geometry::PRIMITIVES_SPHERES);
  ASSERT_TRUE(scene.GetGeometry(1).type == Geometry::PRIMITIVES_SPHERES);
  ASSERT_TRUE(scene.GetGeometry(2).type == Geometry::PRIMITIVES_SPHERES);
  auto Get = [&scene](int i) -> std::pair<const Spheres*, int>
  {
    int geo_index = 0;
    int prim_index = 0;
    switch (i)
    {
    case 1:
      geo_index = 1;
      break;
    case 2:
      geo_index = 2;
      break;
    case 3:
      geo_index = 1;
      prim_index = 1;
      break;
    default:
      break;
    }
    return std::make_pair(
      static_cast<const Spheres*>(&scene.GetGeometry(geo_index)),
      prim_index);
  };
  auto GetCenter = [Get](int i) -> Float3
  {
    auto geo = Get(i);
    auto [center, _] = geo.first->Get(geo.second);
    return center;
  };
  auto GetMaterial = [&scene, Get](int i) -> auto
  {
    auto geo = Get(i);
    return scene.GetMaterialOf({ geo.first, geo.second });
  };
  // Checking the coordinates for correct application of the transform statements.
  ASSERT_NEAR(GetCenter(0)[0], 1., 1.e-3);
  ASSERT_NEAR(GetCenter(1)[0], 5., 1.e-3);
  ASSERT_NEAR(GetCenter(2)[0], 8. + 11., 1.e-3); // Using the child scope transform.
  ASSERT_NEAR(GetCenter(3)[0], 14. + 5., 1.e-3); // Using the parent scope transform.
  ASSERT_TRUE(GetMaterial(3) == GetMaterial(1)); // Shaders don't persist beyond scopes.
  ASSERT_TRUE(!(GetMaterial(2) == GetMaterial(1))); // Shader within the scope was actually created and assigned.
  ASSERT_TRUE(!(GetMaterial(2) == GetMaterial(0))); // Shader within the scope is not the default.
}

} // namespace


TEST(Parser, Scopes)
{
  const char* scenestr = R"""(
s 1 2 3 0.5
transform 5 6 7
diffuse themat 1 1 1 0.9
s 0 0 0 0.5
{
transform 8 9 10
diffuse themat 1 1 1 0.3
s 11 12 13 0.5
}
s 14 15 16 0.5
)""";
  Scene scene;
  scene.ParseNFFString(scenestr);
  CheckSceneParsedWithScopes(scene);
}


TEST(Parser, ScopesAndIncludes)
{
  namespace fs = boost::filesystem;
  auto path1 = fs::temp_directory_path() / fs::unique_path("scene1-%%%%-%%%%-%%%%-%%%%.nff");
  std::cout << "scenepath1: " << path1.string() << std::endl;
  const char* scenestr1 = R"""(
transform 5 6 7
diffuse themat 1 1 1 0.9
s 0 0 0 0.5
)""";
  {
    std::ofstream os(path1.string());
    os.write(scenestr1, strlen(scenestr1));
  }
  const char* scenestr2 = R"""(
transform 8 9 10
diffuse themat 1 1 1 0.3
s 11 12 13 0.5
)""";
  auto path2 = fs::unique_path("scene2-%%%%-%%%%-%%%%-%%%%.nff"); // Relative filepath.
  auto path2_full = fs::temp_directory_path() / path2;
  std::cout << "scenepath2: " << path2.string() << std::endl;
  {
    std::ofstream os(path2_full.string());
    os.write(scenestr2, strlen(scenestr2));
  }
  const char* scenestr_fmt = R"""(
s 1 2 3 0.5
include {}
{{
include {}
}}
s 14 15 16 0.5
)""";
  auto path3 = fs::temp_directory_path() / fs::unique_path("scene3-%%%%-%%%%-%%%%-%%%%.nff");
  std::string scenestr = fmt::format(scenestr_fmt, path1.string(), path2.string());
  {
    std::ofstream os(path3.string());
    os.write(scenestr.c_str(), scenestr.size());
  }
  Scene scene;
  scene.ParseNFF(path3);
  CheckSceneParsedWithScopes(scene);
}





TEST(Parser, ImportDAE)
{
  const char* scenestr = R"""(
diffuse DefaultMaterial 1 1 1 0.5
m testing/scenes/unitcube.dae
)""";
  Scene scene;
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();
  constexpr double tol = 1.e-2;
  Box outside;
  outside.Extend({ -0.5 - tol, -0.5 - tol, -0.5 - tol });
  outside.Extend({ 0.5 + tol, 0.5 + tol, 0.5 + tol });
  Box inside;
  inside.Extend({ -0.5 + tol, -0.5 + tol, -0.5 + tol });
  inside.Extend({ 0.5 - tol, 0.5 - tol, 0.5 - tol });
  ASSERT_EQ(scene.GetNumGeometries(), 1);
  ASSERT_TRUE(scene.GetGeometry(0).type == Geometry::PRIMITIVES_TRIANGLES);
  ASSERT_TRUE(scene.GetBoundingBox().InBox(outside));
  ASSERT_FALSE(scene.GetBoundingBox().InBox(inside));
}


TEST(Parser, LightIndices)
{
  const char* scenestr = R"""(
{
larea arealight2 uniform 1 1 1 1
diffuse black  1 1 1 0.
m testing/scenes/unitcube.dae
}
{
larea arealight2 uniform 1 1 1 1
diffuse black  1 1 1 0.
s 0 0 0 1
}
{
m testing/scenes/unitcube.dae
}
{
s 0 0 0 1
}
)""";
  Scene scene;
  scene.ParseNFFString(scenestr);
  const auto n = scene.GetNumAreaLights();
  ASSERT_EQ(n, 13); // Cube has 12 triangles, plus 1 sphere.
  for (Scene::index_t i = 0; i < n; ++i)
  {
    PrimRef pr = scene.GetPrimitiveFromAreaLightIndex(i);
    auto reverse = scene.GetAreaLightIndex(pr);
    EXPECT_EQ(i, reverse);
  }
}


TEST(Parser, ImportCompleteSceneWithDaeBearingMaterials)
{
  const char* scenestr = R"""(
v
from 0 1.2 -1.3
at 0 0.6 0
up 0 1 0
resolution 128 128
angle 50

l 0 0.75 0  1 1 1 1

diffuse white  1 1 1 0.5
diffuse red    1 0 0 0.5
diffuse green  0 1 0 0.5
diffuse blue   0 0 1 0.5

m testing/scenes/cornelbox.dae
)""";
  Scene scene;
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();
  Box b = scene.GetBoundingBox();
  double size = Length(b.max - b.min);
  ASSERT_GE(size, 1.);
  ASSERT_LE(size, 3.);
}


TEST(Parser, YamlEmbedTransformLoad)
{
  const char* scenestr = R"""(
v
from 0 1.2 -1.3
at 0 0.6 0
up 0 1 0
resolution 128 128
angle 50

yaml{
transform:
  pos: [ 1, 2, 3]
  hpb: [ 0, 90, 0]
  angle_in_degree : true
}yaml
s 0 0 0 1.0
)""";
  Scene scene;
  scene.ParseNFFString(scenestr);
}
