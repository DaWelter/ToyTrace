#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

#ifdef HAVE_JSON
#include "json.hxx"
#endif

#include "ray.hxx"
#include "image.hxx"
#include "camera.hxx"
#include "infiniteplane.hxx"
#include "sampler.hxx"
#include "scene.hxx"
#include "renderingalgorithms_simplebase.hxx"
#include "atmosphere.hxx"
#include "util.hxx"
#include "hashgrid.hxx"
#include "photonintersector.hxx"
#include "shader_util.hxx"
#include "shader_physics.hxx"
#include "lightpicker_ucb.hxx"
#include "memory_arena.hxx"
#include "ndarray.hxx"

using namespace materials;

TEST(TestRaySegment, ExprTemplates)
{
  auto e = RaySegment{}.EndPoint();
  ASSERT_NE(typeid(e), typeid(e.eval()));
  ASSERT_EQ(typeid(e.eval()), typeid(Eigen::Matrix<double, 3, 1>));
}


TEST(TestRaySegment, EndPointNormal)
{
  Double3 p = RaySegment{{Double3(1., 0., 0.), Double3(0., 1., 0.)}, 2.}.EndPoint();
  ASSERT_EQ(p[0], 1.);
  ASSERT_EQ(p[1], 2.);
  ASSERT_EQ(p[2], 0.);
}


TEST(TestRaySegment, Reversed)
{
  RaySegment s{{{1, 2, 3}, {42, 0, 0}}, 5.};
  RaySegment r = s.Reversed();
  ASSERT_NEAR(s.EndPoint()[0], r.ray.org[0], 1.e-8);
  ASSERT_NEAR(r.EndPoint()[0], s.ray.org[0], 1.e-8);
}



TEST(TestMath, TakeFromVectorByIndices)
{
  Eigen::Array<double, 10, 1> m; m << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::Array<double, 3, 1> v = Take(m, Index3{3,4,9});
  ASSERT_EQ(v[0], 3);
  ASSERT_EQ(v[1], 4);
  ASSERT_EQ(v[2], 9);
}


TEST(TestMath, OrthogonalSystemZAligned)
{
  Double3 directions[] = {
    { 1., 0., 0. },
    { 0., 1., 0. },
    { 0., 0., 3. },
    { 1., 1., 0. }
  };
  for (auto dir : directions)
  {
    Eigen::Matrix<double, 3, 3> m = OrthogonalSystemZAligned(Normalized(dir));
    std::cout << "dir = " << dir << std::endl;
    std::cout << "m{dir} = " << m << std::endl;
    EXPECT_TRUE(m.isUnitary(1.e-6));
    ASSERT_NEAR(m.determinant(), 1., 1.e-6);
    auto mi = m.inverse().eval();
    auto dir_local = mi * dir;
    EXPECT_NEAR(dir_local[0], 0., 1.e-6);
    EXPECT_NEAR(dir_local[1], 0., 1.e-6);
    EXPECT_NEAR(dir_local[2], Length(dir), 1.e-6);
  }
}


TEST(TestMath, Quadratic)
{
  using util::Quadratic;
  float a = 1;
  float b = 100;
  float c = 1;
  float t0, t1, e0, e1;
  ASSERT_TRUE(Quadratic<float>(a, b, c, 0, 0, 0, t0, t1, e0, e1));
  std::cout << t0 << " " << t1 << " " << e0 << " " << e1 << std::endl;
  EXPECT_NEAR(t0, -99.99f, 1.e-4);
  EXPECT_NEAR(t1, -0.010001f, 1.e-7);
  EXPECT_NEAR(e0, 3.57604003511369e-5f, 1.e-10);
  EXPECT_NEAR(e1, 3.57675622453257e-9f, 1.e-15);
  ASSERT_TRUE(Quadratic<float>(a, b, c, 0.1, 0, 0, t0, t1, e0, e1));
  std::cout << e0 << " " << e1 << std::endl;
  EXPECT_NEAR(e0, 10.0000362396240, 1.e-4);
  EXPECT_NEAR(e1, 1.03616784485894e-7, 1.e-10);
  ASSERT_TRUE(Quadratic<float>(a, b, c, 0, 1, 0, t0, t1, e0, e1));
  std::cout << e0 << " " << e1 << std::endl;
  EXPECT_NEAR(e0, 1.00013589859009, 1.e-7);
  EXPECT_NEAR(e1, 0.000100033619673923, 1.e-10);
  ASSERT_TRUE(Quadratic<float>(a, b, c, 0, 0, 0.1, t0, t1, e0, e1));
  std::cout << e0 << " " << e1 << std::endl;
  EXPECT_NEAR(e0, 0.00103596039116383, 1.e-9);
  EXPECT_NEAR(e1, 0.00100020365789533, 1.e-9);
}





TEST(TestMath, Reflected)
{
  Double3 n{0., 1., 0.};
  auto in = Normalized(Double3{0., 1., 2.});
  Double3 out = Reflected(in, n);
  auto out_expected = Normalized(Double3{0., 1., -2.});
  ASSERT_GE(Dot(out, out_expected), 0.99);
  // Test normal flip
  n = Double3{0.,-1.,0.};
  out = Reflected(in, n);
  ASSERT_GE(Dot(out, out_expected), 0.99);
}


TEST(TestMath, Refracted)
{
  Double3 n{0., 1., 0.};
  auto w1 = Normalized(Double3{0., 1., 2.});
  double eta1 = 1.;
  double eta2 = 1.1;
  auto w2 = Refracted(w1, n, eta1/eta2);
  ASSERT_TRUE((bool)w2);
  auto w3 = Refracted(*w2, n, eta2/eta1);
  ASSERT_TRUE((bool)w3);
  EXPECT_NEAR((*w3)[0], w1[0], 1.e-3);
  EXPECT_NEAR((*w3)[1], w1[1], 1.e-3);
  EXPECT_NEAR((*w3)[2], w1[2], 1.e-3);
  // Test normal flip
  n = Double3{0.,-1.,0.};
  Double3 w4 = *Refracted(w1, n, eta1/eta2);
  EXPECT_NEAR(w4[0], (*w2)[0], 1.e-3);
  EXPECT_NEAR(w4[1], (*w2)[1], 1.e-3);
  EXPECT_NEAR(w4[2], (*w2)[2], 1.e-3);  
  // Test total reflection.
  w1 = Normalized(Double3{0., 1., 100.});
  eta1 = 1.2;
  eta2 = 1.;
  w2 = Refracted(w1, n, eta1/eta2);
  ASSERT_FALSE((bool)w2);
}


TEST(TestMath, RefractedHalfVector)
{
  double eta_1_over_2 = 1.0/1.5;
  Double3 w1 = Normalized(Double3{0., 3., 1.});
  Double3 w2  = Normalized(Double3{0., -1., 0.});
  boost::optional<Double3> h = HalfVectorRefracted(w1, w2, eta_1_over_2);
  boost::optional<Double3> wr = Refracted(w1, *h, eta_1_over_2);
  ASSERT_TRUE((bool)wr);
  ASSERT_NEAR((*wr)[0], w2[0], 1.e-3);
  ASSERT_NEAR((*wr)[0], w2[0], 1.e-3);
  ASSERT_NEAR((*wr)[0], w2[0], 1.e-3);
}


TEST(TestMath, RefractedHalfVectorDegenerate)
{
  double eta_1_over_2 = 1.0/1.5;
  Double3 w1{0., 1., 0.};
  Double3 w2{0., -1., 0.};
  boost::optional<Double3> h = HalfVectorRefracted(w1, w2, eta_1_over_2);
  boost::optional<Double3> wr = Refracted(w1, *h, eta_1_over_2);
  ASSERT_TRUE((bool)wr);
  ASSERT_NEAR((*wr)[0], w2[0], 1.e-3);
  ASSERT_NEAR((*wr)[0], w2[0], 1.e-3);
  ASSERT_NEAR((*wr)[0], w2[0], 1.e-3);
}


TEST(TestMath, HalfVector)
{
  Double3 w1 = Normalized(Double3{0., 3., 1.});
  Double3 w2  = Normalized(Double3{0., -1., 0.});
  Double3 h = HalfVector(w1, w2);
  Double3 wr = Reflected(w1, h);
  ASSERT_NEAR(wr[0], w2[0], 1.e-3);
  ASSERT_NEAR(wr[0], w2[0], 1.e-3);
  ASSERT_NEAR(wr[0], w2[0], 1.e-3);
}


TEST(TestMath, HalfVectorDegenerate)
{
  Double3 w1{0., 1., 0.};
  Double3 w2{0., -1., 0.};
  Double3 h = HalfVector(w1, w2);
  Double3 wr = Reflected(w1, h);
  ASSERT_NEAR(wr[0], w2[0], 1.e-3);
  ASSERT_NEAR(wr[0], w2[0], 1.e-3);
  ASSERT_NEAR(wr[0], w2[0], 1.e-3);
}



TEST(TestMath, RaySphereClip)
{
  { // Start ray outside of the sphere
    Double3 org{1,0,-1};
    Double3 dir{0,0,1};
    Double3 p{1,0,2};
    double  r{2};
    auto [ok, tnear, tfar] = ClipRayToSphereInterior(org, dir, 0, LargeNumber, p, r);
    EXPECT_TRUE(ok);
    EXPECT_NEAR(tnear, 1., 1.e-6);
    EXPECT_NEAR(tfar, 5., 1.e-6);
  }
  { // Start ray inside
    Double3 org{1,0,-1};
    Double3 dir{0,0,1};
    Double3 p{1,0,1};
    double  r{3};
    auto [ok, tnear, tfar] = ClipRayToSphereInterior(org, dir, 0, LargeNumber, p, r);
    EXPECT_TRUE(ok);
    EXPECT_NEAR(tnear, 0., 1.e-6);
    EXPECT_NEAR(tfar, 5., 1.e-6);
  }
}


TEST(TestMath, Pow)
{
  EXPECT_EQ(util::Pow(5.,0), 1.);
  EXPECT_EQ(util::Pow(5.,1), 5.);
  EXPECT_NEAR(util::Pow(5.,27), std::pow(5.,27.), 1.e-9);
  EXPECT_NEAR(util::Pow(0.25,12), std::pow(0.25,12), 1.e-9);
}



TEST(Spectral, RGBConversion)
{
  using namespace Color;
  std::array<RGB, 10> trials = {{
    { 0._rgb, 0._rgb, 0._rgb },
    { 1._rgb, 0._rgb, 0._rgb },
    { 0._rgb, 1._rgb, 0._rgb },
    { 0._rgb, 0._rgb, 1._rgb },
    { 1._rgb, 1._rgb, 0._rgb },
    { 1._rgb, 0._rgb, 1._rgb },
    { 0._rgb, 1._rgb, 1._rgb },
    { 1._rgb, 1._rgb, 1._rgb },
    { 1._rgb, 0.5_rgb, 0.3_rgb },
    { 0.5_rgb, 0.8_rgb, 0.9_rgb },
  }};
  for (RGB trial : trials)
  {
    SpectralN s = RGBToSpectrum(trial);
    for (int i=0; i<s.size(); ++i)
    {
      EXPECT_GE(s[i], 0.);
      EXPECT_LE(s[i], 1.01);
    }
    RGB back = SpectrumToRGB(s);
    EXPECT_NEAR(value(trial[0]), value(back[0]), 1.3e-2);
    EXPECT_NEAR(value(trial[1]), value(back[1]), 1.3e-2);
    EXPECT_NEAR(value(trial[2]), value(back[2]), 1.3e-2);
  }
}


TEST(Spectral, RGBConversionLinearity)
{
  using namespace Color;
  SpectralN spectra[3] = {
    SpectralN::Zero(), SpectralN::Zero(), SpectralN::Zero()};
  spectra[0][0] = 1.;
  spectra[1][2] = 1.;
  spectra[2][NBINS-1] = 1.;
  auto CheckLinear = [](const SpectralN &a, const SpectralN &b)
  {
    RGB rgb_a = SpectrumToRGB(a);
    RGB rgb_b = SpectrumToRGB(b);
    RGB rgb_ab = SpectrumToRGB(a+b);
    EXPECT_LE(value((rgb_ab - rgb_b - rgb_a).abs().maxCoeff()), 1.e-3);
  };
  CheckLinear(spectra[0], spectra[1]);
  CheckLinear(spectra[0], spectra[2]);
  CheckLinear(spectra[1], spectra[2]);
}


TEST(Spectral, RGBConversionSelection)
{
  /* Back and forth transform using RGBToSpectralSelection and SpectralSelectionToRGB.
   * Only part of the spectrum is considered by these function, namely the wavelengths
   * given by the indices. Because of that, the complete reconstruction of the RGB 
   * original requires summation over parts of the spectrum. It is basically an inner
   * product of the spectrum with the color matching function, except that on each iteration 
   * is not one wavelengths but multiple added to the output. */
  
  using namespace Color;
  auto rgb = RGB(0.8_rgb, 0.2_rgb, 0.6_rgb);
  RGB converted_rgb = RGB::Zero();
  for (int lambda=0; lambda<Color::NBINS-2; lambda += static_size<Spectral3>())
  {
    Index3 idx{lambda, lambda+1, lambda+2};
    converted_rgb += SpectralSelectionToRGB(RGBToSpectralSelection(rgb, idx), idx);
  }
  EXPECT_NEAR(value(rgb[0]), value(converted_rgb[0]), 1.e-2);
  EXPECT_NEAR(value(rgb[1]), value(converted_rgb[1]), 1.e-2);
  EXPECT_NEAR(value(rgb[2]), value(converted_rgb[2]), 1.e-2);
}



class IntersectorTests : public testing::Test
{
  EmbreeAccelerator scene;
protected:
  SurfaceInteraction intersection;
  double distance;
  
  void Initialize(std::function<void(EmbreeAccelerator &)> scene_filler)
  {
    scene_filler(scene);
    scene.Build();
  }
  
  void Intersect(const Ray &ray, bool expect_hit = true)
  {
    RaySegment rs{ray, LargeNumber};
    bool bhit = scene.FirstIntersection(rs.ray, 0., rs.length, intersection);
    ASSERT_TRUE(bhit == expect_hit);
    distance = rs.length;
  }
  
  void CheckPosition(const Double3 &p) const
  {
    for (int i=0; i<3; ++i)
      EXPECT_NEAR(intersection.pos[i], p[i], 1.e-6);
  }
  
  void CheckNormal(const Double3 &n)
  {
    for (int i=0; i<3; ++i)
      EXPECT_NEAR(intersection.normal[i], n[i], 1.e-6);
  }
};


inline RaySegment MakeSegmentAt(const SurfaceInteraction &intersection, const Double3 &ray_dir)
{
  return RaySegment{
    {intersection.pos + AntiSelfIntersectionOffset(intersection, ray_dir), ray_dir},
    LargeNumber
  };
}


TEST_F(IntersectorTests, Sphere)
{
  Spheres s; 
  this->Initialize([&](EmbreeAccelerator &scene) {
    s.Append({ 0., 0., 2. }, 2.);
    scene.InsertRefTo(s);
  });
  Intersect({{0., 0., -1.},{0., 0., 1.}});
  EXPECT_NEAR(distance, 1., 1.e-6);
  CheckPosition({0., 0., 0.});
  CheckNormal({0.,0.,-1.});
}


TEST_F(IntersectorTests, Triangle)
{
  /*     x
   *   / |
   *  /  |
   * x---x
   * Depiction of the triangle */
  Mesh m{ 0,0 };
  this->Initialize([&](EmbreeAccelerator &scene) {
    float q = 0.5;
    AppendSingleTriangle(m, {-q, -q, 0}, {q, -q, 0}, {q, q, 0}, {0,0,0});
    m.MakeFlatNormals();
    scene.InsertRefTo(m);
  });
  Intersect(Ray{{0.1, 0., -1.},{0., 0., 1.}});
  EXPECT_NEAR(distance, 1., 1.e-6);
  CheckPosition({0.1, 0., 0.});
  CheckNormal({0., 0., -1.});
}


TEST_F(IntersectorTests, TriangleEdgeCase)
{
  Mesh m{ 0,0 };
  this->Initialize([&](EmbreeAccelerator &scene) {
    float q = 0.5;
    AppendSingleTriangle(m, {-q, -q, 0}, {q, -q, 0}, {q, q, 0}, {0,0,0});
    AppendSingleTriangle(m, {-q, -q, 0}, {-q, q, 0}, {q, q, 0}, {0,0,0});
    m.MakeFlatNormals();
    scene.InsertRefTo(m);
  });
  Intersect(Ray{{0., 0., -1.},{0., 0., 1.}});
  EXPECT_NEAR(distance, 1., 1.e-6);
  CheckPosition({0., 0., 0.});
  CheckNormal({0., 0., -1.});
  Intersect(Ray{{1.e-7, 0., -1.},{0., 0., 1.}}); // 10^{-7} still works. It happens that log2(10^7) ~= 23, the number of significant bits in IEEE float.
  EXPECT_EQ(intersection.hitid.index, int{0});
  Intersect(Ray{{-1.e-7, 0., -1.},{0., 0., 1.}});
  EXPECT_EQ(intersection.hitid.index, int{1});
}




TEST_F(IntersectorTests, SelfIntersection)
{
  // Intersect with primitive far off the origin.
  // Go to intersection point and shoot another ray.
  // Is there self intersection?
  Mesh m{ 0,0 };
  this->Initialize([&](EmbreeAccelerator &scene) {
    float q = 0.5;
    const float dz = 10;
    const float z = 1.e5;
    AppendSingleTriangle(m, { -q, -q, z+dz }, { q, -q, z-dz }, { q, q, z-dz }, { 0,0,0 });
    AppendSingleTriangle(m, { -q, -q, z+dz }, { -q, q, z+dz }, { q, q, z-dz }, { 0,0,0 });
    m.MakeFlatNormals();
    scene.InsertRefTo(m);
  });

  Ray r{{0, 0, 1.e5 + 1.}, {0, 0, -1}};
  Intersect(r, true);
  r.org = intersection.pos;
  Intersect(r, false);
}


TEST(Embree, SphereIntersectionMadness)
{
  // Intersect sphere with ray. Start new ray from intersection point.
  // Is there self-intersection?
  Sampler sampler;
  Double3 sphere_org{0., 0., 2};
  double sphere_rad = 6300;
  Spheres geom;
  geom.Append(sphere_org.cast<float>(), (float)sphere_rad);
  EmbreeAccelerator world;
  world.InsertRefTo(geom);
  world.Build();
  
  const int N = 100;
  for (int num = 0; num < N; ++num)
  {
    // Sample random position outside the sphere as start point.
    // And position inside the sphere as end point.
    // Intersection is thus guaranteed.
    Double3 org = 
      SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare())
      * sphere_rad * 10. + sphere_org;
    Double3 target = 
      SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare())
      * sphere_rad * 0.99 + sphere_org;
    Double3 dir = target - org; 
    Normalize(dir);
    
    // Shoot ray to the inside. Expect intersection at the
    // front side of the sphere.
    RaySegment rs{{org, dir}, LargeNumber};
    SurfaceInteraction intersect1;
    bool bhit = world.FirstIntersection(rs.ray, 0, rs.length, intersect1);
    ASSERT_TRUE(bhit);
    ASSERT_LE(Length(org - intersect1.pos), Length(org - sphere_org));
    
    // Put the origin at the intersection and shoot a ray to the outside
    // in a random direction. Expect no further hit.
    auto m  = OrthogonalSystemZAligned(intersect1.geometry_normal);
    auto new_dir = m * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    rs = MakeSegmentAt(intersect1, new_dir);
    bhit = world.FirstIntersection(rs.ray, 0, rs.length, intersect1);
    ASSERT_FALSE(bhit);
    ASSERT_EQ(rs.length, LargeNumber);
  }
}


TEST(Embree, SphereIntersectionMadness2)
{
  // Bounce a particle around in the inside of the sphere.
  // It must not escape.
  Sampler sampler;
  Double3 sphere_org{0., 0., 2};
  double sphere_rad = 6300;
  Spheres geom;
  geom.Append(sphere_org.cast<float>(), (float)sphere_rad);
  EmbreeAccelerator world;
  world.InsertRefTo(geom);
  world.Build();
  
  RaySegment rs{{sphere_org, {1., 0., 0.}}, LargeNumber};
  HitId hit, last_hit;
  const int N = 100;
  for (int num = 0; num < N; ++num)
  {
    SurfaceInteraction intersect;
    bool bhit = world.FirstIntersection(rs.ray, 0, rs.length, intersect);
    ASSERT_TRUE(bhit);
    double rho = Length(intersect.pos-sphere_org);
    auto m  = OrthogonalSystemZAligned(intersect.normal);
    auto new_dir = m * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    rs = MakeSegmentAt(intersect, new_dir);
    rho = Length(rs.ray.org-sphere_org);
    EXPECT_LE(rho, sphere_rad);
    last_hit = hit;
  }
}


TEST(Embree, SphereIntersectionMadness3)
{
  // Bounce a particle around within a shell between two spheres.
  // It must not escape.
  Sampler sampler;
  Spheres geom;
  double rad1 = 6300.;
  double rad2 = 6350.;
  geom.Append({0.f, 0.f, 0.f}, (float)rad1);
  geom.Append({0.f, 0.f, 0.f}, (float)rad2);
  EmbreeAccelerator world;
  world.InsertRefTo(geom);
  world.Build();
  
  Double3 start_point = {0.5*(rad1+rad2), 0., 0.};
  double max_dist_from_start = 0.;
  RaySegment rs{{start_point, {1., 0., 0.}}, LargeNumber};
  const int N = 100;
  for (int num = 0; num < N; ++num)
  {
    SurfaceInteraction intersect;
    bool bhit = world.FirstIntersection(rs.ray, 0., rs.length, intersect);
    ASSERT_TRUE(bhit);
    double rho = Length(intersect.pos);
    max_dist_from_start = std::max(max_dist_from_start, Length(intersect.pos-start_point));
    auto eps = rad2*std::numeric_limits<float>::epsilon();
    EXPECT_LE(rho-eps, rad2);
    EXPECT_GE(rho+eps, rad1);
    auto m  = OrthogonalSystemZAligned(intersect.normal);
    auto new_dir = m * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    rs = MakeSegmentAt(intersect, new_dir);
  }
  EXPECT_GE(max_dist_from_start, rad1*1.e-3);
}


TEST(Embree, FindPlaceIfUnique)
{
  ToyVector<int> items{ 1, 3, 5, 7, 9 };
  auto run = [&](int i) {
    return EmbreeAcceleratorDetail::FindPlaceIfUnique(
      items.begin(),
      items.end(),
      i, std::less<int>(), std::equal_to<int>());
  };
  {
    auto[it, found] = run(10);
    EXPECT_FALSE(found);
    EXPECT_EQ(it, items.end());
  }
  {
    auto[it, found] = run(8);
    EXPECT_FALSE(found);
    EXPECT_EQ(it, items.end() - 1);
  }
  {
    auto[it, found] = run(9);
    EXPECT_TRUE(found);
    EXPECT_EQ(it, items.end() - 1);
  }
  {
    auto[it, found] = run(0);
    EXPECT_FALSE(found);
    EXPECT_EQ(it, items.begin());
  }
  {
    auto[it, found] = run(1);
    EXPECT_TRUE(found);
    EXPECT_EQ(it, items.begin());
  }
  {
    auto[it, found] = run(5);
    EXPECT_TRUE(found);
    EXPECT_EQ(it, items.begin() + 2);
  }
  {
    auto[it, found] = run(4);
    EXPECT_FALSE(found);
    EXPECT_EQ(it, items.begin() + 2);
  }
}


TEST(Embree, AllIntersections)
{
  Spheres geom;
  geom.Append({ 0.f, 0.f, 0.f }, 1.f);
  geom.Append({ 0.f, 0.f, 0.f }, 2.f);
  geom.Append({ 0.f, 0.f, 0.f }, 2.f); // Intentional duplicate
  EmbreeAccelerator world;
  world.InsertRefTo(geom);
  world.Build(true);
  Ray ray{ {0, 0, -3}, {0, 0, 1} };
  //ToyVector<double> distances;
  auto intersections = world.IntersectionsInOrder(ray, 0, 3);
  for (auto is : intersections)
  {
    fmt::print("d={}, geo={}, prim={}", is.t, is.geom, is.prim);
  }
}


//TEST(Scene, VolumeIntersection)
//{
//  const char* scenestr = R"""(
//medium medium 1 1
//shader none
//m testing/scenes/unitcube.dae
//)""";
//  Scene scene;
//  scene.ParseNFFString(scenestr);
//  scene.BuildAccelStructure();
//  const Ray r{ {0, 0, -2.}, {0, 0, 1} };
//  auto is = scene.FirstIntersectionWithVolumes(r, 0, LargeNumber);
//  ASSERT_TRUE(is);
//  EXPECT_NEAR(is->t, 1.5, 1.e-6);
//  EXPECT_TRUE(is->entering);
//  is = scene.FirstIntersectionWithVolumes(r, is->t, LargeNumber);
//  ASSERT_TRUE(is);
//  EXPECT_NEAR(is->t, 2.5, 1.e-6);
//  EXPECT_FALSE(is->entering);
//  is = scene.FirstIntersectionWithVolumes(r, is->t, LargeNumber);
//  ASSERT_FALSE(is);
//}


TEST(Display, Open)
{
  ImageDisplay display;
  Image img(128, 128);
  img.SetColor(255, 0, 0);
  img.DrawLine(0, 0, 128, 128, 5);
  img.DrawLine(0, 128, 128, 0, 5);
  display.show(img);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  img.SetColor(255,255,255);
  img.DrawLine(64, 0, 64, 128);
  img.DrawLine(0, 64, 128, 64);
  display.show(img);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}




class PerspectiveCameraTesting : public testing::Test
{
  Double3 pos{0, 0, 0},
          dir{0, 0, 1},
          up{0, 1, 0};
  double fov = 90;
  Sampler sampler;
  InfinitePlane imageplane;
  std::unique_ptr<PerspectiveCamera> cam;
  Scene scene;
public:
  PerspectiveCameraTesting() :
    imageplane({0, 0, 1}, {0, 0, 1})
  {
  }
  
  void init(int _xres, int _yres)
  {
    cam = std::make_unique<PerspectiveCamera>(pos, dir, up, fov, _xres, _yres);
  }
  
  void run(int _pixel_x, int _pixel_y)
  {
  /* Pixel bins:
    |         i=0         |           i=1           |        .....           |         x=xres-1       |
    At distance z=1:
    x=0                   x=i/xres                 x=2/xres           ....   x=(xres-1)/xres          x=1
  */
    Double3 possum{0,0,0}, possqr{0,0,0};
    Box box;
    constexpr int Nsamples = 100;
    for (int i = 0; i < Nsamples; ++i)
    {
      auto s = cam->TakeDirectionSampleFrom(
        cam->PixelToUnit({_pixel_x, _pixel_y}), pos, sampler,
        PathContext{SelectRgbPrimaryWavelengths()});
      Double3 barry;
      double length = 100.;
      bool is_hit = imageplane.Intersect({pos, s.coordinates}, 0., length, barry);
      ASSERT_TRUE(is_hit);
      Double3 endpos = pos + length * s.coordinates;
      possum += endpos;
      possqr += Product(endpos, endpos);
      box.Extend(endpos);
    }
    // int_x=-s-s x^2 / (2s) dx = 1/3 x^3 /2s | x=-s..s = 1/(6s) 2*(s^3) = 1/3 s^2. -> std = sqrt(1/3) s
    // s = 1 -> std = 0.577
    Double3 average = possum / Nsamples;
    Double3 stddev  = (possqr / Nsamples - Product(average, average)).array().sqrt().matrix();
    std::cout << "@Pixel: ix=" << _pixel_x << "," << _pixel_y << std::endl;
    std::cout << "Pixel: " << average << " +/- " << stddev << std::endl;
    Double3 exactpos {((_pixel_x + 0.5) * 2.0 / cam->xres - 1.0),
                      ((_pixel_y + 0.5) * 2.0 / cam->yres - 1.0),
                      1.0};
    std::cout << "Box: " << box.min << " to " << box.max << std::endl;
    Double3 radius { 2.0/cam->xres, 2.0/cam->yres, 0. };
    std::cout << "Exact: " << exactpos << " +/- " << radius << std::endl;
  }
};


TEST_F(PerspectiveCameraTesting, Sampling1)
{
  init(11,11);
  run(0,0);
  run(10,10);
  run(5,5);
}


class TextureLoadTest : public testing::Test
{
protected:
  void CompareTexture(const Texture &tex, int w, int h, const ToyVector<RGB> &expected)
  {
    ASSERT_EQ(tex.Width() , w);
    ASSERT_EQ(tex.Height() , h);
    for (int y=0; y<tex.Height(); ++y) 
    {
    for (int x=0; x<tex.Width(); ++x)
    {
      RGB tex_col = tex.GetPixel(x, y);
      RGB expected_col = expected[y*tex.Width()+x];
      EXPECT_TRUE(((tex_col-expected_col).abs() < 1.e-3_rgb).all());
    }
    std::cerr << "\n";
    }
  }
};


TEST_F(TextureLoadTest, UCharSingleChn)
{
  ToyVector<RGB> expected {
    { 1._rgb, 1._rgb, 1._rgb }, { 0._rgb, 0._rgb, 0._rgb },
    { 0._rgb, 0._rgb, 0._rgb }, { 1._rgb, 1._rgb, 1._rgb },
    { 0._rgb, 0._rgb, 0._rgb }, { 0._rgb, 0._rgb, 0._rgb },
    { 0._rgb, 0._rgb, 0._rgb }, { 0._rgb, 0._rgb, 0._rgb },
    { 0._rgb, 0._rgb, 0._rgb }, { 0._rgb, 0._rgb, 0._rgb }
  };
  CompareTexture(Texture("testing/scenes/texloadtest1.png"), 2, 5, expected);
}


TEST_F(TextureLoadTest, Float3Chn)
{
  ToyVector<RGB> expected {
    { 10._rgb, 9._rgb, 8._rgb }  , { 0._rgb, 0._rgb, 0._rgb },
    { 0._rgb, 0._rgb, 0._rgb }   , { 1._rgb, 1._rgb, 1._rgb },
    { 0._rgb, 0._rgb, 0._rgb }   , { 0._rgb, 0._rgb, 0._rgb },
    { 0._rgb, 0._rgb, 0._rgb }   , { 0._rgb, 0._rgb, 0._rgb },
    { 0._rgb, 0._rgb, 0._rgb }   , { 0._rgb, 0._rgb, 0._rgb }
  };
  CompareTexture(Texture("testing/scenes/texloadtest2.exr"), 2, 5, expected);
}

namespace materials { namespace Atmosphere
{

TEST(SimpleAtmosphereTest, Altitude)
{
  Atmosphere::SphereGeometry geom{{0.,0.,1.}, 100.};
  Double3 pos{0., 110., 1.};
  double altitude = geom.ComputeAltitude(pos);
  EXPECT_NEAR(altitude, 10., 1.e-6);
}


TEST(SimpleAtmosphereTest, CollisionCoefficients)
{
  Atmosphere::ExponentialConstituentDistribution constituents{};
  double altitude = 10.;
  Spectral3 sigma_s, sigma_a;
  Index3 lambda_idx = Color::LambdaIdxClosestToRGBPrimaries();
  constituents.ComputeCollisionCoefficients(altitude, sigma_s, sigma_a, lambda_idx);
  double expected_sigma_s =
      std::exp(-altitude/8.)*0.0076 +
      std::exp(-altitude/1.2)*20.e-3;
  double expected_sigma_a =
      std::exp(-altitude/8.)*0.e-3 +
      std::exp(-altitude/1.2)*2.22e-3;
  EXPECT_NEAR(sigma_s[0], expected_sigma_s, expected_sigma_s*1.e-2);
  EXPECT_NEAR(sigma_a[0], expected_sigma_a, expected_sigma_a*1.e-2);
}


TEST(SimpleAtmosphereTest, LowestPoint)
{
  Atmosphere::SphereGeometry geom{{0.,0.,1.}, 100.};
  { // Looking down through the atmospheric layer.
    RaySegment seg{{{-1000., 0., 101.}, {1., 0., 0.}}, 2000.};
    Double3 p = geom.ComputeLowestPointAlong(seg);
    EXPECT_NEAR(p[0], 0., 1.e-8);
    EXPECT_NEAR(p[1], 0., 1.e-8);
    EXPECT_NEAR(p[2], 101., 1.e-8);
  }

  { // Same as above but limit the distance.
    RaySegment seg{{{-1000., 0., 101.}, {1., 0., 0.}}, 20.};
    Double3 p = geom.ComputeLowestPointAlong(seg);
    EXPECT_NEAR(p[0], -1000+20., 1.e-8);
    EXPECT_NEAR(p[1], 0., 1.e-8);
    EXPECT_NEAR(p[2], 101., 1.e-8);
  }

  { // Look up
    RaySegment seg{{{0., 0., 101.}, Normalized(Double3{1., 0., 10.})}, 5.};
    Double3 p = geom.ComputeLowestPointAlong(seg);
    EXPECT_NEAR(p[0], 0., 1.e-8);
    EXPECT_NEAR(p[1], 0., 1.e-8);
    EXPECT_NEAR(p[2], 101., 1.e-8);
  }
}


#ifdef HAVE_JSON
TEST(AtmosphereTest, LoadTabulatedData)
{
  const std::string filename = "testing/scenes/earth_atmosphere_collision_coefficients.json";
  std::ifstream is(filename.c_str(), std::ios::binary | std::ios::ate);
  std::string data;
  { // Following http://en.cppreference.com/w/cpp/io/basic_istream/read
    auto size = is.tellg();
    data.resize(size);
    is.seekg(0);
    is.read(&data[0], size);
  }
  rapidjson::Document d;
  d.Parse(data.c_str());
  ASSERT_TRUE(d.IsObject());
  const rapidjson::Value& val = d["H0"];
  ASSERT_TRUE(std::isfinite(val.GetDouble()));
  const auto& sigma_t = d["sigma_t"]; // Note: My version of rapidjson is age old. In up to date versions this should read blabla.GetArray().
  ASSERT_GT(sigma_t.Size(), 0ul);
  const auto& sigma_t_at_altitude0 = sigma_t[0];
  ASSERT_EQ(sigma_t_at_altitude0.Size(), Color::NBINS);
  const auto& value = sigma_t_at_altitude0[0];
  ASSERT_TRUE(std::isfinite(value.GetDouble()));
}
#endif

} } // materials::atmosphere


TEST(Misc, Sample)
{
  struct Testing {};
  Sample<double, double, Testing> smpl_pdf{1., 1., 42.};
  Sample<double, double, Testing> smpl_pmf{1., 1., 42.};
  Sample<double, double, Testing> zero_pmf{1., 1., 0.};
  Sample<double, double, Testing> zero_pdf{1., 1., 0.};
  SetPmfFlag(zero_pmf);
  SetPmfFlag(smpl_pmf);
  ASSERT_TRUE(IsFromPdf(smpl_pdf));
  ASSERT_TRUE(IsFromPdf(zero_pdf));
  ASSERT_FALSE(IsFromPdf(smpl_pmf));
  ASSERT_FALSE(IsFromPdf(zero_pmf));
  ASSERT_EQ(PdfValue(smpl_pdf),42.);
  ASSERT_EQ(PmfValue(smpl_pmf),42.);
}


//////////////////////////////////////////////
/// Bisection test
//////////////////////////////////////////////


TEST(Bisection, Search)
{
  std::vector<float> vals{
    1., 3., 9., 10.
  };
  EXPECT_EQ(BisectionSearch<float>(AsSpan(vals), 0.5), 0);
  EXPECT_EQ(BisectionSearch<float>(AsSpan(vals), 1.5), 1);
  EXPECT_EQ(BisectionSearch<float>(AsSpan(vals), 9.5), 3);
  EXPECT_EQ(BisectionSearch<float>(AsSpan(vals), 11), 4);
}


TEST(TowerSampling, ComputeCumSum)
{
  std::vector<float> vals{
    1,3,3,2,1
  };
  TowerSamplingComputeNormalizedCumSum(AsSpan(vals));
  EXPECT_NEAR(vals[0], 1./10., 1.e-4);
  EXPECT_NEAR(vals[1], 4./10., 1.e-3);
  EXPECT_NEAR(vals[2], 7./10., 1.e-3);
  EXPECT_NEAR(vals[3], 9./10., 1.e-3);
  EXPECT_NEAR(vals[4], 1., 1.e-3);
}


//////////////////////////////////////////////
/// UV projection
//////////////////////////////////////////////

TEST(Projections, UvToSpherical)
{
  Float2 uv{0.1f, 0.2f};
  Float2 angles = Projections::UvToSpherical(uv);
  Float2 uv_back = Projections::SphericalToUv(angles);
  EXPECT_NEAR(uv[0], uv_back[0], 1.e-4);
  EXPECT_NEAR(uv[1], uv_back[1], 1.e-4);
}

TEST(Projections, KartesianToSpherical2)
{
  const float phi = Pi*0.4;
  const float theta = Pi*0.2;
  Float2 angles{phi, theta};
  Float3 dir = Projections::SphericalToUnitKartesian(angles);
  Float2 angles_back = Projections::KartesianToSpherical(dir);
  EXPECT_NEAR(phi, angles_back[0], 1.e-4);
  EXPECT_NEAR(theta, angles_back[1], 1.e-4);
}

TEST(Projections, KartesianToSpherical1)
{
  Float3 dir;
  dir = Projections::SphericalToUnitKartesian(Float2{0.,0.});
  EXPECT_NEAR(dir[0], 0., 1.e-4);
  EXPECT_NEAR(dir[1], 0., 1.e-4);
  EXPECT_NEAR(dir[2], 1., 1.e-4);
  dir = Projections::SphericalToUnitKartesian(Float2{0.,Pi*0.5});
  EXPECT_NEAR(dir[0], 1., 1.e-4);
  EXPECT_NEAR(dir[1], 0., 1.e-4);
  EXPECT_NEAR(dir[2], 0., 1.e-4);
}

//////////////////////////////////////////////
/// Grid Indices
//////////////////////////////////////////////

TEST(GridIndices, RowMajor2d)
{
  int x = 3, y = 4, w = 5, h = 9;
  int offset = util::RowMajorOffset(x, y, w, h);
  std::tie(x, y) = util::RowMajorPixel(offset, w, h);
  EXPECT_EQ(x, 3);
  EXPECT_EQ(y, 4);
}

//////////////////////////////////////////////
/// Utilities
//////////////////////////////////////////////

TEST(Utils,StartsEndsWith)
{
    EXPECT_TRUE(util::startswith("foobar","foo"));
    EXPECT_FALSE(util::startswith("foobar","baz"));
    EXPECT_TRUE(util::endswith("foobar","bar"));
    EXPECT_FALSE(util::endswith("foobar","foo"));
}

TEST(Utils,TransformArray)
{
  std::array<int,2> xs = { 1, 5 };
  std::array<std::tuple<int>,2> y = util::TransformArray(xs, [](size_t x) { return std::make_tuple(int(x)*2); });
  EXPECT_EQ(std::get<0>(y[0]), 2);
  EXPECT_EQ(std::get<0>(y[1]), 2*5);
}

TEST(Utils,GenerateArray)
{
  std::array<std::tuple<int>,2> y = util::GenerateArray<2>([](size_t i) { return std::make_tuple(42+int(i)*2); });
  EXPECT_EQ(std::get<0>(y[0]), 42);
  EXPECT_EQ(std::get<0>(y[1]), 44);
}




///////////////////////////////////////////////
/// HashGrid
///////////////////////////////////////////////

TEST(HashGrid,HashGrid)
{
  // Generate points uniformly distributed on a sphere. Put them into the hash grid.
  // Generate query points, check if points in the hashgrid are correctly reported
  Sampler sampler;
  ToyVector<Double3> points;
  for (int i=0; i<10; ++i)
  {
    points.emplace_back(SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare()));
  }
  HashGrid hashgrid(0.1, points);
  ToyVector<bool> reported_in_range(points.size(), false);
  for (int j=0; j<10; ++j)
  {
    Double3 p_query = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
    std::fill(reported_in_range.begin(), reported_in_range.end(), false);
    hashgrid.Query(p_query,[&](int i){
      reported_in_range[i] = (p_query - points[i]).norm() <= hashgrid.Radius();
    });
    for (int i=0; i<points.size(); ++i)
    {
      if (!reported_in_range[i])
      {
        auto distance = (p_query - points[i]).norm();
        EXPECT_GE(distance,hashgrid.Radius());
      }
    }
  }
}

///////////////////////////////////////////////
/// Photonintersector
///////////////////////////////////////////////


inline std::tuple<double, double> PointLineDistance(const Ray &r, const Double3 &p)
{
  Double3 d = p - r.org;
  double distance = Dot(d, r.dir);
  double radial_distance2 = LengthSqr(Cross(d, r.dir));
  return std::make_tuple(distance, radial_distance2);
}


TEST(Photonintersector, Photonintersector)
{
  // Like the hash grid case. But here we use ray intersections.
  const double radius = 0.2;
  Sampler sampler;
  ToyVector<Double3> points;
  for (int i=0; i<10; ++i)
  {
    points.emplace_back(SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare()));
  }
  PhotonIntersector intersector(radius, points);
  ToyVector<bool> reported_in_range(points.size(), false);
  ToyVector<double> reported_distance(points.size());
  for (int j=0; j<100; ++j)
  {
    Ray ray {
      SampleTrafo::ToUniformDisc(sampler.UniformUnitSquare())*1.5,
      {0,0,1}};
    ray.org[2] = -2.;
    const double distance = 2.;

    std::fill(reported_in_range.begin(), reported_in_range.end(), false);
    constexpr int BUFFER_SIZE = 1024;
    int items[BUFFER_SIZE];
    float distances[BUFFER_SIZE];
    int n = intersector.Query(ray, distance, items, distances, BUFFER_SIZE);
    for (int i=0; i<n; ++i)
    {
      reported_in_range[items[i]] = true;
      reported_distance[items[i]] = distances[i];
    }
    for (int i=0; i<points.size(); ++i)
    {
      auto [d, r2] = PointLineDistance(ray, points[i]);
      bool in_range = d>=0. && d<=distance && r2 <= radius*radius;
      EXPECT_EQ(in_range, reported_in_range[i]);
      if (reported_in_range[i])
        EXPECT_NEAR(d, reported_distance[i], 1.e-3);
    }
  }
}

///////////////////////////////////////////////
////   Light Picker
///////////////////////////////////////////////
#if 0
TEST(LightPickerUcb, SelectionProbabilities)
{
  const char* scenestr = R"""(
{
larea arealight2 uniform 1 1 1 1
diffuse black  1 1 1 0.
m testing/scenes/unitrectangle.obj
}

l 0 0.75 0  1 1 1 1

lddome 0 1 0  0.02 0.02 0.5
)""";

  Scene scene;
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();
  using Lights::MakeLightRef;
  using Lights::LightRef;
  namespace ROI = RadianceOrImportance;

  const int NUM_PICKERS = 4;
  const int NUM_LIGHTS = 4;
  const int NUM_ITERS = 2;
  const int NUM_INNER_ITERS = 4;

  std::array<UcbLightPicker, NUM_PICKERS> pickers{
    scene, scene, scene, scene
  };
  pickers[0].Init();
  for (int i=1; i<NUM_PICKERS; ++i)
    pickers[i].InitSharedWith(pickers[0]);
  pickers[0].UpdateAllShared();

  auto CheckDistributionProbabilities = [&](const UcbLightPicker &picker, Span<const double> desired_probs)
  {
    const double p_env = picker.PmfOfLight(MakeLightRef(scene, scene.GetTotalEnvLight()));
    const double p_area1 = picker.PmfOfLight(MakeLightRef(scene, scene.GetPrimitiveFromAreaLightIndex(0)));
    const double p_area2 = picker.PmfOfLight(MakeLightRef(scene, scene.GetPrimitiveFromAreaLightIndex(1)));
    const double p_point = picker.PmfOfLight(MakeLightRef(scene, scene.GetPointLight(0)));
    const double prob_array[4] = {
      p_env, p_area1, p_area2, p_point
    };
    for (int i = 0; i < NUM_LIGHTS; ++i)
      ASSERT_LE(std::abs(prob_array[i] - desired_probs[i]), 0.1);
  };

  const auto UNIFORM_PROBS =
    Eigen::Array<double, NUM_LIGHTS, 1>::Constant(1. / NUM_LIGHTS).eval();
  const double TARGET_PROBS[NUM_LIGHTS] = {
    1. / 10., 2. / 10., 3. / 10., 4. / 10.
  };

  for (const auto &picker : pickers)
    CheckDistributionProbabilities(picker, AsSpan(UNIFORM_PROBS));

  struct Callback
  {
    UcbLightPicker &picker;
    void operator()(const Lights::Env &, double prob, LightRef ref)
    {
      picker.ObserveLightContribution(ref, Spectral3{ 1. });
    }
    void operator()(const Lights::Point &, double prob, LightRef ref)
    {
      picker.ObserveLightContribution(ref, Spectral3{ 4. });
    }
    void operator()(const Lights::Area &light, double prob, LightRef ref)
    {
      const double val = light.Get().index == 0 ? 2. : 3.;
      picker.ObserveLightContribution(ref, Spectral3{ val });
    }
    void operator()(const Medium &, double prob, LightRef ref) 
    {
      picker.ObserveLightContribution(ref, Spectral3::Zero());
    }
  };

  Sampler sampler;
  int picker_idx = 0;
  for (int iter = 0; iter < NUM_ITERS; ++iter)
  {
    for (int inner = 0; inner < NUM_INNER_ITERS; ++inner)
    {
      pickers[picker_idx].PickLight(sampler, Callback{ pickers[picker_idx] });
      picker_idx = (picker_idx + 1) % NUM_PICKERS;
    }
    pickers[0].UpdateAllShared();
  }

  for (const auto &picker : pickers)
    CheckDistributionProbabilities(picker, Span<const double>(TARGET_PROBS, NUM_LIGHTS));
}
#endif

//////////////////////////////////////////////
/// JSON
//////////////////////////////////////////////

#ifdef HAVE_JSON
TEST(JSON, EigenToJson)
{
  using namespace rapidjson_util;
  Double3 x{1,2,3};
  Float3 y{4,5,6};
  Index3 i{1,2,3};

  rj::Document doc;
  auto &alloc = doc.GetAllocator();
  doc.SetObject();
  doc.AddMember("Double3", ToJSON(x, alloc), alloc);
  doc.AddMember("Float3", ToJSON(y, alloc), alloc);
  doc.AddMember("Index3", ToJSON(i, alloc), alloc);

  const std::string s = ToString(doc);
  //std::cout << s << std::endl;

  const std::string expected = \
R"""({
    "Double3": [
        1.0,
        2.0,
        3.0
    ],
    "Float3": [
        4.0,
        5.0,
        6.0
    ],
    "Index3": [
        1,
        2,
        3
    ]
})""";
  EXPECT_EQ(s, expected);
}


TEST(JSON, ContainerToJSON)
{
  using namespace rapidjson_util;
  std::list<ToyVector<int>> c;
  c.push_back({1,2,3});
  c.push_back({4,5,6});

  rj::Document doc;
  auto &a = doc.GetAllocator();
  doc.SetObject();
  doc.AddMember("Container", ToJSON(c, a), a);

  const std::string s = ToString(doc);
  //std::cout << s << std::endl;

  const std::string expected = \
R"""({
    "Container": [
        [
            1,
            2,
            3
        ],
        [
            4,
            5,
            6
        ]
    ]
})""";

  EXPECT_EQ(expected, s);
}
#endif

//////////////////////////////////////////////
// pmr pointer bump allocator
//////////////////////////////////////////////


struct alignas(128) PmrAlignTester
{
  PmrAlignTester(int x_) : x(x_)
  {
    //std::cout << "PmrAlignTester(" << x_ << ")" << std::endl;
  }
  ~PmrAlignTester()
  {
    //std::cout << "~PmrAlignTester(" << x << ")" << std::endl;
  }
  int x;
};


TEST(MemoryArena, Alignment)
{
  util::MemoryArena arena{1024*4096};
  auto t1 = arena.MakeUnique<PmrAlignTester>(7);
  auto t2 = arena.MakeUnique<PmrAlignTester>(42);
  size_t misalign1 = (size_t)t1.get() % 128;
  size_t misalign2 = (size_t)t2.get() % 128;
  ASSERT_EQ(misalign1, 0);
  ASSERT_EQ(misalign2, 0);
}


TEST(MemoryArena, DefaultBufferSize)
{
  constexpr size_t itemsize = sizeof(PmrAlignTester);
  constexpr size_t n = 100000; 
  constexpr size_t sz = itemsize * (n+1); // +1 because alignment, to not cause additional allocations other than the first.
  util::MemoryArena arena{sz};

  auto AllocationLoop = [&]() { 
    auto last = arena.MakeUnique<PmrAlignTester>(0);
    for (size_t i = 1; i<n; ++i)
    { 
      auto current = arena.MakeUnique<PmrAlignTester>(static_cast<int>(i));
      ASSERT_EQ(current.get() - last.get(), 1); // If it's really just bumping the pointer ...
      last = std::move(current);
    }
  };

  AllocationLoop();

  arena.Release();

  AllocationLoop();
}


//////////////////////////////////////////////
// nd array and lookup table
//////////////////////////////////////////////

TEST(NdArray, Lattice)
{
  LatticeIndexing<4, long> lattice;
  lattice = LatticeIndexing<4,long>({ 2l, 3l, 3l, 2l });
  const auto s = lattice.Size();
  ASSERT_EQ(s[0], 2l);
  ASSERT_EQ(s[1], 3l);
  ASSERT_EQ(s[2], 3l);
  ASSERT_EQ(s[3], 2l);

  ASSERT_EQ(lattice.LatticeToSite({1l, 0l, 0l, 0l}), 1l);
  ASSERT_EQ(lattice.LatticeToSite({1l, 2l, 2l, 1l}), lattice.NumSites()-1);

  const auto pt = lattice.SiteToLattice(lattice.LatticeToSite({0l,2l,1l,1l}));
  ASSERT_TRUE(
    (pt == Eigen::Array<long,4,1>({0l,2l,1l,1l})).all()
  );
}


TEST(NdArray, RuleOfFive)
{
  using A = SimpleNdArray<int, 3>;
  A a{{2,2,2}};
  a[{0,0,0}] = 42;
  a[{1,1,1}] = 666;
  
  Eigen::Array<int,3,1> i0{0,0,0}, i1{1,1,1};

  // Copy C'tor
  A b{a};
  ASSERT_EQ(b[i0], 42);
  ASSERT_EQ(b[i1], 666);
  
  // Move C'tor
  A c{std::move(a)};
  ASSERT_EQ(c.size(), 2*2*2);
  ASSERT_EQ(c[i0], 42);
  ASSERT_EQ(c[i1], 666);
  ASSERT_EQ(a.size(), 0);

  // Copy Assign
  a = c;
  ASSERT_EQ(a[i0], 42);
  ASSERT_EQ(a[i1], 666);

  // Move Assign
  // First clear c
  std::fill(c.begin(), c.end(), 0);
  ASSERT_EQ(c[i0], 0);
  ASSERT_EQ(c[i1], 0);
  c = std::move(a);
  ASSERT_EQ(c[i0], 42);
  ASSERT_EQ(c[i1], 666);
  ASSERT_EQ(a.size(), 2*2*2); // Still has the memory from c
}


//////////////////////////////////////////////
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
