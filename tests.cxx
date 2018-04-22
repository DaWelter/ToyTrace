#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <rapidjson/document.h>
#include <boost/filesystem.hpp>

#include "ray.hxx"
#include "image.hxx"
#include "perspectivecamera.hxx"
#include "infiniteplane.hxx"
#include "sampler.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
#include "sphere.hxx"
#include "triangle.hxx"
#include "atmosphere.hxx"
#include "util.hxx"


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



class TestIntersection : public testing::Test
{
protected:
  RaySurfaceIntersection intersection;
  double distance;
  
  void Intersect(const Primitive &prim, const Ray &ray, bool expect_hit = true)
  {
    RaySegment rs{ray, LargeNumber};
    HitId hit;
    bool bhit = prim.Intersect(rs.ray, rs.length, hit);
    ASSERT_TRUE(bhit == expect_hit);
    if (bhit)
    {
      distance = rs.length;
      intersection = RaySurfaceIntersection{hit, rs};
    }
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


TEST_F(TestIntersection, Sphere)
{
  Sphere s{{0., 0., 2.}, 2.};
  Intersect(s, {{0., 0., -1.},{0., 0., 1.}});
  EXPECT_NEAR(distance, 1., 1.e-6);
  CheckPosition({0., 0., 0.});
  CheckNormal({0.,0.,-1.});
}


//TEST_F(TestIntersection, SphereRepeatedIntersection)
//{
//  Sphere s{{0., 0., 0.}, 6300.};
//  Ray r{{-10000., 0., 0.}, {1., 0., 0.}};
//  std::cout << std::setprecision(std::numeric_limits<double>::digits10) << "one ulp at r=6300 is" << boost::math::float_advance<double>(6300.,1) - 6300. << std::endl;
//  for (int iter=0; iter<100; ++iter)
//  {
//    Intersect(s, r);
//    EXPECT_LE(intersection.pos[0], 0.);
//    EXPECT_NEAR(intersection.pos[1], 0., 1.e-6);
//    EXPECT_NEAR(intersection.pos[2], 0., 1.e-6);
//    r.org[0] = (r.org[0]+6300.)*0.5 - 6300.;
//    std::cout << std::setprecision(std::numeric_limits<double>::digits10) << "start=" << r.org[0] << " xpos=" << intersection.pos[0] << std::endl;
//  }
//}


TEST_F(TestIntersection, Triangle)
{
  /*     x
   *   / |
   *  /  |
   * x---x
   * Depiction of the triangle */
  double q = 0.5;
  Triangle prim{{-q, -q, 0}, {q, -q, 0}, {q, q, 0}};
  Intersect(prim, Ray{{0.1, 0., -1.},{0., 0., 1.}});
  EXPECT_NEAR(distance, 1., 1.e-6);
  CheckPosition({0.1, 0., 0.});
  CheckNormal({0., 0., -1.});
}


TEST_F(TestIntersection, TriangleEdgeCase)
{
  double q = 0.5;
  Triangle prim{{-q, -q, 0}, {q, -q, 0}, {q, q, 0}};
  Intersect(prim, Ray{{0., 0., -1.},{0., 0., 1.}});
  EXPECT_NEAR(distance, 1., 1.e-6);
  CheckPosition({0., 0., 0.});
  CheckNormal({0., 0., -1.});
  // Slightly offset the ray, there should be no intersection.
  Intersect(prim, Ray{{-Epsilon, 0., -1.},{0., 0., 1.}}, false);
  // An adjacent triangle should catch the ray, however.
  // This one covers the top left corner.
  Triangle prim_top_left{{-q, -q, 0}, {-q, q, 0}, {q, q, 0}};
  Intersect(prim_top_left, Ray{{-Epsilon, 0., -1.},{0., 0., 1.}}, true);
}


inline RaySegment MakeSegmentAt(const RaySurfaceIntersection &intersection, const Double3 &ray_dir)
{
 return RaySegment{
   {intersection.pos + AntiSelfIntersectionOffset(intersection, RAY_EPSILON, ray_dir), ray_dir},
   LargeNumber
 };
}


TEST(TestMath, Reflected)
{
  Double3 n{0., 1., 0.};
  auto in = Normalized(Double3{0., 1., 2.});
  Double3 out = Reflected(in, n);
  auto out_expected = Normalized(Double3{0., 1., -2.});
  ASSERT_GE(Dot(out, out_expected), 0.99);
}


// Adapted from pbrt. eta is the ratio of refractive indices eta_i / eta_t
inline boost::optional<Double3> Refracted(const Double3 &wi, const Double3 &n, double eta_i_over_t) 
{
    const double eta = eta_i_over_t;
    double cosThetaI = Dot(n, wi);
    double sin2ThetaI = std::max(double(0), double(1 - cosThetaI * cosThetaI));
    double sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return boost::none;
    double cosThetaT = std::sqrt(1 - sin2ThetaT);
    double n_prefactor = (eta * std::abs(cosThetaI) - cosThetaT);
           n_prefactor = cosThetaI>=0. ? n_prefactor : -n_prefactor; // Invariance to normal flip.
    return Double3{-eta * wi + n_prefactor * n};
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
  // Test total reflection.
  w1 = Normalized(Double3{0., 1., 100.});
  eta1 = 1.2;
  eta2 = 1.;
  w2 = Refracted(w1, n, eta1/eta2);
  ASSERT_FALSE((bool)w2);
}


TEST(TestMath, SphereIntersectionMadness)
{
  Sampler sampler;
  Double3 sphere_org{0., 0., 2.};
  double sphere_rad = 6300;
  Sphere s{sphere_org, sphere_rad};
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
    HitId hit1;
    bool bhit = s.Intersect(rs.ray, rs.length, hit1);
    ASSERT_TRUE(bhit);
    RaySurfaceIntersection intersect1{hit1, rs};
    ASSERT_LE(Length(org - intersect1.pos), Length(org - sphere_org));
    
    // Put the origina at at the intersection and shoot a ray to the outside
    // in a random direction. Expect no further hit.
    auto m  = OrthogonalSystemZAligned(intersect1.geometry_normal);
    auto new_dir = m * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    rs = MakeSegmentAt(intersect1, new_dir);
    bhit = s.Intersect(rs.ray, rs.length, hit1);
    ASSERT_FALSE(bhit);
    ASSERT_EQ(rs.length, LargeNumber);
  }
}


TEST(TestMath, SphereIntersectionMadness2)
{
  Sampler sampler;
  Double3 sphere_org{0., 0., 2.};
  double sphere_rad = 6300;
  Sphere s{sphere_org, sphere_rad};
  RaySegment rs{{sphere_org, {1., 0., 0.}}, LargeNumber};
  HitId hit, last_hit;
  const int N = 100;
  for (int num = 0; num < N; ++num)
  {
    bool bhit = s.Intersect(rs.ray, rs.length, hit); //, last_hit, HitId());
    ASSERT_TRUE(bhit);
    RaySurfaceIntersection intersect{hit, rs};
    double rho = Length(intersect.pos-sphere_org);
    auto m  = OrthogonalSystemZAligned(intersect.normal);
    auto new_dir = m * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    rs = MakeSegmentAt(intersect, new_dir);
    rho = Length(rs.ray.org-sphere_org);
    EXPECT_LE(rho, sphere_rad);
    last_hit = hit;
  }
}


TEST(TestMath, SphereIntersectionMadness3)
{
  Sampler sampler;
  Scene scene;
  double rad1 = 6300.;
  double rad2 = 6350.;
  scene.AddPrimitive<Sphere>(Double3{0.}, rad1);
  scene.AddPrimitive<Sphere>(Double3{0.}, rad2);
  scene.BuildAccelStructure();
  Double3 start_point = {0.5*(rad1+rad2), 0., 0.};
  double max_dist_from_start = 0.;
  RaySegment rs{{start_point, {1., 0., 0.}}, LargeNumber};
  HitId hit, last_hit;
  const int N = 100;
  for (int num = 0; num < N; ++num)
  {
    hit = scene.MakeIntersectionCalculator().First(rs.ray, rs.length);
    ASSERT_TRUE((bool)hit);
    RaySurfaceIntersection intersect{hit, rs};
    double rho = Length(intersect.pos);
    max_dist_from_start = std::max(max_dist_from_start, Length(intersect.pos-start_point));
    EXPECT_LE(rho-RAY_EPSILON, rad2);
    EXPECT_GE(rho+RAY_EPSILON, rad1);
    auto m  = OrthogonalSystemZAligned(intersect.normal);
    auto new_dir = m * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    rs = MakeSegmentAt(intersect, new_dir);
    last_hit = hit;
  }
  EXPECT_GE(max_dist_from_start, rad1*1.e-3);
}



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
  double bound_at_z1 = 0.5;
  Sampler sampler;
  InfinitePlane imageplane;
  std::unique_ptr<PerspectiveCamera> cam;
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
        PathContext{Color::LambdaIdxClosestToRGBPrimaries()});
      HitId hit;
      double length = 100.;
      bool is_hit = imageplane.Intersect({pos, s.coordinates}, length, hit);
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

namespace {
void CheckSceneParsedWithScopes(const Scene &scene);
}

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
include %
{
include %
}
s 14 15 16 0.5
)""";
  auto path3 = fs::temp_directory_path() / fs::unique_path("scene3-%%%%-%%%%-%%%%-%%%%.nff");
  std::string scenestr = strformat(scenestr_fmt, path1.string(), path2.string());
  {
    std::ofstream os(path3.string());
    os.write(scenestr.c_str(), scenestr.size());
  }  
  Scene scene;
  scene.ParseNFF(path3);
  CheckSceneParsedWithScopes(scene);
}


namespace 
{
  
void CheckSceneParsedWithScopes(const Scene &scene)
{
  auto Getter = [&scene](int i) -> auto 
  { 
    return scene.GetPrimitive(i).CalcBounds().Center();
  };
  ASSERT_EQ(scene.GetNumPrimitives(), 4);
  std::array<Double3,4> c{ 
    Getter(0),
    Getter(1),
    Getter(2),
    Getter(3)
  };
  // Checking the coordinates for correct application of the transform statements.
  ASSERT_NEAR(c[0][0], 1., 1.e-3);
  ASSERT_NEAR(c[1][0], 5., 1.e-3);
  ASSERT_NEAR(c[2][0], 8.+11., 1.e-3); // Using the child scope transform.
  ASSERT_NEAR(c[3][0], 14.+5., 1.e-3); // Using the parent scope transform.
  ASSERT_EQ(scene.GetPrimitive(3).shader, scene.GetPrimitive(1).shader); // Shaders don't persist beyond scopes.
  ASSERT_NE(scene.GetPrimitive(2).shader, scene.GetPrimitive(1).shader); // Shader within the scope was actually created and assigned.
  ASSERT_NE(scene.GetPrimitive(2).shader, scene.GetPrimitive(0).shader); // Shader within the scope is not the default.
  ASSERT_NE(scene.GetPrimitive(2).shader, nullptr); // And ofc it should not be null.
}
  
}



TEST(Parser, ImportDAE)
{
  const char* scenestr = R"""(
diffuse DefaultMaterial 1 1 1 0.5
m testing/scenes/unitcube.dae
)""";
  Scene scene;
  scene.ParseNFFString(scenestr);
  constexpr double tol = 1.e-2;
  Box outside; 
  outside.Extend({-0.5-tol, -0.5-tol, -0.5-tol}); 
  outside.Extend({0.5+tol, 0.5+tol, 0.5+tol});
  Box inside;
  inside.Extend({-0.5+tol, -0.5+tol, -0.5+tol});
  inside.Extend({0.5-tol, 0.5-tol, 0.5-tol});
  EXPECT_EQ(scene.GetNumPrimitives(), 6 * 2);
  for (int i=0; i<scene.GetNumPrimitives(); ++i)
  {
    auto* prim = dynamic_cast<const Triangle*>(&scene.GetPrimitive(i));
    ASSERT_NE(prim, nullptr);
    Box b = prim->CalcBounds();
    ASSERT_TRUE(b.InBox(outside));
    ASSERT_TRUE(!b.InBox(inside));
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
  Box b = scene.CalcBounds();
  double size = Length(b.max - b.min);
  ASSERT_GE(size, 1.);
  ASSERT_LE(size, 3.);
}




namespace Atmosphere
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
  ASSERT_GT(sigma_t.Size(), 0);
  const auto& sigma_t_at_altitude0 = sigma_t[0];
  ASSERT_EQ(sigma_t_at_altitude0.Size(), Color::NBINS);
  const auto& value = sigma_t_at_altitude0[0];
  ASSERT_TRUE(std::isfinite(value.GetDouble()));
}


}


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


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}