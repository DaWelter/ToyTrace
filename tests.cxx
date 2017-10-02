#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>

#include "ray.hxx"
#include "image.hxx"
#include "perspectivecamera.hxx"
#include "infiniteplane.hxx"
#include "sampler.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
#include "sphere.hxx"
#include "triangle.hxx"


TEST(BasicAssumptions, EigenTypes)
{
  // Spectral is currently an Eigen::Array type. It is still a row vector/array.
  EXPECT_EQ(Spectral::ColsAtCompileTime, 1);
  EXPECT_EQ(Spectral::RowsAtCompileTime, 3);
  // Vectors in eigen are Eigen::Matrix row vectors.
  EXPECT_EQ(Double3::ColsAtCompileTime, 1);
  EXPECT_EQ(Double3::RowsAtCompileTime, 3);
};


// Throw a one with probability p and zero with probability 1-p.
// Let this be reflected by random variable X \in {0, 1}.
// What is the expectation E[X] and std deviation sqrt(Var[X])?
std::pair<double, double> MeanAndSigmaOfThrowingOneWithPandZeroOtherwise(double p)
{
  // Expected hit indication = 1 * P + 0 * (1-P) = P
  // Variance of single throw = (1-P)^2 * P + (0-P)^2 * (1-P) = P - 2 PP + PPP + PP - PPP = P - PP
  return std::make_pair(p, std::sqrt(p-p*p));
}

// For a sample average <X_i> = 1/N sum_i X_i, 
// <X_i> is a random variable itself. It has some distribution
// around the true expectation E[X]. The standard deviation 
// of this distribution is:
double SigmaOfAverage(int N, double sample_sigma)
{
  return sample_sigma/std::sqrt(N);
}


void CheckNumberOfSamplesInBin(const char *name, int Nbin, int N, double p_of_bin)
{
  double mean, sigma;
  std::tie(mean, sigma) = MeanAndSigmaOfThrowingOneWithPandZeroOtherwise(p_of_bin);
  mean *= N;
  sigma = SigmaOfAverage(N, sigma * N);
  
  std::cout << "Expected in " << name << ": " << mean << "+/-" << sigma << " Actual: " << Nbin << " of " << N << std::endl;
  EXPECT_NEAR(Nbin, mean, sigma*3);
}



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
    Eigen::Matrix<double, 3, 3> m = OrthogonalSystemZAligned(dir);
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


TEST(TestMath, SphereIntersectionMadness)
{
  Sampler sampler;
  Double3 sphere_org{0., 0., 2.};
  double sphere_rad = 1.;
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
      * sphere_rad * 0.75 + sphere_org;
    Double3 dir = target - org; 
    Normalize(dir);
    
    // Shoot ray to the inside. Expect intersection at the
    // front side of the sphere.
    RaySegment rs{{org, dir}, LargeNumber};
    HitId hit1, hit2, hit3;
    double length = LargeNumber;
    bool bhit = s.Intersect(rs.ray, rs.length, hit1);
    ASSERT_TRUE(bhit);
    RaySurfaceIntersection intersect1{hit1, rs};
    ASSERT_LE(Length(org - intersect1.pos), Length(org - sphere_org));
    
    // Put new ray origin at the intersection and ignore the
    // intersection for further intersection computations.
    // Now expect the next intersection at the back side of the sphere.
    rs.ray.org = intersect1.pos;
    rs.length = LargeNumber;
    bhit = s.Intersect(rs.ray, rs.length, hit2, hit1, hit3);
    ASSERT_TRUE(bhit);
    RaySurfaceIntersection intersect2{hit2, rs};
    ASSERT_LE(Length(org - intersect1.pos), Length(org - intersect2.pos));
    ASSERT_GE((intersect1.pos - intersect2.pos).norm(), 1.e-3 * sphere_rad);

    // Put the ray origin back to the start. Now we ignore all two previous
    // intersection. Hence we expect to get no new intersection.
    rs.ray.org = org;
    rs.length = LargeNumber;
    bhit = s.Intersect(rs.ray, rs.length, hit3, hit1, hit2);
    ASSERT_FALSE(bhit);
    ASSERT_EQ(rs.length, LargeNumber);
  }
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



class RandomSamplingFixture : public ::testing::Test
{
protected:
  Sampler sampler;
public:
  RandomSamplingFixture() {}
};


TEST_F(RandomSamplingFixture, Uniform01)
{
  static constexpr int N = 1000;
  double buffer[N];
  sampler.Uniform01(buffer, N);
  double max_elem = *std::max_element(buffer, buffer+N);
  double min_elem = *std::min_element(buffer, buffer+N);
  ASSERT_LE(max_elem, 1.0);
  ASSERT_GE(max_elem, 0.5);  // With pretty high probability ...
  ASSERT_GE(min_elem, 0.0);
  ASSERT_LE(min_elem, 0.5);
}


TEST_F(RandomSamplingFixture, UniformIntInclusive)
{
  int i = sampler.UniformInt(10,10);
  ASSERT_EQ(i, 10);
}


TEST_F(RandomSamplingFixture, UniformIntDistribution)
{
  // Put random throws in bins and do statistics on the number of hits per bin.
  static constexpr int N = 5;
  int counts[N] = { 0, 0, 0, 0, 0 };
  static constexpr int M = N * 1000;
  for (int i = 0; i < M; ++i)
  {
    int k = sampler.UniformInt(0,N-1);
    ASSERT_LE(k, N-1);
    ASSERT_GE(k, 0);
    ++counts[k];
  }
  for (int k = 0; k < N; ++k)
  {
    std::stringstream ss; ss << "'uniform int bin " << k << "'";
    CheckNumberOfSamplesInBin(ss.str().c_str(), counts[k], M, 1./N);
  }
}


TEST_F(RandomSamplingFixture, UniformSphereDistribution)
{
  static constexpr int N = 100;
  for (int i=0; i<N; ++i)
  {
    Double3 v = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
    double len = Length(v);
    ASSERT_NEAR(len, 1.0, 1.e-6);
  }
}


TEST_F(RandomSamplingFixture, UniformHemisphereDistribution)
{
  static constexpr int N = 100;
  for (int i=0; i<N; ++i)
  {
    Double3 v = SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    double len = Length(v);
    ASSERT_NEAR(len, 1.0, 1.e-6);
    ASSERT_GE(v[2], 0.);
  }
}


TEST_F(RandomSamplingFixture, CosHemisphereDistribution)
{
  static constexpr int N = 100;
  int n_samples_per_quadrant[2][2] = { {0, 0}, {0, 0} };
  int n_samples_z_test[3] = { 0, 0, 0, };
  const double z_thresholds[3] = { 0.25 * Pi*0.5, 0.5 * Pi*0.5, 0.75 * Pi*0.5 };
  for (int i=0; i<N; ++i)
  {
    auto v = SampleTrafo::ToCosHemisphere(sampler.UniformUnitSquare());
    ASSERT_NEAR(Length(v), 1.0, 1.e-6);
    ASSERT_GE(v[2], 0.);
    ++n_samples_per_quadrant[v[0]<0. ? 0 : 1][v[1]<0. ? 0 : 1];
    auto angle = std::acos(v[2]);
    if (angle<z_thresholds[2]) ++n_samples_z_test[2];
    if (angle<z_thresholds[1]) ++n_samples_z_test[1];
    if (angle<z_thresholds[0]) ++n_samples_z_test[0];
  }
  const char* bin_names[2][2] = {
    { "00", "01" }, { "10", "11" }
  };
  for (int qx = 0; qx <= 1; ++qx)
  for (int qy = 0; qy <= 1; ++qy)
  {
    CheckNumberOfSamplesInBin(bin_names[qx][qy], n_samples_per_quadrant[qx][qy], N, 0.25);
  }
  for (int z_bin = 0; z_bin < 3; ++z_bin)
  {
    std::stringstream ss; ss << "'Theta<" << (z_thresholds[z_bin]*180./Pi) <<"deg'";
    double p = std::pow(std::sin(z_thresholds[z_bin]), 2.);
    CheckNumberOfSamplesInBin(ss.str().c_str(), n_samples_z_test[z_bin], N, p);
  }
}


TEST_F(RandomSamplingFixture, HomogeneousTransmissionSampling)
{
  ImageDisplay display;
  Image img{500, 10};
  img.SetColor(64, 64, 64);
  Spectral length_scales{1., 5., 10.};
  double cutoff_length = 5.; // Integrate transmission T(x) up to this x.
  double img_dx = img.width() / cutoff_length / 2.;
  img.DrawRect(0, 0, img_dx * cutoff_length, img.height());
  Spectral sigma_s{0.}, sigma_a{1./length_scales};
  IsotropicHomogeneousMedium medium{sigma_s, sigma_a, 0};
  int N = 1000;
  Spectral integral{0.};
  for (int i=0; i<N; ++i)
  {
    Medium::InteractionSample s = medium.SampleInteractionPoint(RaySegment{{{0.,0.,0.}, {0., 0., 1.,}}, cutoff_length}, sampler);
    EXPECT_TRUE(s.t < cutoff_length || (s.transmission.abs().sum() < 1.e-6));
    for (int k=0; k<static_size<Spectral>(); ++k)
    {
      EXPECT_NEAR(s.sigma_a[k], sigma_a[k], 1.e-6);
      EXPECT_NEAR(s.sigma_s[k], sigma_s[k], 1.e-6);
    }
    if (s.transmission.abs().sum() < 1.e-6)
      img.SetColor(255, 0, 0);
    else
      img.SetColor(255 * s.transmission[0], 255 * s.transmission[1], 255 * s.transmission[2]);
    int imgx = std::min<int>(img.width()-1, s.t * img_dx);
    img.DrawLine(imgx, 0, imgx, img.height()-1);
    integral += s.transmission / s.pdf;
  }
  integral *= 1./N;
  Spectral exact_solution = length_scales * (1. - (-(sigma_a+sigma_s)*cutoff_length).exp());
  for (int k=0; k<static_size<Spectral>(); ++k)
    EXPECT_NEAR(integral[k], exact_solution[k], 0.1 * integral[k]);
  display.show(img);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}



class StratifiedFixture : public testing::Test
{
public:
  Stratified2DSamples stratified;
  StratifiedFixture() :
    stratified(2,2)
  {}
  
  Double2 Get(Double2 v) 
  { 
    return stratified.UniformUnitSquare(v); 
  }
  
  void CheckTrafo(double r1, double r2)
  {
    stratified.current_x = 0;
    stratified.current_y = 0;
    Double2 r = Get({r1, r2});
    ASSERT_FLOAT_EQ(r[0], r1 * 0.5);
    ASSERT_FLOAT_EQ(r[1], r2 * 0.5);
  }
  
  void Next()
  {
    stratified.UniformUnitSquare({0.,0.});
  }
};


TEST_F(StratifiedFixture, Incrementing)
{
  ASSERT_EQ(stratified.current_x, 0);
  ASSERT_EQ(stratified.current_y, 0);
  Next();
  ASSERT_EQ(stratified.current_x, 1);
  ASSERT_EQ(stratified.current_y, 0);
  Next();
  ASSERT_EQ(stratified.current_x, 0);
  ASSERT_EQ(stratified.current_y, 1);
  Next();
  ASSERT_EQ(stratified.current_x, 1);
  ASSERT_EQ(stratified.current_y, 1);
  Next();
  ASSERT_EQ(stratified.current_x, 0);
  ASSERT_EQ(stratified.current_y, 0);
  Next();
}


TEST_F(StratifiedFixture, SampleLocation)
{
  CheckTrafo(0., 0.);
  CheckTrafo(1., 1.);
  CheckTrafo(0.5, 0.5);
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
    cam->current_pixel_x = _pixel_x;
    cam->current_pixel_y = _pixel_y;
    Double3 possum{0,0,0}, possqr{0,0,0};
    Box box;
    constexpr int Nsamples = 100;
    for (int i = 0; i < Nsamples; ++i)
    {
      auto s = cam->TakeDirectionSampleFrom(pos, sampler);
      HitId hit;
      double length = 100.;
      bool is_hit = imageplane.Intersect(s.ray_out, length, hit);
      ASSERT_TRUE(is_hit);
      Double3 endpos = s.ray_out.org + length * s.ray_out.dir;
      possum += endpos;
      possqr += Product(endpos, endpos);
      box.Extend(endpos);
    }
    // int_x=-s-s x^2 / (2s) dx = 1/3 x^3 /2s | x=-s..s = 1/(6s) 2*(s^3) = 1/3 s^2. -> std = sqrt(1/3) s
    // s = 1 -> std = 0.577
    Double3 average = possum / Nsamples;
    Double3 stddev  = (possqr / Nsamples - Product(average, average)).array().sqrt().matrix();
    std::cout << "@Pixel: ix=" << cam->current_pixel_x << "," << cam->current_pixel_y << std::endl;
    std::cout << "Pixel: " << average << " +/- " << stddev << std::endl;
    Double3 exactpos {((cam->current_pixel_x + 0.5) * 2.0 / cam->xres - 1.0),
                      ((cam->current_pixel_y + 0.5) * 2.0 / cam->yres - 1.0),
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



class SimpleRenderTests : public testing::Test
{
protected:
  Scene scene;
  void SetCameraSimple(double z_distance, double fov, int xres, int yres)
  {
    scene.SetCamera<PerspectiveCamera>(
      Double3{0., 0., z_distance},
      Double3{0., 0., 1.},
      Double3{0., 1., 0.},
      fov,
      xres, yres);
  }

  void ToNextIntersection(RaySegment &seg, RaySurfaceIntersection &intersection)
  {
    seg.length = LargeNumber;
    HitId hit = scene.Intersect(seg.ray, seg.length, intersection.hitid);
    EXPECT_TRUE(hit == true);
    intersection = RaySurfaceIntersection{hit, seg};
    seg.ray.org = intersection.pos;
  }
};


TEST_F(SimpleRenderTests, OnePixelBackground)
{
  SetCameraSimple(0., 90., 1, 1);
  scene.bgColor = Double3{ 0.5, 0., 0. };
  scene.BuildAccelStructure();
  Raytracing rt(scene);
  auto col = rt.MakePrettyPixel();
  ASSERT_FLOAT_EQ(col[0], 0.5);
  ASSERT_FLOAT_EQ(col[1], 0.);
  ASSERT_FLOAT_EQ(col[2], 0.);
}


TEST_F(SimpleRenderTests, MediaTracker)
{
  scene.ParseNFF("scenes/test_media.nff");
  scene.BuildAccelStructure();
  const Medium *vac = &scene.GetEmptySpaceMedium();
  const Medium *m1 = scene.GetPrimitive(0).medium;
  const Medium *m2 = scene.GetPrimitive(1).medium;
  std::printf("Pointers:\n vac=%p, m1=%p, m2=%p\n", vac, m1, m2);
  for (int i=0; i<5; ++i)
  {
    std::printf("Medium of prim %i = %p\n", i, scene.GetPrimitive(i).medium);
  }
  MediumTracker mt(scene);
  mt.initializePosition({0, 0, -10});
  ASSERT_EQ(&mt.getCurrentMedium(), vac);
  RaySegment seg{{{0, 0, -10}, {0, 0, 1}}, LargeNumber};
  RaySurfaceIntersection intersection;
  const Medium *media_after_intersect[] = {
    m1, vac, m2, m2, m1, vac, m1, m2, m1, vac
  };
  double intersect_pos[] = {
    -2, 0,   1,  2,  3,   4,  5,  5.5,6.5, 7
  };
  std::printf("Media Trace:\n");
  for (int i=0; i<sizeof(media_after_intersect)/sizeof(void*); ++i)
  {
    ToNextIntersection(seg, intersection);
    mt.goingThroughSurface(seg.ray.dir, intersection);
    std::printf("IS[%i]: pos=%f, med_expect=%p, got=%p\n", i, intersection.pos[2], media_after_intersect[i], &mt.getCurrentMedium());
    EXPECT_NEAR(intersection.pos[2], intersect_pos[i], 1.e-6);
    ASSERT_EQ(&mt.getCurrentMedium(), media_after_intersect[i]);
  }
}


TEST_F(SimpleRenderTests, ImportDAE)
{
  scene.ParseNFF("scenes/test_dae.nff");
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


TEST_F(SimpleRenderTests, ImportDAE2)
{
  scene.ParseNFF("scenes/test_dae2.nff");
  Box b = scene.CalcBounds();
  double size = Length(b.max - b.min);
  ASSERT_GE(size, 1.);
  ASSERT_LE(size, 3.);
}


TEST_F(SimpleRenderTests, MediaTransmission)
{
  scene.ParseNFF("scenes/test_dae3.nff");
  scene.BuildAccelStructure();
  scene.PrintInfo();
  Raytracing rt(scene);
  MediumTracker medium_tracker(scene);
  medium_tracker.initializePosition({0.,0.,-10.});
  double ray_offset = 0.1; // because not so robust handling of intersection edge cases. No pun intended.
  RaySegment seg{{{ray_offset,0.,-10.}, {0.,0.,1.}}, LargeNumber};
  auto res = rt.TransmittanceEstimate(seg, HitId(), medium_tracker);
  
  auto sigma_e = Spectral{3.};
  Spectral expected = (-2.*sigma_e).exp();
  
  for (int i=0; i<static_size<Spectral>(); ++i)
    ASSERT_NEAR(res[i], expected[i], 1.e-3);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
