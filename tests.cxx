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



TEST(TestMath, RaySphereIntersection)
{
  Sphere s{{0., 0., 2.}, 2., nullptr};
  RaySegment rs{Ray{{0., 0., -1.},{0., 0., 1.}}, LargeNumber};
  HitId hit;
  double length = LargeNumber;
  bool bhit = s.Intersect(rs.ray, rs.length, hit);
  EXPECT_NEAR(rs.length, 1., 1.e-6);
  ASSERT_TRUE(bhit);
  RaySurfaceIntersection intersect{hit, rs};
  EXPECT_NEAR(intersect.pos[0], 0., 1.e-6);
  EXPECT_NEAR(intersect.pos[1], 0., 1.e-6);
  EXPECT_NEAR(intersect.pos[2], 0., 1.e-6);
  EXPECT_NEAR(intersect.normal[0], 0., 1.e-6);
  EXPECT_NEAR(intersect.normal[1], 0., 1.e-6);
  EXPECT_NEAR(intersect.normal[2], -1., 1.e-6);
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
    imageplane({0, 0, 1}, {0, 0, 1}, nullptr)
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




int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}