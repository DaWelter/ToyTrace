#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>

#include "ray.hxx"
#include "image.hxx"
#include "sampler.hxx"


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
  // First expectance of depositing one number in a particular bin. 
  // Random variable is 1 in one of N cases, and 0 in the others.
  double avg = 1./N;
  double var = (N-1)*avg*avg + (1.-avg)*(1.-avg);
  // Standard deviation of the calculated average from the true expectation.
  double sigma_of_avg = std::sqrt(var / M);
  // Allow 5 sigma deviation.
  std::cout << "avg=" << avg << " var=" << var << " sigma=" << sigma_of_avg << std::endl;
  int count_min = (avg - 5 * sigma_of_avg) * M;
  int count_max = (avg + 5 * sigma_of_avg) * M;
  std::cout << " bounds=" << count_min << ", " << count_max << std::endl;
  for (int k = 0; k < N; ++k)
  {
    ASSERT_GE(counts[k], count_min);
    ASSERT_LE(counts[k], count_max);
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


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}