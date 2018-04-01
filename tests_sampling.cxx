#include "gtest/gtest.h"
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "ray.hxx"
#include "sampler.hxx"
#include "sphere.hxx"
#include "triangle.hxx"
#include "util.hxx"
#include "shader.hxx"
#include "sampler.hxx"


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


void CheckNumberOfSamplesInBin(const char *name, int num_smpl_in_bin, int total_num_smpl, double p_of_bin, double number_of_sigmas_threshold=3.)
{
  double mean, sigma;
  std::tie(mean, sigma) = MeanAndSigmaOfThrowingOneWithPandZeroOtherwise(p_of_bin);
  mean *= total_num_smpl;
  sigma = SigmaOfAverage(total_num_smpl, sigma * total_num_smpl);
  if (name)
    std::cout << "Expected in " << name << ": " << mean << "+/-" << sigma << " Actual: " << num_smpl_in_bin << " of " << total_num_smpl << std::endl;
  EXPECT_NEAR(num_smpl_in_bin, mean, sigma*number_of_sigmas_threshold);
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


TEST_F(RandomSamplingFixture, DiscDistribution)
{
  static constexpr int N = 100;
  static constexpr int NUM_REGIONS=5;
  int n_samples[NUM_REGIONS] = {};
  const double center_disc_radius = 0.5;
  const double center_disc_area = Pi*Sqr(center_disc_radius);
  const double outer_quadrant_area = (Pi - center_disc_area) * 0.25;
  const double region_area[NUM_REGIONS] = 
  {
    center_disc_area,
    outer_quadrant_area,
    outer_quadrant_area,
    outer_quadrant_area,
    outer_quadrant_area
  };
  for (int i=0; i<N; ++i)
  {
    Double3 v = SampleTrafo::ToUniformDisc(sampler.UniformUnitSquare());
    ASSERT_EQ(v[2], 0.);
    ASSERT_LE(Length(v), 1.);
    int region = 0;
    if (Length(v)>center_disc_radius)
    {
      if (v[0]>0 && v[1]>0) region=1;
      if (v[0]>0 && v[1]<=0) region=2;
      if (v[0]<=0 && v[1]>0) region=3;
      if (v[0]<=0 && v[1]<=0) region=4;
    }
    ++n_samples[region];
  }
  for (int bin=0; bin<NUM_REGIONS; ++bin)
  {
    CheckNumberOfSamplesInBin("disc_bin", n_samples[bin], N, region_area[bin]/Pi);
  }
}



TEST_F(RandomSamplingFixture, TriangleSampling)
{
  Eigen::Matrix3d m;
  auto X = m.col(0);
  auto Y = m.col(1);
  auto Z = m.col(2);
  X = Double3{0.4, 0.2, 0.};
  Y = Double3{0.1, 0.8, 0.};
  Z = Double3{0. , 0. , 1.};
  Eigen::Matrix3d minv = m.inverse();
  Double3 offset{0.1, 0.1, 0.1};
  Triangle t(
    offset,
    offset+X,
    offset+Y);
  static constexpr int NUM_SAMPLES = 100;
  for (int i=0; i<NUM_SAMPLES; ++i)
  {
    {
      Double3 barry = SampleTrafo::ToTriangleBarycentricCoords(sampler.UniformUnitSquare());
      EXPECT_NEAR(barry[0]+barry[1]+barry[2], 1., 1.e-3);
      EXPECT_GE(barry[0], 0.); EXPECT_LE(barry[0], 1.);
      EXPECT_GE(barry[1], 0.); EXPECT_LE(barry[1], 1.);
      EXPECT_GE(barry[2], 0.); EXPECT_LE(barry[2], 1.);
    }
    HitId hit = t.SampleUniformPosition(sampler);
    Double3 pos, normal, shading_normal;
    t.GetLocalGeometry(hit, pos, normal, shading_normal);
    Double3 local = minv * (pos - offset);
    EXPECT_NEAR(local[2], 0., 1.e-3);
    EXPECT_GE(local[0], 0.); EXPECT_LE(local[0], 1.);
    EXPECT_GE(local[1], 0.); EXPECT_LE(local[1], 1.);
    EXPECT_LE(local[0] + local[1], 1.);
  }
}


class PhasefunctionTests : public testing::Test
{
public:
  Sampler sampler;
  static constexpr int Nbins = 9;
  std::ofstream output;  // If really deperate!

  // Use a cube mapping of 6 uniform grids to the 6 side faces of a cube projected to the unit sphere.
  using CubeBinDouble = std::array<std::array<std::array<double, Nbins>, Nbins>, 6>;
  using CubeVertexDouble = std::array<std::array<std::array<double, Nbins+1>, Nbins+1>, 6>;
    
  void TestPfSampling(const PhaseFunctions::PhaseFunction &pf, const Double3 &reverse_incident_dir, int num_samples, double number_of_sigmas_threshold)
  {
    /* I want to check if the sampling frequency of some solid angle areas match the probabilities
     * returned by the Evaluate function. I further want to validate the normalization constraint. */
    auto bin_probabilities = ComputeBinProbabilities(pf, reverse_incident_dir);
    Spectral3 integral{0.};
    int bin_sample_count[6][Nbins][Nbins] = {}; // Zero initialize?
    for (int snum = 0; snum<num_samples; ++snum)
    {
      auto smpl = pf.SampleDirection(reverse_incident_dir, sampler);
      int side, i, j;
      CalcCubeIndices(smpl.coordinates, side, i, j);
      i = std::max(0, std::min(i, Nbins-1));
      j = std::max(0, std::min(j, Nbins-1));
      bin_sample_count[side][i][j] += 1.;
      integral += smpl.value / smpl.pdf_or_pmf;
      if (output.is_open())
        output << smpl.coordinates[0] << " " << smpl.coordinates[1] << " " << smpl.coordinates[2] << " " << smpl.pdf_or_pmf << std::endl;
    }
    integral /= num_samples; // Due to the normalization contraint, this integral should equal 1.
    EXPECT_NEAR(integral[0], 1., 0.01);
    EXPECT_NEAR(integral[1], 1., 0.01);
    EXPECT_NEAR(integral[2], 1., 0.01);
    for (int side = 0; side < 6; ++side)
    {
      for (int i = 0; i<Nbins; ++i)
      {
        for (int j = 0; j<Nbins; ++j)
        {
          CheckNumberOfSamplesInBin(
            nullptr, //strconcat(side,"[",i,",",j,"]").c_str(),
            bin_sample_count[side][i][j], 
            num_samples, 
            bin_probabilities[side][i][j],
            number_of_sigmas_threshold);
        }
      }
    }
  }
 
 
  CubeBinDouble ComputeBinProbabilities(const PhaseFunctions::PhaseFunction &pf, const Double3 &reverse_incident_dir)
  {
    // Cube projection.
    // +/-x,y,z which makes 6 grid, one per side face, and i,j indices of the corresponding grid.
    CubeVertexDouble bin_corner_densities;
    CubeBinDouble bin_probabilities;
    for (int side = 0; side < 6; ++side)
    {
      for (int i = 0; i<=Nbins; ++i)
      {
        for (int j=0; j<=Nbins; ++j)
        {
          Double3 corner = CalcCubeGridVertex(side, i, j);
          pf.Evaluate(reverse_incident_dir, corner, &bin_corner_densities[side][i][j]);
          if (output.is_open())
            output << corner[0] << " " << corner[1] << " " << corner[2] << " " << bin_corner_densities[side][i][j] << std::endl;
        }
      }
    }
    if (output.is_open())
      output << std::endl;

    double total_probability = 0.;
    double total_solid_angle = 0.;
    for (int side = 0; side < 6; ++side)
    {
      for (int i = 0; i<Nbins; ++i)
      {
        for (int j = 0; j<Nbins; ++j)
        {
          Double3 w[2][2] = {
            { 
              CalcCubeGridVertex(side, i, j),
              CalcCubeGridVertex(side, i, j+1) 
            },
            {
              CalcCubeGridVertex(side, i+1, j),
              CalcCubeGridVertex(side, i+1, j+1)
            }
          };
          // Crude approximation of the surface integral over pieces of the unit sphere.
          // https://en.wikipedia.org/wiki/Surface_integral
          Double3 du = 0.5 * (w[1][0]+w[1][1] - w[0][0]-w[0][1]);
          Double3 dv = 0.5 * (w[0][1]+w[1][1] - w[0][0]-w[1][0]);
          double spanned_solid_angle = Length(Cross(du,dv));
          bin_probabilities[side][i][j] = 0.25 * spanned_solid_angle *
            (bin_corner_densities[side][i][j] + 
             bin_corner_densities[side][i+1][j] + 
             bin_corner_densities[side][i+1][j+1] + 
             bin_corner_densities[side][i][j+1]);
          total_probability += bin_probabilities[side][i][j];
          total_solid_angle += spanned_solid_angle;
        }
      }
    }
    EXPECT_NEAR(total_solid_angle, UnitSphereSurfaceArea, 0.1);
    EXPECT_NEAR(total_probability, 1.0, 0.01);
//    // Renormalize to make probabilities sum to one.
//    // This accounts for inaccuracies through which
//    // the sum of the original probabilities deviated from 1.
//    for (int side = 0; side < 6; ++side)
//    {
//      for (int i = 0; i<Nbins; ++i)
//      {
//        for (int j = 0; j<Nbins; ++j)
//        {
//          bin_probabilities[side][i][j] /= total_probability;
//        }
//      }
//    }
    return bin_probabilities;
  }
  
  
  Double3 CalcCubeGridVertex(int side, int i, int j)
  {
    double delta = 1./Nbins;
    double u = 2.*i*delta-1.;
    double v = 2.*j*delta-1.;
    Double3 x;
    switch (side)
    {
      case 0:
        x = Double3{1., u, v}; break;
      case 1:
        x = Double3{-1., u, v}; break;
      case 2:
        x = Double3{u, 1., v}; break;
      case 3:
        x = Double3{u, -1., v}; break;
      case 4:
        x = Double3{u, v, 1.}; break;
      case 5:
        x = Double3{u, v, -1.}; break;
      default:
        EXPECT_FALSE((bool)"We should not get here.");
    }
    return Normalized(x);
  }
  
  
  void CalcCubeIndices(const Double3 &w, int &side, int &i, int &j)
  {
    double u, v, z;
    int max_abs_axis;
    z = w.array().abs().maxCoeff(&max_abs_axis);
    side = max_abs_axis*2 + (w[max_abs_axis]>0 ? 0 : 1);
    switch(side)
    {
      case 0:
        u = w[1]; v = w[2]; z = w[0]; break;
      case 1:
        u = w[1]; v = w[2]; z = -w[0]; break;
      case 2:
        u = w[0]; v = w[2]; z = w[1]; break;
      case 3:
        u = w[0]; v = w[2]; z = -w[1]; break;
      case 4:
        u = w[0]; v = w[1]; z = w[2]; break;
      case 5:
        u = w[0]; v = w[1]; z = -w[2]; break;
      default:
        FAIL();
    }
    ASSERT_GT(z, 0.);
    u /= z;
    v /= z;
    i = (u+1.)*0.5*Nbins;
    j = (v+1.)*0.5*Nbins;
  }
};


TEST_F(PhasefunctionTests, CubeGridMapping)
{
  // I need a test for the test!
  for (int side = 0; side < 6; ++side)
  {
    std::cout << "Corners of side " << side << std::endl;
    std::cout << CalcCubeGridVertex(side, 0, 0) << std::endl;
    std::cout << CalcCubeGridVertex(side, Nbins, 0) << std::endl;
    std::cout << CalcCubeGridVertex(side, Nbins, Nbins) << std::endl;
    std::cout << CalcCubeGridVertex(side, 0, Nbins) << std::endl;

    for (int i = 0; i<Nbins; ++i)
    {
      for (int j=0; j<Nbins; ++j)
      {
        Double3 center = 0.25 * (
            CalcCubeGridVertex(side, i, j) +
            CalcCubeGridVertex(side, i+1, j) +
            CalcCubeGridVertex(side, i+1, j+1) +
            CalcCubeGridVertex(side, i, j+1));
        int side_, i_, j_;
        CalcCubeIndices(center, side_, i_, j_);
        EXPECT_EQ(side_, side);
        EXPECT_EQ(i, i_);
        EXPECT_EQ(j, j_);
      }
    }
  }
}


TEST_F(PhasefunctionTests, Uniform)
{
  PhaseFunctions::Uniform pf;
  TestPfSampling(pf, Double3{0,0,1}, 10000, 3.5);
}


TEST_F(PhasefunctionTests, Rayleigh)
{
  // output.open("PhasefunctionTests.Rayleight.txt");
  PhaseFunctions::Rayleigh pf;
  TestPfSampling(pf, Double3{0,0,1}, 10000, 3.5);
}


TEST_F(PhasefunctionTests, HenleyGreenstein)
{
  //output.open("PhasefunctionTests.HenleyGreenstein.txt");
  PhaseFunctions::HenleyGreenstein pf(0.4);
  TestPfSampling(pf, Double3{0,0,1}, 10000, 4.0);
}


TEST_F(PhasefunctionTests, Combined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::Combined pf{{1., 1., 1.}, {.1, .2, .3}, pf1, {.3, .4, .5}, pf2};
  TestPfSampling(pf, Double3{0,0,1}, 10000, 3.5);
}


TEST_F(PhasefunctionTests, SimpleCombined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::SimpleCombined pf{{.1, .2, .3}, pf1, {.3, .4, .5}, pf2};
  TestPfSampling(pf, Double3{0,0,1}, 10000, 3.5);
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



TEST(TestMath, TowerSampling)
{
  Sampler sampler;
  double probs[] = { 0, 1, 5, 1, 0 };
  constexpr int n = sizeof(probs)/sizeof(double);
  double norm = 0.;
  for (auto p : probs)
    norm += p;
  for (auto &p : probs)
    p /= norm;
  int bins[n] = {};
  constexpr int Nsamples = 1000;
  for (int i = 0; i < Nsamples; ++i)
  {
    int bin = TowerSampling<n>(probs, sampler.Uniform01());
    EXPECT_GE(bin, 0);
    EXPECT_LE(bin, n-1);
    bins[bin]++;
  }
  for (int i=0; i<n; ++i)
  {
    CheckNumberOfSamplesInBin(strconcat("Bin[",i,"]").c_str(), bins[i], Nsamples, probs[i]);
  }
}



TEST_F(RandomSamplingFixture, HomogeneousTransmissionSampling)
{
  ImageDisplay display;
  Image img{500, 10};
  img.SetColor(64, 64, 64);
  RGB length_scales{1._rgb, 5._rgb, 10._rgb};
  double cutoff_length = 5.; // Integrate transmission T(x) up to this x.
  double img_dx = img.width() / cutoff_length / 2.;
  img.DrawRect(0, 0, img_dx * cutoff_length, img.height()-1);
  Index3 lambda_idx = Color::LambdaIdxClosestToRGBPrimaries();
  SpectralN sigma_s = Color::RGBToSpectrum(1._rgb/length_scales);
  SpectralN sigma_a = Color::RGBToSpectrum(1._rgb/length_scales);
  SpectralN sigma_t = sigma_s + sigma_a;
  HomogeneousMedium medium{
    sigma_s, 
    sigma_a, 
    0};
  int N = 1000;
  Spectral3 integral{0.};
  for (int i=0; i<N; ++i)
  {
    PathContext context{lambda_idx};
    Medium::InteractionSample s = medium.SampleInteractionPoint(RaySegment{{{0.,0.,0.}, {0., 0., 1.,}}, cutoff_length}, sampler, context);
    if (s.t  > cutoff_length)
      img.SetColor(255, 0, 0);
    else
    {
      img.SetColor(255, 255, 255);
      // Estimate transmission by integrating sigma_e*T(x).
      // Divide out sigma_s which is baked into the weight.
      // Multiply by sigma_e.
      SpectralN integrand = (sigma_a + sigma_s) / sigma_s;
      integral += s.weight * Take(integrand, lambda_idx);
    }
    int imgx = std::min<int>(img.width()-1, s.t * img_dx);
    img.DrawLine(imgx, 0, imgx, img.height()-1);
  }
  integral *= 1./N;
  Spectral3 exact_solution = Take((1. - (-sigma_t*cutoff_length).exp()).eval(), lambda_idx);
  for (int k=0; k<static_size<Spectral3>(); ++k)
    EXPECT_NEAR(integral[k], exact_solution[k], 0.1 * integral[k]);
  //display.show(img);
  //std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

