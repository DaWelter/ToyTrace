#include "gtest/gtest.h"
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <boost/math/distributions/chi_squared.hpp>

#include "ray.hxx"
#include "sampler.hxx"
#include "sphere.hxx"
#include "triangle.hxx"
#include "util.hxx"
#include "shader.hxx"
#include "sampler.hxx"

#include "cubature.h"


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


double ChiSquaredProbability(const int *counts, const double *weights, int num_bins)
{
  double probability_renormalization = 0.;
  int num_nonzero_bins = 0;
  int num_samples = 0;
  for (int i=0; i<num_bins; ++i)
  {
    if (weights[i] > 0.)
    {
      EXPECT_GE(counts[i], 5);
      num_samples += counts[i];
      ++num_nonzero_bins;
      probability_renormalization += weights[i];
    }
    else
    {
      EXPECT_EQ(counts[i], 0);
    }
  }
  probability_renormalization = probability_renormalization>0. ? 1./probability_renormalization : 0.;
  double chi_sqr = 0.;
  for (int i=0; i<num_bins; ++i)
  {
    if (weights[i] > 0.)
    {
      double expected = weights[i]*probability_renormalization*num_samples;
      chi_sqr += Sqr(counts[i]-expected)/expected;
    }
  }

  if (num_nonzero_bins <= 0)
    return 1.;

  // https://www.boost.org/doc/libs/1_66_0/libs/math/doc/html/math_toolkit/dist_ref/dists/chi_squared_dist.html
  // https://www.boost.org/doc/libs/1_66_0/libs/math/doc/html/math_toolkit/dist_ref/nmp.html#math_toolkit.dist_ref.nmp.cdf
  boost::math::chi_squared_distribution<double> distribution(num_nonzero_bins-1);
  double prob_observe_ge_chi_sqr = cdf(complement(distribution, chi_sqr));
  
  //EXPECT_GE(prob_observe_ge_chi_sqr, p_threshold);
  return prob_observe_ge_chi_sqr;  
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


class CubeMap
{
  // Use a cube mapping of 6 uniform grids to the 6 side faces of a cube projected to the unit sphere.
  const int bins_per_axis;
  
  // (Mostly) auto generated by my notebook IntegrationOnManifoldsSympy.
  double cube_map_J(double u, double v) const
  {
    using std::pow;
    using std::sqrt;
    double cube_map_J_result;
    cube_map_J_result = sqrt((-pow(u, 2)*pow(v, 2)*pow(pow(u, 2) + pow(v, 2) + 1.0, 2) + (pow(u, 2)*pow(v, 2) + pow(u, 2) + pow(pow(v, 2) + 1.0, 2))*(pow(u, 2)*pow(v, 2) + pow(v, 2) + pow(pow(u, 2) + 1.0, 2)))/pow(pow(u, 2) + pow(v, 2) + 1.0, 6));
    return cube_map_J_result;
  }
  
public:
  CubeMap(int _bins_per_axis) :
    bins_per_axis{_bins_per_axis}
  {}
  
  int TotalNumBins() const
  {
    return bins_per_axis*bins_per_axis*6;
  }
  
  // i,j indices of the grid vertices.
  Double2 VertexToUV(int i, int j) const
  {
    assert (i >= 0 && i <= bins_per_axis);
    assert (j >= 0 && j <= bins_per_axis);
    double delta = 1./bins_per_axis;
    double u = 2.*i*delta-1.;
    double v = 2.*j*delta-1.;
    return Double2{u, v};
  }
  
  // Normally, like here, i,j are indices of the grid cells.
  std::tuple<Double2, Double2> CellToUVBounds(int i, int j)
  {
    return std::make_tuple(
      VertexToUV(i, j),
      VertexToUV(i+1, j+1));
  }
  
  Double3 UVToOmega(int side, Double2 uv) const
  {
    double u = uv[0];
    double v=  uv[1];
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
  
  double UVtoJ(Double2 uv) const
  {
    return cube_map_J(uv[0], uv[1]);
  }
  
  std::tuple<int, int, int> OmegaToCell(const Double3 &w) const
  {
    int side, i, j;
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
        assert(!"Should not get here!");
    }
    assert(z > 0.);
    u /= z;
    v /= z;
    assert(u >= -1.001 && u <= 1.001);
    assert(v >= -1.001 && v <= 1.001);
    i = (u+1.)*0.5*bins_per_axis;
    j = (v+1.)*0.5*bins_per_axis;
    i = std::max(0, std::min(bins_per_axis-1, i));
    j = std::max(0, std::min(bins_per_axis-1, j));
    return std::make_tuple(side, i, j);
  }
  
  int CellToIndex(int side, int i, int j) const
  {
    assert (i >= 0 && i < bins_per_axis);
    assert (j >= 0 && j < bins_per_axis);
    assert (side >= 0 && side < 6);
    const int num_per_side = bins_per_axis*bins_per_axis;
    return side*num_per_side + i*bins_per_axis + j;
  }
  
  std::tuple<int, int, int> IndexToCell(int idx) const
  {
    const int num_per_side = bins_per_axis*bins_per_axis;
    assert (idx >= 0 && idx < num_per_side*6);
    int side = idx / num_per_side;
    idx %= num_per_side;
    int i = idx / bins_per_axis;
    idx %= bins_per_axis;
    int j = idx;
    assert (i >= 0 && i < bins_per_axis);
    assert (j >= 0 && j < bins_per_axis);
    assert (side >= 0 && side < 6);
    return std::make_tuple(side, i, j);
  }
};


TEST(CubeMap, IndexMapping)
{
  // I need a test for the test!
  const int SZ = 9;
  CubeMap cm{SZ};
  
  for (int side = 0; side < 6; ++side)
  {
    std::cout << "Corners of side " << side << std::endl;
    std::cout << cm.UVToOmega(side, cm.VertexToUV(0, 0)) << std::endl;
    std::cout << cm.UVToOmega(side, cm.VertexToUV(SZ, 0)) << std::endl;
    std::cout << cm.UVToOmega(side, cm.VertexToUV(SZ, SZ)) << std::endl;
    std::cout << cm.UVToOmega(side, cm.VertexToUV(0, SZ)) << std::endl;

    for (int i = 0; i<SZ; ++i)
    {
      for (int j=0; j<SZ; ++j)
      {
        int idx = cm.CellToIndex(side, i, j);
        int _side, _i, _j;
        std::tie(_side, _i, _j) = cm.IndexToCell(idx);
        EXPECT_EQ(_side, side);
        EXPECT_EQ(_i, i);
        EXPECT_EQ(_j, j);
        
        Double2 uv0, uv1;
        std::tie(uv0, uv1) = cm.CellToUVBounds(i, j);
        Double2 uv_center = 0.5*(uv0 + uv1);
        Double3 omega = cm.UVToOmega(side, uv_center);
        ASSERT_NORMALIZED(omega);
        std::tie(_side, _i, _j) = cm.OmegaToCell(omega);
        EXPECT_EQ(side, _side);
        EXPECT_EQ(i, _i);
        EXPECT_EQ(j, _j);
      }
    }
  }
}


namespace cubature_wrapper
{

template<class Func>
int f1d(unsigned ndim, const double *x, void *fdata_,
      unsigned fdim, double *fval)
{
  auto* callable = static_cast<Func*>(fdata_);
  assert(ndim == 2);
  assert(fdim == 1);
  Eigen::Map<const Eigen::Vector2d> xm{x};
  *fval = (*callable)(xm);
  return 0;
}

template<class Func>
int fnd(unsigned ndim, const double *x, void *fdata_,
        unsigned fdim, double *fval)
{
  auto* callable = static_cast<Func*>(fdata_);
  assert(ndim == 2);
  Eigen::Map<const Eigen::Vector2d> xm{x};
  const auto value = ((*callable)(xm)).eval();
  std::copy(value.data(), value.data()+fdim, fval);
  return 0;
}

} // namespace cubature_wrapper


namespace 
{
template<class Func>
double Integral2D(Func func, Double2 start, Double2 end, double absError, double relError)
{
  double result, error;
  int status = hcubature(1, cubature_wrapper::f1d<Func>, &func, 2, start.data(), end.data(), 0, absError, relError, ERROR_L2, &result, &error);
  if (status != 0)
    throw std::runtime_error("Cubature failed!");
  return result;
}

template<class Func>
auto MultivariateIntegral2D(Func func, Double2 start, Double2 end, double absError, double relError) -> decltype(func(Double2{}))
{
  using R  = decltype(func(Double2{}));
  R result, error;
  int status = hcubature(result.size(), cubature_wrapper::fnd<Func>, &func, 2, start.data(), end.data(), 0, absError, relError, ERROR_L2, result.data(), error.data());
  if (status != 0)
    throw std::runtime_error("Cubature failed!");
  return result;
}

} // namespace


TEST(Cubature, Integral)
{
  auto func = [](const Double2 x) -> double
  {
    return x[0]*x[0] + x[1]*x[1];
  };
  
  double result = Integral2D(
    func, Double2(-1,-1), Double2(1,1), 0.01, 0.01);
  
  double exact = 8./3.;
  ASSERT_NEAR(result, exact, 1.e-2);
}

TEST(Cubature, MultivariateIntegral)
{
  auto func = [](const Double2 x) -> Eigen::Array2d
  {
    double f1 = x[0]*x[0] + x[1]*x[1];
    double f2 = 1.;
    Eigen::Array2d a; a << f1, f2;
    return a;
  };
  
  Eigen::Array2d result = MultivariateIntegral2D(
    func, Double2(-1,-1), Double2(1,1), 0.01, 0.01);
  
  double exact1 = 8./3.;
  double exact2 = 4.;
  ASSERT_NEAR(result[0], exact1, 1.e-2);
  ASSERT_NEAR(result[1], exact2, 1.e-2);
}


TEST(Cubature, SphereSurfacArea)
{
  CubeMap cm(5);
  double result = 0.;
  // See notebook misc/IntegrationOnManifoldsSympy.*
  auto func = [&](const Double2 x) -> double
  {
    return cm.UVtoJ(x);
  };
  
  for (int idx=0; idx<cm.TotalNumBins(); ++idx)
  {
    int side, i, j;
    std::tie(side, i, j) = cm.IndexToCell(idx);
    Double2 start, end;
    std::tie(start, end) = cm.CellToUVBounds(i, j);
    result += Integral2D(func, start, end, 1.e-2, 1.e-2);
  }
  
  ASSERT_NEAR(result, UnitSphereSurfaceArea, 1.e-2);
}


class ScatterTests : public testing::Test
{
  /* I want to check if the sampling frequency of some solid angle areas match the probabilities
    * returned by the Evaluate function. I further want to validate the normalization constraint. */
public:
  CubeMap cubemap;
  Sampler sampler;
  static constexpr int Nbins = 4;
  std::ofstream output;  // If really desperate!

  // Filled by member functions.
  std::vector<int> bin_sample_count;
  std::vector<double> bin_probabilities;
  double total_probability;
  Spectral3 integral_estimate;
  Spectral3 integral_estimate_delta;
  Spectral3 integral_cubature;
  double probability_of_delta_peaks;
  int num_samples;
  
  ScatterTests()
    : cubemap{Nbins}
  {}
  
  // Template Method Design Pattern ?!
  // Override that in derived classes for Phasefunction and BSDF, respectively.
  virtual double ProbabilityDensity(const Double3 &omega) const = 0;
  virtual Spectral3 ScatterFunction(const Double3 &omega) const = 0;
  virtual ScatterSample MakeScatterSample() = 0;
  virtual void CheckSymmetry() = 0;
  
  void RunAllCalculations(int num_samples)
  {
    DoSampling(num_samples);
    ComputeBinProbabilities();
    ComputeCubatureIntegral();
    CheckSymmetry();
  }
  
  
  void TestCountsDeviationFromSigma(double number_of_sigmas_threshold)
  {
    for (int idx=0; idx<cubemap.TotalNumBins(); ++idx)
    {
      int side, i, j;
      std::tie(side, i, j) = cubemap.IndexToCell(idx);
      CheckNumberOfSamplesInBin(
        nullptr, //strconcat(side,"[",i,",",j,"]").c_str(),
        bin_sample_count[idx], 
        num_samples, 
        bin_probabilities[idx],
        number_of_sigmas_threshold);
    }
  }
  

  void TestChiSqr(double p_threshold)
  {
    double chi_sqr_probability = ChiSquaredProbability(&bin_sample_count[0], &bin_probabilities[0], cubemap.TotalNumBins());
    std::cout << "chi sqr probability = " << chi_sqr_probability << std::endl;
    EXPECT_GE(chi_sqr_probability, p_threshold);
  }
  
  void CheckIntegral(double tolerance, boost::optional<Spectral3> by_value = boost::none)
  {
    for (int i=0; i<integral_cubature.size(); ++i)
    {
      EXPECT_NEAR(integral_cubature[i], integral_estimate[i], tolerance);
      if (by_value)
      {
        // As a user of the test, I probably only know the combined albedo of 
        // specular and diffuse/glossy parts. So take that for comparison.
        Spectral3 total_reflected = integral_estimate + integral_estimate_delta;
        EXPECT_NEAR(total_reflected[i], (*by_value)[i], tolerance);
      }
    }
    EXPECT_NEAR(total_probability, 1.-probability_of_delta_peaks, 1.e-3);
  }
 
  void DoSampling(int num_samples)
  {
    this->bin_sample_count = std::vector<int>(cubemap.TotalNumBins(), 0);
    this->integral_estimate = Spectral3{0.};  
    this->num_samples = num_samples;
    this->probability_of_delta_peaks = 0.;
    std::vector<Double3> delta_peaks;
    auto CheckDeltaPeakNewAndMemorize = [&delta_peaks](const Double3 &v) -> bool 
    {
      auto it = std::find_if(delta_peaks.begin(), delta_peaks.end(), [&](const Double3 &v) { return v.cwiseEqual(v).all(); });
      bool is_new = it == delta_peaks.end();
      if (is_new) delta_peaks.push_back(v);
      return is_new;
    };
    auto SampleIndex = [this](const ScatterSample &smpl) ->  int
    {
      int side, i, j;
      std::tie(side, i, j) = cubemap.OmegaToCell(smpl.coordinates);
      return cubemap.CellToIndex(side, i, j);
    };
    auto CrossValidate = [this](const ScatterSample &smpl) -> void
    {
      // Do pdf and scatter function values in the smpl struct match what comes out of member functions?
      double pdf_crossvalidate = this->ProbabilityDensity(smpl.coordinates);
      ASSERT_NEAR(PdfValue(smpl), pdf_crossvalidate, 1.e-6);
      Spectral3 val_crossvalidate = this->ScatterFunction(smpl.coordinates);
      ASSERT_NEAR(smpl.value[0], val_crossvalidate[0], 1.e-6);
      ASSERT_NEAR(smpl.value[1], val_crossvalidate[1], 1.e-6);
      ASSERT_NEAR(smpl.value[2], val_crossvalidate[2], 1.e-6);
    };
    
    for (int snum = 0; snum<num_samples; ++snum)
    {
      auto smpl = this->MakeScatterSample();

      if (IsFromPdf(smpl) && snum<100)
        CrossValidate(smpl);
      // Can only integrate the non-delta density. 
      // Keep track of delta-peaks separately.
      if (IsFromPdf(smpl))
      {
        int idx = SampleIndex(smpl);
        bin_sample_count[idx] += 1;
        integral_estimate += smpl.value / smpl.pdf_or_pmf; 
      }
      else
      {
        // I can ofc integrate by MC.
        integral_estimate_delta += smpl.value / smpl.pdf_or_pmf;
        // "pdf_or_pmf" is the probability of selecting the delta-peak.
        if (CheckDeltaPeakNewAndMemorize(smpl.coordinates))
          probability_of_delta_peaks += PmfValue(smpl);
      }

      if (output.is_open())
        output << smpl.coordinates[0] << " " << smpl.coordinates[1] << " " << smpl.coordinates[2] << " " << smpl.pdf_or_pmf << std::endl;
    }
    integral_estimate /= num_samples;
    integral_estimate_delta /= num_samples;
  }
 
 
  void ComputeBinProbabilities()
  {
    this->bin_probabilities = std::vector<double>(cubemap.TotalNumBins(), 0.);
    this->total_probability = 0.;
    for (int side = 0; side < 6; ++side)
    {
      for (int i = 0; i<Nbins; ++i)
      {
        for (int j = 0; j<Nbins; ++j)
        {
          int idx = cubemap.CellToIndex(side, i, j);
          // Integrating this gives me the probability of a sample falling
          // into the current bin. The integration is carried out on a square in
          // R^2, hence the scale factor J from the Jacobian of the variable transform.
          auto probabilityDensityTimesJ = [&](const Double2 x) -> double
          {
            Double3 omega = cubemap.UVToOmega(side, x);
            double pdf = this->ProbabilityDensity(omega);
            return pdf*cubemap.UVtoJ(x);
          };
          // By design, only diffuse/glossy components will be captured by this.
          Double2 start, end;
          std::tie(start, end) = cubemap.CellToUVBounds(i, j);       
          double prob = Integral2D(probabilityDensityTimesJ, start, end, 1.e-3, 1.e-2);
          bin_probabilities[idx] = prob;
          total_probability += bin_probabilities[idx];
        }
      }
    }
  }
  
  
  void ComputeCubatureIntegral()
  {
    this->integral_cubature = 0.;
    for (int idx=0; idx<cubemap.TotalNumBins(); ++idx)
    {
      int side, i, j;
      std::tie(side, i, j) = cubemap.IndexToCell(idx);
      auto functionValueTimesJ = [&](const Double2 x) -> Eigen::Array3d
      {
        Double3 omega = cubemap.UVToOmega(side, x);
        Spectral3 val = this->ScatterFunction(omega);
        return val*cubemap.UVtoJ(x);
      };
      Double2 start, end;
      std::tie(start, end) = cubemap.CellToUVBounds(i, j);       
      Eigen::Array3d cell_integral = MultivariateIntegral2D(functionValueTimesJ, start, end, 1.e-3, 1.e-2);
      this->integral_cubature += cell_integral;
    }
  }
  
  
  // Supply function returning pdf and scatter function value.
  // Need this because no access to reverse_incident_dir. Also
  // ScatterFunction might have |wi.shadingnormal| term backed in
  // which we do not want here!
  //void CheckSymmetry(
    //std::function<std::pair<double, Spectral3> (const Double3 &wi, const Double3 &wo)> func)
  template<class Func>
  void CheckSymmetryImpl(Func func)
  {
    const int NUM_SAMPLES = 100;
    for (int snum = 0; snum<NUM_SAMPLES; ++snum)
    {
      Double3 wi = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
      Double3 wo = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
      double pdf_val, pdf_rev;
      Spectral3 f_val, f_rev;
      std::tie(pdf_val, f_val) = func(wi, wo);
      std::tie(pdf_rev, f_rev) = func(wo, wi);
      EXPECT_NEAR(pdf_val, pdf_rev, 1.e-6);
      EXPECT_NEAR(f_val[0], f_rev[0], 1.e-6);
      EXPECT_NEAR(f_val[1], f_rev[1], 1.e-6);
      EXPECT_NEAR(f_val[2], f_rev[2], 1.e-6);
    }
  }
};


class PhasefunctionTests : public ScatterTests
{
private:
  const PhaseFunctions::PhaseFunction *pf;
  Double3 reverse_incident_dir;
  
public:
  void RunAllCalculations(const PhaseFunctions::PhaseFunction &pf, const Double3 &reverse_incident_dir, int num_samples)
  {
    this->pf = &pf;
    this->reverse_incident_dir = reverse_incident_dir;
    ScatterTests::RunAllCalculations(num_samples);
  }
  
  void CheckIntegral(double tolerance)
  {
    ScatterTests::CheckIntegral(tolerance, Spectral3{1.});
  }
  
  void CheckSymmetry() override
  {
    CheckSymmetryImpl([&](const Double3 &wi, const Double3 &wo) -> std::pair<double, Spectral3>
    {
      double pdf = NaN;
      auto ret = pf->Evaluate(wi, wo, &pdf);
      return std::make_pair(pdf, ret);
    });
  }
  
private:
  double ProbabilityDensity(const Double3 &omega) const override
  {
    double ret;
    pf->Evaluate(reverse_incident_dir, omega, &ret);
    return ret;
  }
  
  Spectral3 ScatterFunction(const Double3 &omega) const override
  {
    return pf->Evaluate(reverse_incident_dir, omega, nullptr);
  }
  
  ScatterSample MakeScatterSample() override
  {
    return pf->SampleDirection(reverse_incident_dir, sampler);
  }
};


class ShaderTests : public ScatterTests
{
private:
  const Shader *sh = nullptr;
  Double3 reverse_incident_dir;
  Double3 shading_normal;
  RaySurfaceIntersection intersection;
  PathContext context;
  std::unique_ptr<TexturedSmoothTriangle> triangle;
  
public:  
  void RunAllCalculations(const Shader &sh, const Double3 &reverse_incident_dir, const Double3 &shading_normal, int num_samples)
  {
    Init(sh, reverse_incident_dir, shading_normal);
    ScatterTests::RunAllCalculations(num_samples);
  }
  
private:
  void Init(const Shader &sh, const Double3 &reverse_incident_dir, const Double3 &shading_normal)
  {
    this->sh = &sh;
    this->reverse_incident_dir = reverse_incident_dir;
    Double3 normal{0, 0, 1};
    triangle = std::make_unique<TexturedSmoothTriangle>(
      Double3{ -1, -1, 0}, Double3{ 1, -1, 0 }, Double3{ 0, 1, 0 },
      shading_normal, shading_normal, shading_normal,
      Double3{0, 0, 0}, Double3{1, 0, 0}, Double3{0.5, 1, 0});
    context = PathContext{Color::LambdaIdxClosestToRGBPrimaries()};
    intersection = MakeIntersection(reverse_incident_dir);
  }
  
  RaySurfaceIntersection MakeIntersection(Double3 reverse_incident_dir)
  {
    RaySegment seg{{reverse_incident_dir, -reverse_incident_dir}, LargeNumber};
    HitId hit;
    bool ok = triangle->Intersect(seg.ray, seg.length, hit);
    return RaySurfaceIntersection{hit, seg};
  }
  
  double ProbabilityDensity(const Double3 &omega) const override
  {
    double ret = NaN;
    sh->EvaluateBSDF(reverse_incident_dir, intersection, omega, context, &ret);
    return ret;
  }
  
  Spectral3 ScatterFunction(const Double3 &omega) const override
  {
    Spectral3 value = sh->EvaluateBSDF(reverse_incident_dir, intersection, omega, context, nullptr);
    value *= std::abs(Dot(omega, intersection.shading_normal));
    return value;
  }
  
  ScatterSample MakeScatterSample() override
  {
    auto smpl = sh->SampleBSDF(reverse_incident_dir, intersection, sampler, context);
    if (Dot(smpl.coordinates, intersection.normal)<0.)
      EXPECT_TRUE(smpl.value.isZero());
    smpl.value *= std::abs(Dot(smpl.coordinates, intersection.shading_normal));
    return smpl;
  }
  
  void CheckSymmetry() override
  {    
    CheckSymmetryImpl([&](const Double3 &wi, const Double3 &wo) -> std::pair<double, Spectral3>
    {
      // Making a new intersection is needed because the shader expects the first direction to be aligned with the normals.
      RaySurfaceIntersection intersection = MakeIntersection(wi);
      Spectral3 ret = sh->EvaluateBSDF(wi, intersection, wo, context, nullptr);
      // Shader pdf's are in general not symmetric. Therefore I just return 0 here.
      return std::make_pair(0., ret);
    });
  }
};




TEST_F(PhasefunctionTests, Uniform)
{
  PhaseFunctions::Uniform pf;
  this->RunAllCalculations(pf, Double3{0,0,1}, 20000);
  this->TestCountsDeviationFromSigma(3.5);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001);
}


TEST_F(PhasefunctionTests, Rayleigh)
{
  // output.open("PhasefunctionTests.Rayleight.txt");
  PhaseFunctions::Rayleigh pf;
  this->RunAllCalculations(pf, Double3{0,0,1}, 20000);
  this->TestCountsDeviationFromSigma(3.5);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001);
}


TEST_F(PhasefunctionTests, HenleyGreenstein)
{
  //output.open("PhasefunctionTests.HenleyGreenstein.txt");
  PhaseFunctions::HenleyGreenstein pf(0.4);
  this->RunAllCalculations(pf, Double3{0,0,1}, 20000);
  this->TestCountsDeviationFromSigma(4.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001);
}


TEST_F(PhasefunctionTests, Combined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::Combined pf{{1., 1., 1.}, {.1, .2, .3}, pf1, {.3, .4, .5}, pf2};
  this->RunAllCalculations(pf, Double3{0,0,1}, 20000);
  this->TestCountsDeviationFromSigma(3.5);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001);
}


TEST_F(PhasefunctionTests, SimpleCombined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::SimpleCombined pf{{.1, .2, .3}, pf1, {.3, .4, .5}, pf2};
  this->RunAllCalculations(pf, Double3{0,0,1}, 20000);
  this->TestCountsDeviationFromSigma(3.5);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001);
}



TEST_F(ShaderTests, DiffuseShader1)
{
  auto reversed_incident_dir = Normalized(Double3{0,0,1});
  auto shading_normal = Double3{0,0,1};
  DiffuseShader sh(Color::SpectralN{0.5}, nullptr);
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 2000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001, Spectral3{0.5});
}

TEST_F(ShaderTests, DiffuseShader2)
{
  auto reversed_incident_dir = Normalized(Double3{0,1,1});
  auto shading_normal = Double3{0,0,1};
  DiffuseShader sh(Color::SpectralN{0.5}, nullptr);
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 2000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001, Spectral3{0.5});
}

TEST_F(ShaderTests, DiffuseShader3)
{
  // NOTE: High variant because the pdf distributes samples according to geometric normal. Hence many samples needed for integral to converge!
  auto reversed_incident_dir = Normalized(Double3{0,0,1});
  auto shading_normal = Normalized(Double3{0,10,1});
  DiffuseShader sh(Color::SpectralN{0.5}, nullptr);
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 50000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001, Spectral3{0.5});
}


TEST_F(ShaderTests, DiffuseShader4)
{
  auto reversed_incident_dir = Normalized(Double3{0,10,1});
  auto shading_normal = Normalized(Double3{0,1,1});
  DiffuseShader sh(Color::SpectralN{0.5}, nullptr);
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 50000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001, Spectral3{0.5});
}


TEST_F(ShaderTests, SpecularReflectiveShader1)
{
  auto reversed_incident_dir = Normalized(Double3{0,0,1});
  auto shading_normal = Double3{0,0,1};
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 2000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001, Spectral3{0.5});
}

TEST_F(ShaderTests, SpecularReflectiveShader2)
{
  auto reversed_incident_dir = Normalized(Double3{0,1,1});
  auto shading_normal = Double3{0,0,1};
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 2000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001, Spectral3{0.5});
}

TEST_F(ShaderTests, SpecularReflectiveShader3)
{
  auto reversed_incident_dir = Normalized(Double3{0,0,1});
  auto shading_normal = Normalized(Double3{0,10,1});
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 5000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  // This is a pathological case, where due to the strongly perturbed
  // shading normal, the incident direction is specularly reflected
  // below the geometric surface. I don't know what I can do but to
  // assign zero reflected radiance. Thus sane rendering algorithms
  // would not consider this path any further.
  this->CheckIntegral(0.001, Spectral3{0.0});
}


TEST_F(ShaderTests, SpecularReflectiveShader4)
{
  auto reversed_incident_dir = Normalized(Double3{0,10,1});
  auto shading_normal = Normalized(Double3{0,1,1});
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  this->RunAllCalculations(sh, reversed_incident_dir, shading_normal, 5000);
  this->TestCountsDeviationFromSigma(3.);
  this->TestChiSqr(0.05);
  this->CheckIntegral(0.001, Spectral3{0.5});
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

