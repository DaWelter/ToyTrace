#include "gtest/gtest.h"
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>

#include <boost/container/small_vector.hpp>

#ifdef HAVE_JSON
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#endif

#include "cubature_wrapper.hxx"
#include "cubemap.hxx"
#include "tests_stats.hxx"

#include "spectral.hxx"
#include "sampler.hxx"
#include "ray.hxx"
#include "sampler.hxx"
#include "primitive.hxx"
#include "util.hxx"
#include "shader.hxx"
#include "sampler.hxx"
#include "embreeaccelerator.hxx"
#include "cubature.h"

using namespace materials;


inline void ExpectNear(const Spectral3 &a, const Spectral3 &b, double tol)
{
  EXPECT_NEAR(a[0], b[0], 1.e-6);
  EXPECT_NEAR(a[1], b[1], 1.e-6);
  EXPECT_NEAR(a[2], b[2], 1.e-6);
}



#ifdef HAVE_JSON
namespace rj
{

using namespace rapidjson;

using Alloc = rapidjson::Document::AllocatorType; // For some reason I must supply an "allocator" almost everywhere. Why? Who knows?!

rapidjson::Value Array3ToJSON(const Eigen::Array<double, 3, 1> &v, Alloc &alloc)
{
  rj::Value json_vec(rj::kArrayType);
  json_vec.PushBack(rj::Value(v[0]).Move(), alloc).
            PushBack(rj::Value(v[1]).Move(), alloc).
            PushBack(rj::Value(v[2]).Move(), alloc);
  return json_vec;
}


rapidjson::Value ScatterSampleToJSON(const ScatterSample &smpl, Alloc &alloc)
{
  rj::Value json_smpl(rj::kObjectType);
  json_smpl.AddMember("pdf", (double)smpl.pdf_or_pmf, alloc);
  rj::Value json_vec = Array3ToJSON(smpl.value, alloc);
  json_smpl.AddMember("value", json_vec, alloc);
  json_vec = Array3ToJSON(smpl.coordinates.array(), alloc);
  json_smpl.AddMember("coord", json_vec, alloc);
  return json_smpl;
}


void Write(rj::Document &doc, const std::string &filename)
{
  rj::StringBuffer buffer;
  rj::PrettyWriter<rj::StringBuffer> writer(buffer);
  doc.Accept(writer);
  
  std::ofstream os(filename.c_str());
  os.write(buffer.GetString(), std::strlen(buffer.GetString()));
}

}
#endif



TEST(OnlineVariance, ArrayAccumulator)
{
  static constexpr int Narms = 3;
  static constexpr int Nsamples = 100;
  std::mt19937 gen;
  OnlineVariance::ArrayAccumulator<double> accum1(Narms);
  OnlineVariance::ArrayAccumulator<double> accum2(Narms);
  
  const double means[Narms] = {
    10., 20., 30.
  };
  const double stddev[Narms] = {
    1., 33., 7.
  };
  boost::container::small_vector<std::normal_distribution<double>, Narms> dist;
  for (int i = 0; i < Narms; ++i)
    dist.emplace_back(means[i], stddev[i]);

  for (int isample = 0; isample < Nsamples; ++isample)
  {
    for (int iarm = 0; iarm < Narms; ++iarm)
    {
      accum1.Add(iarm, dist[iarm](gen));
      accum2.Add(iarm, dist[iarm](gen));
    }
  }

  accum2.Add(accum1);

  auto Check = [means,stddev](OnlineVariance::ArrayAccumulator<double> &accum)
  {
    auto sample_means = accum.Mean();
    auto sample_var = accum.Var();
    auto sample_stddev = sample_var.sqrt().eval();
    for (int i = 0; i < accum.Size(); ++i)
    {
      // Allow at most two standard deviations of error
      double mean_stddev = StddevOfAverage(sample_means[i], Nsamples);
      EXPECT_LT(std::abs(sample_means[i] - means[i]), mean_stddev*2);
      double stddev_stddev = StddevOfStddev(sample_stddev[i], Nsamples);
      EXPECT_LE(std::abs(sample_stddev[i] - stddev[i]), stddev_stddev*2);
    }
  };

  Check(accum1);
  Check(accum2);
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
  Mesh triangle{0,0};
  AppendSingleTriangle(triangle,
    (offset).cast<float>(),
    (offset+X).cast<float>(),
    (offset+Y).cast<float>(),
    Float3{0.f,0.f,0.f});
  triangle.MakeFlatNormals();
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
    HitId hit = triangle.SampleUniformPosition(0, sampler);
    SurfaceInteraction si{hit};
    Double3 local = minv * (si.pos - offset);
    EXPECT_NEAR(local[2], 0., 1.e-3);
    EXPECT_GE(local[0], 0.); EXPECT_LE(local[0], 1.);
    EXPECT_GE(local[1], 0.); EXPECT_LE(local[1], 1.);
    EXPECT_LE(local[0] + local[1], 1.);
  }
}


TEST_F(RandomSamplingFixture, Sphere3dSampling)
{
  static constexpr int N = 100;
  static constexpr int NUM_REGIONS=10;
  int n_samples[NUM_REGIONS] = {};
  const double center_sphere_radius = 0.5;
  const double center_area = UnitSphereVolume*std::pow(center_sphere_radius, 3.);
  const double outer_octant_area = (Pi - center_area) / 8.;
  const double region_area[NUM_REGIONS] = 
  {
    center_area,
    outer_octant_area, outer_octant_area,
    outer_octant_area, outer_octant_area,
    outer_octant_area, outer_octant_area,
    outer_octant_area, outer_octant_area,
    0 // This bin collects points which fall outside the sphere.
  };
  for (int i=0; i<N; ++i)
  {
    Double3 rvs = { sampler.Uniform01(), sampler.Uniform01(), sampler.Uniform01() };
    Double3 v = SampleTrafo::ToUniformSphere3d(rvs);
    int region = 9;
    const auto r = Length(v);
    if (r<center_sphere_radius)
      region = 0;
    else if (r<=1.)
    {
      if (v[0] > 0)
      {
        if (v[1] > 0)
          region = v[2] > 0 ? 1 : 2;
        else
          region = v[2] > 0 ? 3 : 4;
      }
      else
      {
        if (v[1] > 0)
          region = v[2] > 0 ? 5 : 6;
        else
          region = v[2] > 0 ? 7 : 8;
      }
    }
    ++n_samples[region];
  }
  for (int bin=0; bin<NUM_REGIONS; ++bin)
  {
    CheckNumberOfSamplesInBin("sphere_bin", n_samples[bin], N, region_area[bin]/Pi);
  }
}




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
  
  Eigen::Array2d result = Integral2D(
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


class Scatterer
{
public:
  virtual ~Scatterer() {};
  virtual double ProbabilityDensity(const Double3 &wi, const Double3 &wo) const = 0;
  virtual Spectral3 ScatterFunction(const Double3 &wi, const Double3 &wo) const = 0;
  virtual ScatterSample MakeScatterSample(const Double3 &wi, Sampler &sampler) const = 0;
  virtual void SetAdjoint(bool use_adjoint) = 0;
  virtual double SurfaceNormalCosineOrOne(const Double3 &w, bool use_shading_normal) const = 0;
};


class SamplingConsistencyTest
{
  /* I want to check if the sampling frequency of some solid angle areas match the probabilities
    * returned by the Evaluate function. I further want to validate the normalization constraint. */
protected:
  struct DeltaPeak
  {
    double pr; // Probability
    int sample_count; // Fallen into this bin.
    Double3 direction; // Where the peak is.
  };
  
  CubeMap cubemap;
  static constexpr int Nbins = 4;
  Double3 reverse_incident_dir {NaN};
  const Scatterer &scatterer;
  
  // Filled by member functions.
  std::vector<int> bin_sample_count;
  std::vector<double> bin_probabilities;
  std::vector<double> bin_probabilities_error; // Numerical integration error.
  std::vector<DeltaPeak> delta_peaks;
  double total_probability;
  double total_probability_error; // Numerical integration error.
  double probability_of_delta_peaks;
  OnlineVariance::Accumulator<Spectral3> diffuse_scattered_estimate;
  OnlineVariance::Accumulator<Spectral3> total_scattered_estimate;
  Spectral3 integral_cubature;
  Spectral3 integral_cubature_error; // Numerical integration error.
  int num_samples{0};
  
public:
    Sampler sampler;
  
public:
  SamplingConsistencyTest(const Scatterer &_scatterer, const Double3 &_reverse_incident_dir)
    : cubemap{Nbins}, reverse_incident_dir{_reverse_incident_dir}, scatterer{_scatterer}
  {
  }
  
  void TestCountsDeviationFromSigma(double number_of_sigmas_threshold = 3.)
  {
    for (int idx=0; idx<cubemap.TotalNumBins(); ++idx)
    {
      int side, i, j;
      std::tie(side, i, j) = cubemap.IndexToCell(idx);
      CheckNumberOfSamplesInBin(
        strconcat(side,"[",i,",",j,"]").c_str(), 
        bin_sample_count[idx], 
        num_samples, 
        bin_probabilities[idx],
        number_of_sigmas_threshold,
        bin_probabilities_error[idx]);
    }
    for (auto &p : delta_peaks)
    {
      CheckNumberOfSamplesInBin(
        strconcat("peak ", p.direction).c_str(),
        p.sample_count,
        num_samples,
        p.pr,
        number_of_sigmas_threshold,
        0.);
    }
  }
  

  void TestChiSqr(double p_threshold = 0.05)
  {
    std::vector<int> sample_count = bin_sample_count;
    std::vector<double> probabilities = bin_probabilities;
    for (const auto &p : delta_peaks)
    {
      sample_count.push_back(p.sample_count);
      probabilities.push_back(p.pr);
    }
    double chi_sqr_probability = ChiSquaredProbability(bin_sample_count.data(), bin_probabilities.data(), isize(bin_probabilities));
    EXPECT_GE(chi_sqr_probability, p_threshold);
  }
  
  
  void CheckIntegral(boost::optional<Spectral3> by_value = boost::none, double p_value = 0.05)
  {
    static constexpr double TEST_EPS = 1.e-8; // Extra wiggle room for roundoff errors.
    for (int i=0; i<integral_cubature.size(); ++i)
    {
      // As a user of the test, I probably only know the combined albedo of 
      // specular and diffuse/glossy parts. So take that for comparison.
      TestSampleAverage(diffuse_scattered_estimate.Mean()[i], diffuse_scattered_estimate.Stddev()[i], num_samples, integral_cubature[i], integral_cubature_error[i]+TEST_EPS, p_value);
      if (by_value)
      {
        TestSampleAverage(total_scattered_estimate.Mean()[i], total_scattered_estimate.Stddev()[i], num_samples, (*by_value)[i], TEST_EPS, p_value);
      }
      TestProbabilityOfMeanLowerThanUpperBound(total_scattered_estimate.Mean()[i], total_scattered_estimate.Stddev()[i], num_samples, 1.+TEST_EPS, p_value);
    }
    CheckTotalProbability();
  }
 
  void CheckTotalProbability()
  {
    static constexpr double TEST_EPS = 1.e-8; // Extra wiggle room for roundoff errors.
    EXPECT_NEAR(total_probability+probability_of_delta_peaks, 1., total_probability_error+TEST_EPS);
  }
 
  void DoSampling(int num_samples)
  {
    this->num_samples = num_samples;
    this->bin_sample_count = std::vector<int>(cubemap.TotalNumBins(), 0);
    this->delta_peaks = decltype(delta_peaks){};
    this->diffuse_scattered_estimate = OnlineVariance::Accumulator<Spectral3>{};  
    this->total_scattered_estimate = OnlineVariance::Accumulator<Spectral3>{};
    this->probability_of_delta_peaks = 0.;
    auto RegisterDeltaSample = [this](const Double3 &v, double pr)
    {
      auto it = std::find_if(delta_peaks.begin(), delta_peaks.end(), [&](const DeltaPeak &pk) { return v.cwiseEqual(pk.direction).all(); });
      if (it == delta_peaks.end())
      {
        delta_peaks.push_back(DeltaPeak{pr, 1, v});
        probability_of_delta_peaks += pr;
      }
      else
      {
        EXPECT_EQ(pr, it->pr);
        ++it->sample_count;
      }
    };
    auto SampleIndex = [this](const ScatterSample &smpl) ->  int
    {
      int side, i, j;
      std::tie(side, i, j) = cubemap.OmegaToCell(smpl.coordinates);
      return cubemap.CellToIndex(side, i, j);
    };    
    for (int snum = 0; snum<num_samples; ++snum)
    {
      auto smpl = scatterer.MakeScatterSample(reverse_incident_dir, sampler);
      Spectral3 contribution = scatterer.SurfaceNormalCosineOrOne(smpl.coordinates, false) 
                               / smpl.pdf_or_pmf * smpl.value;
      // Can only integrate the non-delta density. 
      // Keep track of delta-peaks separately.
      if (IsFromPdf(smpl))
      {
        int idx = SampleIndex(smpl);
        bin_sample_count[idx] += 1;
        diffuse_scattered_estimate += contribution;
      }
      else
      {
        RegisterDeltaSample(smpl.coordinates, smpl.pdf_or_pmf);
        // Think of Russian Roulette. When it is decided to sample the delta-peaks,
        // the other contribution is formally zerod out. In order to make the
        // statistics work, I must still count it as regular sample.
        diffuse_scattered_estimate += Spectral3{0.};
      }
      // I can ofc integrate delta-peak contributions by MC.
      total_scattered_estimate += contribution; 
    }
  }
 
 
  void ComputeBinProbabilities()
  {
    constexpr int MAX_NUM_FUNC_EVALS = 10000000;
    this->bin_probabilities = std::vector<double>(cubemap.TotalNumBins(), 0.);
    this->bin_probabilities_error = std::vector<double>(cubemap.TotalNumBins(), 0.);
    this->total_probability = 0.;
    this->total_probability_error = 0.;
    for (int idx=0; idx<cubemap.TotalNumBins(); ++idx)
    {
      int side, i, j;
      std::tie(side, i, j) = cubemap.IndexToCell(idx);
      // Integrating this gives me the probability of a sample falling
      // into the current bin. The integration is carried out on a square in
      // R^2, hence the scale factor J from the Jacobian of the variable transform.
      auto probabilityDensityTimesJ = [&](const Double2 x) -> double
      {
        Double3 omega = cubemap.UVToOmega(side, x);
        double pdf = scatterer.ProbabilityDensity(reverse_incident_dir, omega);
        double ret = pdf*cubemap.UVtoJ(x);
        assert(std::isfinite(ret));
        return ret;
      };
      // By design, only diffuse/glossy components will be captured by this.
      Double2 start, end;
      std::tie(start, end) = cubemap.CellToUVBounds(i, j);       
      double err = NaN;
      double prob = Integral2D(probabilityDensityTimesJ, start, end, 1.e-4, 1.e-3, MAX_NUM_FUNC_EVALS, &err);
      bin_probabilities[idx] = prob;
      bin_probabilities_error[idx] = err;
      total_probability += bin_probabilities[idx];
      total_probability_error += err;
    }
  }
  
  
  // Check energy conservation by 
  // Theorem 6.2.  in Veach's thesis. It is valid for integration over the entire sphere 
  // of directions. If BSDF is only reflective, it must yield zero for directions that
  // are below the surface.
  void ComputeCubatureIntegral()
  {
    constexpr int MAX_NUM_FUNC_EVALS = 100000;
    this->integral_cubature = 0.;
    this->integral_cubature_error = 0.;
    for (int idx=0; idx<cubemap.TotalNumBins(); ++idx)
    {
      int side, i, j;
      std::tie(side, i, j) = cubemap.IndexToCell(idx);
      auto functionValueTimesJ = [&](const Double2 x) -> Eigen::Array3d
      {
        Double3 omega = cubemap.UVToOmega(side, x);
        Spectral3 val = scatterer.ScatterFunction(reverse_incident_dir, omega);
                  val *= scatterer.SurfaceNormalCosineOrOne(omega, false);
        Spectral3 ret = val*cubemap.UVtoJ(x);
        assert(ret.allFinite());
        return ret;
      };
      Double2 start, end;
      std::tie(start, end) = cubemap.CellToUVBounds(i, j);       
      Eigen::Array3d err{NaN};
      Eigen::Array3d cell_integral = Integral2D(functionValueTimesJ, start, end, 1.e-3, 1.e-2, MAX_NUM_FUNC_EVALS, &err);
      this->integral_cubature += cell_integral;
      this->integral_cubature_error += err;
    }
  }
  
#ifdef HAVE_JSON
  rj::Value CubemapToJSON(rj::Alloc &alloc);
#endif
  void DumpVisualization(const std::string filename);
};


// This test basically amounts to evaluating the scatter function with
// swapped argument and looking if the result equals the evaluation of
// the adjoint.
class PointwiseSymmetryTest
{
    ScatterSample CheckedSample(const Double3 &wi, bool sample_adjoint)
    {
      const double eta_i = Eta(wi);
      scatterer->SetAdjoint(sample_adjoint);
      ScatterSample smpl = scatterer->MakeScatterSample(wi, sampler);
      if (!smpl.pdf_or_pmf.IsFromDelta())
      {
        // Check reversed arguments
        scatterer->SetAdjoint(!sample_adjoint);
        Spectral3 f = scatterer->ScatterFunction(smpl.coordinates, wi);
        constexpr double TOL = 1.e-6;
        const double eta_o = Eta(smpl.coordinates);
        ExpectNear(smpl.value*eta_i, f*eta_o, TOL);
      }
      return smpl;
    }
  
    inline double Eta(const Double3 &w)
    {
      return (Dot(w, n) > 0.) ? eta_above : eta_below;
    }
  
  protected:
    Scatterer *scatterer {nullptr};
    int num_samples;
    Sampler sampler;
    double eta_above;
    double eta_below;
    inline static const Double3 n{0,0,1};
    
  public:
    PointwiseSymmetryTest(Scatterer &_scatterer, int _num_samples, double eta_above, double eta_below, std::uint64_t seed) :
      scatterer{&_scatterer}, num_samples{_num_samples}, eta_above{eta_above}, eta_below{eta_below}
    {
      sampler.Seed(seed);
    }
    
    void Run()
    {
      for (int snum = 0; snum<num_samples; ++snum)
      {
        Double3 wi = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
        Double3 wo = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
        const double eta_i = Eta(wi);
        const double eta_o = Eta(wo);
        scatterer->SetAdjoint(false);
        Spectral3 f_val = scatterer->ScatterFunction(wi, wo);
        scatterer->SetAdjoint(true);
        Spectral3 f_rev = scatterer->ScatterFunction(wo, wi);
        ExpectNear(f_val*eta_i, f_rev*eta_o, 1.e-6);
      }
      
      for (int snum = 0; snum<num_samples; ++snum)
      {
        Double3 wi = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
        CheckedSample(wi, true);
        CheckedSample(wi, false);
      }
    }
};


// This test computes the double integral 
// int_wi int_wo g(wi, wo)*|wi.n|*|wo.n| dwi dwo,
// two times. First using normal scatter function (g = f),
// and secondly using the adjoint (g = f*). Math sais
// that the results should be equal! (Because, we can
// swap the order of integration and rename the integration
// variables).
// This test is useful for scatter functions with Dirac-deltas
// because for the delta peaks, the only operation we can do
// to get some values is sampling.
class IntegralSymmetryTest
{
  protected:
    // parameters
    Scatterer *scatterer {nullptr};
    int num_samples;
    Sampler sampler;    
  public:
    IntegralSymmetryTest(Scatterer &_scatterer, int _num_samples, std::uint64_t seed) :
      scatterer{&_scatterer}, num_samples{_num_samples}
    {
      sampler.Seed(seed);
    }
    
    void Run()
    {
      OnlineVariance::Accumulator<Spectral3> total_albedo_adjoint;
      OnlineVariance::Accumulator<Spectral3> total_albedo;
      
      for (int snum = 0; snum<num_samples; ++snum)
      {
        Double3 wi = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
        scatterer->SetAdjoint(true);
        ScatterSample smpl_adj = scatterer->MakeScatterSample(wi, sampler);
        scatterer->SetAdjoint(false);
        ScatterSample smpl_non = scatterer->MakeScatterSample(wi, sampler);
        double cos_wi = scatterer->SurfaceNormalCosineOrOne(wi, false);
        double cos_wo_adj = scatterer->SurfaceNormalCosineOrOne(smpl_adj.coordinates, false);
        double cos_wo_non = scatterer->SurfaceNormalCosineOrOne(smpl_non.coordinates, false);
        total_albedo_adjoint += cos_wo_adj*cos_wi/PmfOrPdfValue(smpl_adj) * smpl_adj.value;
        total_albedo += cos_wo_non*cos_wi/PmfOrPdfValue(smpl_non) * smpl_non.value;
      }
      
      Spectral3 error = (total_albedo.Mean() - total_albedo_adjoint.Mean()).abs();
      Spectral3 stddev = (total_albedo.Stddev() + total_albedo_adjoint.Stddev());
      static constexpr double TEST_EPS = 1.e-6;
      for (int i=0; i<error.size(); ++i)
        TestSampleAverage(error[i], stddev[i], num_samples, 0., TEST_EPS, 0.05);
    }
};





class PhasefunctionScatterer : public Scatterer
{
  const PhaseFunctions::PhaseFunction &pf;
public:
  PhasefunctionScatterer(const PhaseFunctions::PhaseFunction &_pf)
    : pf{_pf}
  {
  }
  
  double ProbabilityDensity(const Double3 &wi, const Double3 &wo) const override
  {
    double ret;
    pf.Evaluate(wi, wo, &ret);
    return ret;
  }

  Spectral3 ScatterFunction(const Double3 &wi, const Double3 &wo) const override
  {
    return pf.Evaluate(wi, wo, nullptr);
  }
  
  ScatterSample MakeScatterSample(const Double3 &wi, Sampler &sampler) const override
  {
    return pf.SampleDirection(wi, sampler);
  }
  
  void SetAdjoint(bool ) override 
  {
    // Is self adjoint.
  }
  
  double SurfaceNormalCosineOrOne(const Double3 &w, bool use_shading_normal) const override
  {
    return 1;
  }
};


class ShaderScatterer : public Scatterer
{
  std::unique_ptr<Shader> shader;
  PathContext context;
  Mesh mesh;
  mutable SurfaceInteraction last_intersection;
  mutable Double3 last_incident_dir;
  double ior_below_over_above; // Ratio
  EmbreeAccelerator embree;
  
  SurfaceInteraction& MakeIntersection(const Double3 &reverse_incident_dir) const
  {
    if (last_incident_dir != reverse_incident_dir)
    {
      RaySegment seg{{reverse_incident_dir, -reverse_incident_dir}, LargeNumber};
      bool ok = embree.FirstIntersection(seg.ray, 0., seg.length, last_intersection);
      last_incident_dir = reverse_incident_dir;
    }
    return last_intersection;
  }
  
  Spectral3 UndoneIorScaling(Spectral3 value, const Double3 &wi, const SurfaceInteraction &intersection, const Double3 &wo) const
  {
    if (Dot(wo, intersection.normal) < 0. && context.transport == TransportType::RADIANCE) // We have transmission
    {
      // According to Veach's Thesis (Eq. 5.12), transmitted radiance is scaled by (eta_t/eta_i)^2.
      // Here I undo the scaling factor, so that when summing, the total scattered radiance equals
      // the incident radiance. And that conservation is much simpler to check for.
      if (Dot(wi, intersection.geometry_normal) < 0.) // Light goes from below surface to above. Because wi, wo denote direction of randomwalk.
        value *= Sqr(ior_below_over_above);
      else
        value *= Sqr(1./ior_below_over_above);
    }
    return value;
  }
  
  Spectral3 CosThetaTerm(Spectral3 value, const Double3 &wi, const SurfaceInteraction &intersection, const Double3 &wo) const
  {
    // Here, the normal correction is applied. See Veach, p.g. 153.
    if (context.transport==RADIANCE) // light comes from wo
    {
      value *= std::abs(Dot(wo, intersection.shading_normal))/(std::abs(Dot(wo, intersection.normal))+Epsilon);
    }
    else // Light goes from wi to wo.
    {
      value *= std::abs(Dot(wi, intersection.shading_normal))/(std::abs(Dot(wi, intersection.normal))+Epsilon);
    }
    return value;
  }
    
public:
  ShaderScatterer(std::unique_ptr<Shader> _shader)
    : shader{std::move(_shader)},
      mesh{0,0},
      last_incident_dir{NaN},
      ior_below_over_above{1.}
  {
    AppendSingleTriangle(mesh, 
      {-1, -1, 0}, {1, -1, 0}, {0, 1, 0}, {0, 0, 1});
    SetShadingNormal({0,0,1});
    context = PathContext{SelectRgbPrimaryWavelengths()};
    embree.InsertRefTo(mesh);
    embree.Build();
  }
  
  void SetIorRatio(double _ior_below_over_above)
  {
    this->ior_below_over_above = _ior_below_over_above;
  }
    
  void SetShadingNormal(const Double3 &shading_normal)
  {    
    Float3 n =  shading_normal.cast<float>();
    mesh.normals.row(0) = n;
    mesh.normals.row(1) = n;
    mesh.normals.row(2) = n;
    last_incident_dir = Double3{NaN}; // Force regeneration of intersection!
  }
  
  double ProbabilityDensity(const Double3 &wi, const Double3 &wo) const override
  {
    double ret = NaN;
    auto &intersection = MakeIntersection(wi);
    shader->EvaluateBSDF(wi, intersection, wo, context, &ret);
    return ret;
  }
  
  Spectral3 ScatterFunction(const Double3 &wi, const Double3 &wo) const override
  {
    auto &intersection = MakeIntersection(wi);
    auto value = shader->EvaluateBSDF(wi, intersection, wo, context, nullptr);
    value = CosThetaTerm(value, wi, intersection, wo);
    //value = UndoneIorScaling(value, wi, intersection, wo);
    return value;
  }
  
  ScatterSample MakeScatterSample(const Double3 &wi, Sampler &sampler) const override
  {
    auto &intersection = MakeIntersection(wi);
    auto smpl = shader->SampleBSDF(wi, intersection, sampler, context);
    smpl.value= CosThetaTerm(smpl.value,wi, intersection, smpl.coordinates);
    //smpl.value = UndoneIorScaling(smpl.value, wi, intersection, smpl.coordinates);
    return smpl;
  }
  
  void SetAdjoint(bool to_adjoint) override
  {
    context.transport = to_adjoint ? TransportType::IMPORTANCE : TransportType::RADIANCE;
  }
  
  double SurfaceNormalCosineOrOne(const Double3 &w, bool use_shading_normal) const override
  {
    return std::abs(Dot(w, use_shading_normal ? last_intersection.smooth_normal : last_intersection.geometry_normal));
  }
};



class PhaseFunctionTests : public SamplingConsistencyTest
{
    PhasefunctionScatterer pf_wrapper;
  public:
    PhaseFunctionTests(const PhaseFunctions::PhaseFunction &_pf, const Double3 &reverse_incident_dir)
      :  SamplingConsistencyTest{pf_wrapper, reverse_incident_dir}, pf_wrapper{_pf}
    {
    }
    
    void RunAllCalculations(int num_samples)
    {
      DoSampling(num_samples);
      ComputeBinProbabilities();
      ComputeCubatureIntegral();
    }
};



#ifdef HAVE_JSON
class ScatterVisualization
{
  const Scatterer &scatterer;
  rj::Alloc &alloc;
  
  CubeMap cubemap;
  Sampler sampler;
  static constexpr int Nbins = 64;
  static constexpr int Nsamples = 10000;
  
public:  
  ScatterVisualization(const Scatterer &_scatterer, rj::Alloc &alloc)
    : scatterer{_scatterer}, alloc{alloc}, cubemap{Nbins}
  {
  } 
  
  rj::Value RecordSamples(const Double3 &reverse_incident_dir)
  {
    rj::Value json_samples(rj::kArrayType);
    for (int snum = 0; snum<Nsamples; ++snum)
    {
      auto smpl = scatterer.MakeScatterSample(reverse_incident_dir, sampler);
      double cosn = scatterer.SurfaceNormalCosineOrOne(smpl.coordinates, false);
      rj::Value json_smpl = rj::ScatterSampleToJSON(smpl, alloc);
      json_smpl.AddMember("cosn", cosn, alloc);
      json_smpl.AddMember("weight", rj::Array3ToJSON((smpl.value * cosn / smpl.pdf_or_pmf).array(),alloc), alloc);
      json_samples.PushBack(json_smpl, alloc);
    }
    //doc.AddMember("samples", json_samples, alloc);
    return json_samples;
  }
  
  rj::Value RecordCubemap(const Double3 &reverse_incident_dir)
  {
    rj::Value json_side(rj::kArrayType);
    for (int side = 0; side < 6; ++side)
    {
      rj::Value json_i(rj::kArrayType);
      for (int i = 0; i < Nbins; ++i)
      {
        rj::Value json_j(rj::kArrayType);
        for (int j = 0; j < Nbins; ++j)
        {
          rj::Value json_bin(rj::kObjectType);

          int idx = cubemap.CellToIndex(side, i, j);
          auto bounds = cubemap.CellToUVBounds(i, j);
          Double3 pos = cubemap.UVToOmega(side, 0.5*(std::get<0>(bounds)+std::get<1>(bounds)));
          
          json_bin.AddMember("pos", rj::Array3ToJSON(pos.array(), alloc), alloc);
          
          auto integrand = [&](const Double2 x) -> Eigen::Array<double, 5, 1>
          {
            Double3 omega = cubemap.UVToOmega(side, x);
            Spectral3 val = scatterer.ScatterFunction(reverse_incident_dir, omega);
            double pdf = scatterer.ProbabilityDensity(reverse_incident_dir, omega);
            Eigen::Array<double, 5, 1> out; 
            out << val[0], val[1], val[2], pdf, 1.;
            return out*cubemap.UVtoJ(x);
          };
          try 
          {
            Eigen::Array<double, 5, 1> integral = Integral2D(integrand, std::get<0>(bounds), std::get<1>(bounds), 1.e-3, 1.e-2, 1000);
            double area = integral[4];
            Eigen::Array3d val_avg = integral.head<3>() / area;
            double pdf_avg = integral[3] / area;
            json_bin.AddMember("val", rj::Array3ToJSON(val_avg, alloc), alloc);
            json_bin.AddMember("pdf", pdf_avg, alloc);
          }
          catch (std::runtime_error)
          {
            rj::Value rjarr(rj::kArrayType);
            rjarr.PushBack("NaN", alloc).PushBack("NaN", alloc).PushBack("NaN", alloc);
            json_bin.AddMember("val", rjarr, alloc);
            json_bin.AddMember("pdf", "NaN", alloc);
          }
          json_j.PushBack(json_bin, alloc);
        }
        json_i.PushBack(json_j, alloc);
      }
      json_side.PushBack(json_i, alloc);
    }
    return json_side;
  }
};


rj::Value SamplingConsistencyTest::CubemapToJSON(rj::Alloc& alloc)
{
  rj::Value json_side;
  json_side.SetArray();
  for (int side = 0; side < 6; ++side)
  {
    rj::Value json_i(rj::kArrayType);
    for (int i = 0; i < Nbins; ++i)
    {
      rj::Value json_j(rj::kArrayType);
      for (int j = 0; j < Nbins; ++j)
      {
        rj::Value json_bin(rj::kObjectType);
        int idx = cubemap.CellToIndex(side, i, j);
        json_bin.AddMember("prob", bin_probabilities[idx], alloc);
        json_bin.AddMember("cnt", bin_sample_count[idx], alloc);
        json_j.PushBack(json_bin, alloc);
      }
      json_i.PushBack(json_j, alloc);
    }
    json_side.PushBack(json_i, alloc);
  }
  return json_side;
}


void SamplingConsistencyTest::DumpVisualization(const std::string filename)
{
  rj::Document doc;
  auto &alloc = doc.GetAllocator();
  doc.SetObject();
  
  rj::Value thing;
  thing = CubemapToJSON(alloc);
  doc.AddMember("test_cubemap", thing, alloc);
  ScatterVisualization vis(this->scatterer, alloc);
  thing = vis.RecordCubemap(this->reverse_incident_dir);
  doc.AddMember("vis_cubemap", thing, alloc);
  thing = vis.RecordSamples(reverse_incident_dir);
  doc.AddMember("vis_samples", thing, alloc);
  rj::Write(doc, filename);
}
#endif


TEST(PhasefunctionTests, Uniform)
{
  PhaseFunctions::Uniform pf{};
  PhaseFunctionTests test(pf, Double3{0,0,1});
  test.RunAllCalculations(5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, Rayleigh)
{
  PhaseFunctions::Rayleigh pf{};
  PhaseFunctionTests test(pf,Double3{0,0,1});
  test.RunAllCalculations(5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, HenleyGreenstein)
{
  PhaseFunctions::HenleyGreenstein pf(0.4);
  PhaseFunctionTests test(pf, Double3{0,0,1});
  test.RunAllCalculations(5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, Combined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::Combined pf(Spectral3{1., 1., 1.}, Spectral3{.1, .2, .3}, pf1, Spectral3{.3, .4, .5}, pf2);
  PhaseFunctionTests test(pf, Double3{0,0,1});
  test.RunAllCalculations(5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, SimpleCombined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::SimpleCombined pf{Spectral3{.1, .2, .3}, pf1, Spectral3{.3, .4, .5}, pf2};
  PhaseFunctionTests test(pf, Double3{0,0,1});
  test.sampler.Seed(12356);
  test.RunAllCalculations(10000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


namespace ParameterizedShaderTests
{
///////////////////////////////////////
static const Double3 VERTICAL = Double3{0,0,1};
static const Double3 EXACT_45DEG    = Normalized(Double3{0,1,1});
static const Double3 ALMOST_45DEG    = Normalized(Double3{0,1,1.1});
static const Double3 MUCH_DEFLECTED = Normalized(Double3{0,10,1});
static const Double3 UNUSED_VECTOR  = Double3{NaN};

///////////////////////////////////////
struct Params;

using Factory = std::function<Shader*()>; // The code does not work with a naked function pointer. Why? Idk!

struct Params
{
  Factory factory;
  Double3 reverse_incident_dir{NaN};
  Double3 normal{NaN};
  double ior{1.};
  int num_samples{10000};
  boost::optional<Spectral3> albedo{boost::none};
  bool test_sample_distribution{true};
  boost::optional<std::uint64_t> seed;

  Params(Factory f, const Double3 &wi, const Double3 &n, double ior = 1.) : 
    factory{f}, reverse_incident_dir{wi}, normal{n}, ior{ior}
    {}
  Params& NumSamples(int n) {
    num_samples = n;
    return *this;
  }
  Params& TestSampleDistribution(bool do_it) {
    test_sample_distribution = do_it;
    return *this;
  }
  Params& Albedo(boost::optional<Spectral3> albedo) {
    this->albedo = albedo;
    return *this;
  }
  Params& Seed(boost::optional<std::uint64_t> seed) {
    this->seed = seed;
    return *this;
  }
};


std::ostream& operator<<(std::ostream &os, const Params &p)
{
  os << p.reverse_incident_dir << ", " << p.normal
    << ", " << p.ior
    << ", " << p.num_samples;
  if (p.albedo)
  {
    auto tmp = *p.albedo;
    os << ", <" << tmp[0] << "," << tmp[1] << "," << tmp[2] << ">";
  }
  return os;
}


class ShaderSamplingConsistency : public testing::TestWithParam<Params>
{
    std::unique_ptr<ShaderScatterer> scatter;
    std::unique_ptr<SamplingConsistencyTest> test;
  protected:
    void SetUp() override
    {
      const Params& p = this->GetParam();
      {
        std::unique_ptr<Shader> sh(const_cast<Params&>(p).factory());
        this->scatter = std::make_unique<ShaderScatterer>(std::move(sh));
      }
      scatter->SetIorRatio(p.ior);
      scatter->SetShadingNormal(p.normal);
      test = std::make_unique<SamplingConsistencyTest>(*scatter, p.reverse_incident_dir);
      if (p.seed)
        test->sampler.Seed(*p.seed);
    }
    
    void Run(bool adjoint)
    {
      const Params& p = this->GetParam();
      scatter->SetAdjoint(adjoint);
      test->DoSampling(p.num_samples);
      test->ComputeBinProbabilities();
      //test->DumpVisualization("/tmp/debug2.json");
      if (p.test_sample_distribution)
      {
        test->TestCountsDeviationFromSigma(3.5);
        test->TestChiSqr();
      }
      if (adjoint)
        test->CheckTotalProbability();
      else
      {
        test->ComputeCubatureIntegral();
        test->CheckIntegral(p.albedo);
      }
    }
};


TEST_P(ShaderSamplingConsistency, NotAdjoint)
{
  this->Run(false);
}


TEST_P(ShaderSamplingConsistency, Adjoint)
{
  this->Run(true);
}


class ShaderSymmetry : public testing::TestWithParam<Params>
{
protected:
  std::unique_ptr<ShaderScatterer> scatter;
  
  void SetUp() override
  {
    const Params& p = this->GetParam();
    {
      std::unique_ptr<Shader> sh(const_cast<Params&>(p).factory());
      this->scatter = std::make_unique<ShaderScatterer>(std::move(sh));
    }
    scatter->SetIorRatio(p.ior);
    scatter->SetShadingNormal(p.normal);
  }
};


TEST_P(ShaderSymmetry, Pointwise)
{
  const Params& p = this->GetParam();
  PointwiseSymmetryTest test(*scatter, 100, 1.0/p.ior, p.ior, p.seed.value_or(Sampler::default_seed));
  test.Run();
}

TEST_P(ShaderSymmetry, Integral)
{
  const Params& p = this->GetParam();
  IntegralSymmetryTest test(*scatter, p.num_samples, p.seed.value_or(Sampler::default_seed));
  test.Run();
}



///////////////////////////////////////
namespace Diffuse
{

Params P(const Double3 &wi, const Double3 &n)
{
  auto f = []() { return new DiffuseShader(Color::SpectralN{0.5}, nullptr); };
  return Params{f, wi, n}.NumSamples(2000).Albedo(Spectral3{0.5});
}

// TODO: Fix for shading normals!
INSTANTIATE_TEST_CASE_P(Diffuse,
                        ShaderSamplingConsistency,
                        ::testing::Values(
                          P(VERTICAL,       VERTICAL),
                          P(EXACT_45DEG,          VERTICAL)
                          //P(VERTICAL,       MUCH_DEFLECTED),
                          //P(MUCH_DEFLECTED, EXACT_45DEG)
                        ));

INSTANTIATE_TEST_CASE_P(Diffuse,
                        ShaderSymmetry,
                        ::testing::Values(
                          P(UNUSED_VECTOR, VERTICAL)
                          //P(UNUSED_VECTOR, EXACT_45DEG),
                          //P(UNUSED_VECTOR, MUCH_DEFLECTED)
                        ));

} // namespace



namespace SpecularReflective
{

Params P(const Double3 &wi, const Double3 &n)
{
  auto f = []() { return new SpecularReflectiveShader(Color::SpectralN{0.5}); };
  return Params{f, wi, n}.NumSamples(100).Albedo(Spectral3{0.5}).TestSampleDistribution(false);
}
  
INSTANTIATE_TEST_CASE_P(SpecularReflective,
                        ShaderSamplingConsistency,
                        ::testing::Values(
                          P(VERTICAL,       VERTICAL),
                          P(EXACT_45DEG,          VERTICAL),
                          // This is a pathological case, where due to the strongly perturbed
                          // shading normal, the incident direction is specularly reflected
                          // below the geometric surface. I don't know what I can do but to
                          // assign zero reflected radiance. Thus sane rendering algorithms
                          // would not consider this path any further.
                          P(VERTICAL,       MUCH_DEFLECTED).Albedo(Spectral3{0.}),
                          P(MUCH_DEFLECTED, EXACT_45DEG)
                        ));

INSTANTIATE_TEST_CASE_P(SpecularReflective,
                        ShaderSymmetry,
                        ::testing::Values(
                          P(UNUSED_VECTOR, VERTICAL),
                          P(UNUSED_VECTOR, EXACT_45DEG),
                          P(UNUSED_VECTOR, MUCH_DEFLECTED)
                        ));

}



namespace SpecularDenseDielectric
{

// Test case for SpecularDenseDielectric selected with unnaturally
// high albedo of the diffuse layer. That is to uncover potential violation
// in energy conservation. (Too high re-emission might otherwise be masked
// by reflectivity factor.
Params P(const Double3 &wi, const Double3 &n)
{
  auto f = []() { return new SpecularDenseDielectricShader(0.2, Color::SpectralN{0.99}, nullptr); };
  return Params{f, wi, n};
}

// TODO: Fix for shading normals
INSTANTIATE_TEST_CASE_P(SpecularDenseDielectric,
                        ShaderSamplingConsistency,
                        ::testing::Values(
                          P(VERTICAL,       VERTICAL),
                          P(EXACT_45DEG,          VERTICAL)
                          //P(VERTICAL,       MUCH_DEFLECTED),
                          //P(MUCH_DEFLECTED, EXACT_45DEG)
                        ));

INSTANTIATE_TEST_CASE_P(SpecularDenseDielectric,
                        ShaderSymmetry,
                        ::testing::Values(
                          P(UNUSED_VECTOR, VERTICAL),
                          P(UNUSED_VECTOR, EXACT_45DEG),
                          P(UNUSED_VECTOR, MUCH_DEFLECTED)
                        ));

}


namespace Microfacet
{

Params P(const Double3 &wi, const Double3 &n)
{
  auto f = []() { return new MicrofacetShader(Color::SpectralN{0.3}, 0.4, nullptr); };
  return Params{f, wi, n}.NumSamples(10000);
}
  
INSTANTIATE_TEST_CASE_P(Microfacet,
                        ShaderSamplingConsistency,
                        ::testing::Values(
                          P(VERTICAL,       VERTICAL),
                          P(ALMOST_45DEG,            VERTICAL),
                          P(VERTICAL,       MUCH_DEFLECTED),
                          P(MUCH_DEFLECTED, EXACT_45DEG),
                          P(MUCH_DEFLECTED, VERTICAL)
                        ));

INSTANTIATE_TEST_CASE_P(Microfacet,
                        ShaderSymmetry,
                        ::testing::Values(
                          P(UNUSED_VECTOR, VERTICAL),
                          P(UNUSED_VECTOR, EXACT_45DEG)
                          //P(UNUSED_VECTOR, MUCH_DEFLECTED)
                        ));

}


namespace SpecularTransmissiveDielectric
{

Params P(const Double3 &wi, const Double3 &n)
{
  auto f = []() { return new SpecularTransmissiveDielectricShader(1.3); };
  return Params{f, wi, n, 1.3}.NumSamples(100).Albedo(Spectral3{1.}).TestSampleDistribution(false);
}


INSTANTIATE_TEST_CASE_P(SpecularTransmissiveDielectric,
                        ShaderSamplingConsistency,
                        ::testing::Values(
                          P(VERTICAL,       VERTICAL),
                          P(EXACT_45DEG,          VERTICAL),
                          P(MUCH_DEFLECTED, VERTICAL),
                          // No check for energy conservation, since shading normals break it. See Veach pg. 158, Sec 5.3.5
                                          
                          // If reflected, the exit direction is below the goemetrical surface, in case of which the contribution is canceled.
                          // However if transmitted, the exit direction is correctly below the geom. surface. So this makes a contribution to
                          // the integral check. But since the energy from reflection is missing, the total scattered radiance is hard to predict.
                          P(VERTICAL,       MUCH_DEFLECTED).Albedo(boost::none),
                          P(-VERTICAL,       VERTICAL),
                          P(-EXACT_45DEG,          VERTICAL),
                          P(-MUCH_DEFLECTED, VERTICAL),
                          P(-VERTICAL,       MUCH_DEFLECTED).Albedo(boost::none)
                        ));

INSTANTIATE_TEST_CASE_P(SpecularTransmissiveDielectric,
                        ShaderSymmetry,
                        ::testing::Values(
                          P(UNUSED_VECTOR, VERTICAL)
                          // Shading normals make the BSDF asymmetric and break energy conservation.
                          // So I probably cannot expect this test to succeed.
                          //P(UNUSED_VECTOR, EXACT_45DEG),
                          //P(UNUSED_VECTOR, MUCH_DEFLECTED)
                        ));

} // namespace



namespace GlossyTransmissiveDielectric
{

Params P(const Double3 &wi, const Double3 &n)
{
  auto f = []() { return new GlossyTransmissiveDielectricShader(1.3, 0.2, 0., nullptr); };
  return Params{f, wi, n, 1.3}.NumSamples(100).Albedo(Spectral3{1.}).TestSampleDistribution(false);
}


INSTANTIATE_TEST_CASE_P(GlossyTransmissiveDielectric,
                        ShaderSamplingConsistency,
                        ::testing::Values(
                          P(VERTICAL,       VERTICAL),
                          P(EXACT_45DEG,          VERTICAL),
                          P(MUCH_DEFLECTED, VERTICAL),
                          // No check for energy conservation, since shading normals break it. See Veach pg. 158, Sec 5.3.5
                                          
                          // If reflected, the exit direction is below the goemetrical surface, in case of which the contribution is canceled.
                          // However if transmitted, the exit direction is correctly below the geom. surface. So this makes a contribution to
                          // the integral check. But since the energy from reflection is missing, the total scattered radiance is hard to predict.
                          P(VERTICAL,       MUCH_DEFLECTED).Albedo(boost::none),
                          P(-VERTICAL,       VERTICAL),
                          P(-EXACT_45DEG,          VERTICAL),
                          P(-MUCH_DEFLECTED, VERTICAL),
                          P(-VERTICAL,       MUCH_DEFLECTED).Albedo(boost::none)
                        ));

INSTANTIATE_TEST_CASE_P(GlossyTransmissiveDielectric,
                        ShaderSymmetry,
                        ::testing::Values(
                          P(UNUSED_VECTOR, VERTICAL)
                          // Shading normals make the BSDF asymmetric and break energy conservation.
                          // So I probably cannot expect this test to succeed.
                          //P(UNUSED_VECTOR, EXACT_45DEG),
                          //P(UNUSED_VECTOR, MUCH_DEFLECTED)
                        ));

} // namespace


} // namespace parameterized shader tests



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
  LambdaSelection wavelengths = SelectRgbPrimaryWavelengths();
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
    PathContext context{wavelengths};
    Medium::InteractionSample s = medium.SampleInteractionPoint(RaySegment{{{0.,0.,0.}, {0., 0., 1.,}}, cutoff_length}, Spectral3::Ones(), sampler, context);
    if (s.t  > cutoff_length)
      img.SetColor(255, 0, 0);
    else
    {
      img.SetColor(255, 255, 255);
      // Estimate transmission by integrating sigma_e*T(x).
      // Divide out sigma_s which is baked into the weight.
      // Multiply by sigma_e.
      SpectralN integrand = (sigma_a + sigma_s) / sigma_s;
      integral += s.weight * s.sigma_s * Take(integrand, wavelengths.indices);
    }
    int imgx = std::min<int>(img.width()-1, s.t * img_dx);
    img.DrawLine(imgx, 0, imgx, img.height()-1);
  }
  integral *= 1./N;
  Spectral3 exact_solution = Take((1. - (-sigma_t*cutoff_length).exp()).eval(), wavelengths.indices);
  for (int k=0; k<static_size<Spectral3>(); ++k)
    EXPECT_NEAR(integral[k], exact_solution[k], 0.1 * integral[k]);
  //display.show(img);
  //std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

