#include "gtest/gtest.h"
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/numeric/interval.hpp>

#include "ray.hxx"
#include "sampler.hxx"
#include "sphere.hxx"
#include "triangle.hxx"
#include "util.hxx"
#include "shader.hxx"
#include "sampler.hxx"

#include "cubature.h"

// A review of statistical testing applied to computer graphics!
// http://www0.cs.ucl.ac.uk/staff/K.Subr/Files/Papers/PG2007.pdf

//http://mathworld.wolfram.com/BinomialDistribution.html
std::tuple<double, double> BinomialDistributionMeanStdev(int n, double p)
{
  assert(p >= 0.);
  return std::make_tuple(p*n, std::sqrt(n*p*(1.-p)));
}

std::tuple<double, double> BinomialDistributionMeanStdev(int _n, double _p, double _p_err)
{
  assert(_p >= 0. && _p_err >= 0.);
  using namespace boost::numeric;
  using namespace interval_lib;
  using I = interval<double>;
  double pl = std::max(0., _p-_p_err);
  double ph = std::min(1., _p+_p_err);
  I p{pl, ph};
  I n{_n, _n};
  I mean = p*n;
  I stddev = sqrt(n*p*(1.-p));
  double uncertainty = stddev.upper() + 0.5*(mean.upper()-mean.lower());
  return std::make_tuple(_p*_n, uncertainty);
}


TEST(BinomialDistribution, WithIntervals)
{
  double mean1, stddev1;
  double mean2, stddev2;
  std::tie(mean1, stddev1) = BinomialDistributionMeanStdev(10, 0.6);
  std::tie(mean2, stddev2) = BinomialDistributionMeanStdev(10, 0.6, 0.);
  EXPECT_NEAR(mean1, mean2, 1.e-9);
  EXPECT_NEAR(stddev1, stddev2, 1.e-9);
}


void CheckNumberOfSamplesInBin(const char *name, int num_smpl_in_bin, int total_num_smpl, double p_of_bin, double number_of_sigmas_threshold=3., double p_of_bin_error = 0.)
{
  double mean, sigma;
  std::tie(mean, sigma) = BinomialDistributionMeanStdev(total_num_smpl, p_of_bin, p_of_bin_error);
  std::stringstream additional_output;
  if (name)
    additional_output << "Expected in " << name << ": " << mean << "+/-" << sigma << " Actual: " << num_smpl_in_bin << " of " << total_num_smpl << std::endl;
  EXPECT_NEAR(num_smpl_in_bin, mean, sigma*number_of_sigmas_threshold) << additional_output.str();
}


namespace ChiSqrInternal
{

struct Bin {
  double count;
  double expected;
};


// TODO: I wish I had a super lightweight range checked (in debug mode only ofc.) array view.

void MergeBinsToGetAboveMinNumSamplesCutoff(Bin *bins, int &num_bins, double low_expected_num_samples_cutoff)
{
  //http://en.cppreference.com/w/cpp/algorithm/make_heap
  // ... and so on.
  auto cmp = [](const Bin &a, const Bin &b) -> bool { return a.count > b.count; }; // Using > instead of < makes it a min-heap.
  std::make_heap(bins, bins+num_bins, cmp);
  while (num_bins > 1)
  {
    // Get smallest element to the end.
    std::pop_heap(bins, bins+num_bins, cmp);
    if (bins[num_bins-1].expected > low_expected_num_samples_cutoff)
      break;
    // Get the second smallest element to the second last position.
    std::pop_heap(bins, bins+num_bins-1, cmp);
    // Merge them.
    Bin &merged = bins[num_bins-2];
    Bin &removed = bins[num_bins-1];
    merged.count += removed.count;
    merged.expected += removed.expected;
    removed.count = 0;
    removed.expected = 0;
    --num_bins; 
    // Modifications done. Get last element back into heap.
    std::push_heap(bins, bins+num_bins, cmp);
  }
  assert(num_bins >= 1);
  assert((num_bins == 1) || (bins[num_bins-1].expected > low_expected_num_samples_cutoff));
}

}


double ChiSquaredProbability(const int *counts, const double *weights, int num_bins, double low_expected_num_samples_cutoff = 5)
{
  using namespace ChiSqrInternal;
  
  int num_samples = std::accumulate(counts, counts+num_bins, 0); // It's the sum.
  double probability_normalization = std::accumulate(weights, weights+num_bins, 0.);

  probability_normalization = probability_normalization>0. ? 1./probability_normalization : 0.;
  
  std::unique_ptr<Bin[]> bins{new Bin[num_bins]};
  for (int i=0; i<num_bins; ++i)
    bins[i] = Bin{(double)counts[i], weights[i]*num_samples*probability_normalization};

  int original_num_bins = num_bins;
  MergeBinsToGetAboveMinNumSamplesCutoff(bins.get(), num_bins, low_expected_num_samples_cutoff);
  
  double chi_sqr = 0.;
  for (int i=0; i<num_bins; ++i)
  {
    chi_sqr += Sqr(bins[i].count - bins[i].expected)/bins[i].expected;
  }

  if (original_num_bins > num_bins*3)
  {
    std::cout << "Chi-Sqr Test merged " << (original_num_bins-num_bins) << " bins because their sample count was below " << low_expected_num_samples_cutoff << std::endl;
  }
  
  // https://www.boost.org/doc/libs/1_66_0/libs/math/doc/html/math_toolkit/dist_ref/dists/chi_squared_dist.html
  // https://www.boost.org/doc/libs/1_66_0/libs/math/doc/html/math_toolkit/dist_ref/nmp.html#math_toolkit.dist_ref.nmp.cdf
  boost::math::chi_squared_distribution<double> distribution(num_bins-1);
  double prob_observe_ge_chi_sqr = cdf(complement(distribution, chi_sqr));
  
  //std::cout << "chi-sqr prob = " << prob_observe_ge_chi_sqr << std::endl;
  //EXPECT_GE(prob_observe_ge_chi_sqr, p_threshold);
  return prob_observe_ge_chi_sqr;  
}


// By https://en.wikipedia.org/wiki/Central_limit_theorem
// for large number of samples.
double StddevOfAverage(double sample_stddev, int num_samples)
{
  return sample_stddev / std::sqrt(num_samples);
}


double StudentsTValue(double sample_avg, double avg_stddev, double true_mean)
{
  return (sample_avg - true_mean) / avg_stddev;
}


double StudentsTDistributionOutlierProbability(double t, double num_samples)
{
  boost::math::students_t_distribution<double> distribution(num_samples);
  if (t > 0.)
    return cdf(complement(distribution, t));
  else
    return cdf(distribution, t);
}


void TestSampleAverage(double sample_avg, double sample_stddev, double num_samples, double true_mean, double true_mean_error, double p_value)
{
  double avg_stddev = StddevOfAverage(sample_stddev, num_samples);
  if (avg_stddev > 0.)
  {
    double prob;
    // Normally I would use the probability of the distance greater than |(sample_avg - true_mean)| w.r.t. to a normal distribution.
    // I use normal distribution rather than t-test stuff because I have a lot of samples.
    // However, since I also have integration errors, I "split" the normal distribution in half and move both sides to the edges of
    // the integration error bounds. Clearly, I the MC estimate is very good, I want to check if sample_avg is precisely within the 
    // integration bounds. On the other hand, if the integration error is zero, I want to use the usual normal distribution as 
    // stated above. My scheme here achives both of these edge cases.
    if (sample_avg < true_mean - true_mean_error)
      prob = StudentsTDistributionOutlierProbability(StudentsTValue(sample_avg, avg_stddev, true_mean - true_mean_error), num_samples);
    else if(sample_avg > true_mean + true_mean_error)
      prob = StudentsTDistributionOutlierProbability(StudentsTValue(sample_avg, avg_stddev, true_mean + true_mean_error), num_samples);
    else
      prob = StudentsTDistributionOutlierProbability(0., num_samples);
    EXPECT_GE(prob, p_value) << "Probability to find sample average " << sample_avg << " +/- " << avg_stddev << " w.r.t. true mean " << true_mean << "+/-" << true_mean_error << 
    " is " << prob << " which is lower than p-value " << p_value << std::endl;
    //std::cout << "Test Sample Avg Prob = " << prob << std::endl;
  }
  else
  {
    EXPECT_NEAR(sample_avg, true_mean, true_mean_error);
  }
}


void TestProbabilityOfMeanLowerThanUpperBound(double sample_avg, double sample_stddev, double num_samples, double bound, double p_value)
{
  double avg_stddev = StddevOfAverage(sample_stddev, num_samples);
  if (avg_stddev > 0.)
  {
    boost::math::students_t_distribution<double> distribution(num_samples);
    double prob = cdf(complement(distribution, StudentsTValue(sample_avg, avg_stddev, bound)));
    EXPECT_GE(prob, p_value) << "Probability to find the mean smaller than " << bound << " is " << prob << " which is lower than the p-value " << p_value << ", given the average " << sample_avg << " +/- " << avg_stddev << std::endl;
    //std::cout << "Test Prob Upper Bound = " << prob << std::endl;
  }
  else
  {
    EXPECT_LE(sample_avg, bound);
  }
}


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
auto Integral2D(Func func, Double2 start, Double2 end, double absError, double relError, int max_eval = 0, double *error_estimate = nullptr)
{
  double result, error;
  int status = hcubature(1, cubature_wrapper::f1d<Func>, &func, 2, start.data(), end.data(), max_eval, absError, relError, ERROR_L2, &result, &error);
  if (status != 0)
    throw std::runtime_error("Cubature failed!");
  if (error_estimate)
    *error_estimate = error;
  return result;
}

template<class Func>
auto MultivariateIntegral2D(Func func, Double2 start, Double2 end, double absError, double relError, int max_eval = 0, decltype(func(Double2{})) *error_estimate = nullptr)
{
  using R  = decltype(func(Double2{}));
  R result, error;
  int status = hcubature(result.size(), cubature_wrapper::fnd<Func>, &func, 2, start.data(), end.data(), max_eval, absError, relError, ERROR_L2, result.data(), error.data());
  if (status != 0)
    throw std::runtime_error("Cubature failed!");
  if (error_estimate)
    *error_estimate = error;
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


class Scatterer
{
public:
  virtual ~Scatterer() {};
  virtual double ProbabilityDensity(const Double3 &wi, const Double3 &wo) const = 0;
  virtual Spectral3 ScatterFunction(const Double3 &wi, const Double3 &wo) const = 0;
  virtual Spectral3 ReflectedRadiance(const Double3 &wi, const Double3 &wo) const = 0;
  virtual ScatterSample MakeScatterSample(const Double3 &wi, Sampler &sampler) const = 0;
  virtual bool IsSymmetric() const = 0;
};


class ScatterTests
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
  int num_samples {0};  

public:
    Sampler sampler;
  
public:
  ScatterTests(const Scatterer &_scatterer)
    : cubemap{Nbins}, scatterer{_scatterer}
  {}
  
  void RunAllCalculations(const Double3 &_reverse_incident_dir, int _num_samples)
  {
    this->reverse_incident_dir = _reverse_incident_dir;
    this->num_samples = _num_samples;
    DoSampling();
    ComputeBinProbabilities();
    ComputeCubatureIntegral();
    CheckSymmetry();
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
    double chi_sqr_probability = ChiSquaredProbability(&bin_sample_count[0], &bin_probabilities[0], bin_probabilities.size());
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
    EXPECT_NEAR(total_probability+probability_of_delta_peaks, 1., total_probability_error+TEST_EPS);
  }
 
 
  void DoSampling()
  {
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
    auto CrossValidate = [this](const ScatterSample &smpl) -> void
    {
      // Do pdf and scatter function values in the smpl struct match what comes out of member functions?
      double pdf_crossvalidate = scatterer.ProbabilityDensity(reverse_incident_dir, smpl.coordinates);
      ASSERT_NEAR(PdfValue(smpl), pdf_crossvalidate, 1.e-6);
      Spectral3 val_crossvalidate = scatterer.ReflectedRadiance(reverse_incident_dir, smpl.coordinates);
      ASSERT_NEAR(smpl.value[0], val_crossvalidate[0], 1.e-6);
      ASSERT_NEAR(smpl.value[1], val_crossvalidate[1], 1.e-6);
      ASSERT_NEAR(smpl.value[2], val_crossvalidate[2], 1.e-6);
    };
    
    for (int snum = 0; snum<num_samples; ++snum)
    {
      auto smpl = scatterer.MakeScatterSample(reverse_incident_dir, sampler);
      if (IsFromPdf(smpl) && snum<100)
        CrossValidate(smpl);
      // Can only integrate the non-delta density. 
      // Keep track of delta-peaks separately.
      if (IsFromPdf(smpl))
      {
        int idx = SampleIndex(smpl);
        bin_sample_count[idx] += 1;
        diffuse_scattered_estimate += smpl.value / smpl.pdf_or_pmf; 
      }
      else
      {
        RegisterDeltaSample(smpl.coordinates, (double)smpl.pdf_or_pmf);
        // Think of Russian Roulette. When it is decided to sample the delta-peaks,
        // the other contribution is formally zerod out. In order to make the
        // statistics work, I must still count it as regular sample.
        diffuse_scattered_estimate += Spectral3{0.};
      }
      // I can ofc integrate delta-peak contributions by MC.
      total_scattered_estimate += smpl.value / smpl.pdf_or_pmf; 
    }
  }
 
 
  void ComputeBinProbabilities()
  {
    constexpr int MAX_NUM_FUNC_EVALS = 100000;
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
      double prob = Integral2D(probabilityDensityTimesJ, start, end, 1.e-3, 1.e-2, MAX_NUM_FUNC_EVALS, &err);
      bin_probabilities[idx] = prob;
      bin_probabilities_error[idx] = err;
      total_probability += bin_probabilities[idx];
      total_probability_error += err;
    }
  }
  
  
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
        Spectral3 val = scatterer.ReflectedRadiance(reverse_incident_dir, omega);
        Spectral3 ret = val*cubemap.UVtoJ(x);
        assert(ret.allFinite());
        return ret;
      };
      Double2 start, end;
      std::tie(start, end) = cubemap.CellToUVBounds(i, j);       
      Eigen::Array3d err{NaN};
      Eigen::Array3d cell_integral = MultivariateIntegral2D(functionValueTimesJ, start, end, 1.e-3, 1.e-2, MAX_NUM_FUNC_EVALS, &err);
      this->integral_cubature += cell_integral;
      this->integral_cubature_error += err;
    }
  }
  
  
  void CheckSymmetry()
  {
    const int NUM_SAMPLES = 100;
    for (int snum = 0; snum<NUM_SAMPLES; ++snum)
    {
      Double3 wi = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
      Double3 wo = SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
      if (scatterer.IsSymmetric())
      {
        double pdf_val = scatterer.ProbabilityDensity(wi, wo); 
        double pdf_rev = scatterer.ProbabilityDensity(wo, wi);
        EXPECT_NEAR(pdf_val, pdf_rev, 1.e-6);
      }
      Spectral3 f_val = scatterer.ScatterFunction(wi, wo);
      Spectral3 f_rev = scatterer.ScatterFunction(wo, wi);
      EXPECT_NEAR(f_val[0], f_rev[0], 1.e-6);
      EXPECT_NEAR(f_val[1], f_rev[1], 1.e-6);
      EXPECT_NEAR(f_val[2], f_rev[2], 1.e-6);
    }
  }

  
  rj::Value CubemapToJSON(rj::Alloc &alloc);
  void DumpVisualization(const std::string filename);
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
  
  Spectral3 ReflectedRadiance(const Double3 &wi, const Double3 &wo) const override
  {
    return ScatterFunction(wi, wo);
  }
  
  ScatterSample MakeScatterSample(const Double3 &wi, Sampler &sampler) const override
  {
    return pf.SampleDirection(wi, sampler);
  }
  
  bool IsSymmetric() const override
  {
    return true;
  }
};


class PhaseFunctionTests : public ScatterTests
{
    PhasefunctionScatterer pf_wrapper;
  public:
    PhaseFunctionTests(const PhaseFunctions::PhaseFunction &_pf)
      :  ScatterTests{pf_wrapper}, pf_wrapper{_pf}
    {
    }
};


class ShaderScatterer : public Scatterer
{
  const Shader &sh;
  PathContext context;
  std::unique_ptr<TexturedSmoothTriangle> triangle;
  
  mutable RaySurfaceIntersection last_intersection;
  mutable Double3 last_incident_dir;
  double ior_below_over_above; // Ratio
  
  RaySurfaceIntersection& MakeIntersection(const Double3 &reverse_incident_dir) const
  {
    if (last_incident_dir != reverse_incident_dir)
    {
      RaySegment seg{{reverse_incident_dir, -reverse_incident_dir}, LargeNumber};
      HitId hit;
      bool ok = triangle->Intersect(seg.ray, seg.length, hit);
      last_intersection = RaySurfaceIntersection{hit, seg};
      last_incident_dir = reverse_incident_dir;
    }
    return last_intersection;
  }
  
  Spectral3 UndoneIorScaling(Spectral3 value, const Double3 &wi, const RaySurfaceIntersection &intersection, const Double3 &wo) const
  {
    // TODO: Consider also the adjoint!
    if (Dot(wo, intersection.normal) < 0.) // We have transmission
    {
      // According to Veach's Thesis (Eq. 5.12), transmitted radiance is scaled by (eta_t/eta_i)^2.
      // Here I undo the scaling factor, so that when summing, the total scattered radiance equals
      // the incident radiance. And that conservation is much simpler to check for.
      if (Dot(wi, intersection.geometry_normal) >= 0.) // Light goes from below surface to above. Because wi, wo denote direction of randomwalk.
        value *= Sqr(ior_below_over_above);
      else
        value *= Sqr(1./ior_below_over_above);
    }
    return value;
  }
  
public:
  ShaderScatterer(const Shader &_sh)
    : sh{_sh},
      last_incident_dir{NaN},
      ior_below_over_above{1.}
  {
    SetShadingNormal(Double3{0,0,1});
    context = PathContext{Color::LambdaIdxClosestToRGBPrimaries()};
  }
  
  void SetIorRatio(double _ior_below_over_above)
  {
    this->ior_below_over_above = _ior_below_over_above;
  }
    
  void SetShadingNormal(const Double3 &shading_normal)
  {
    triangle = std::make_unique<TexturedSmoothTriangle>(
      Double3{ -1, -1, 0}, Double3{ 1, -1, 0 }, Double3{ 0, 1, 0 },
      shading_normal, shading_normal, shading_normal,
      Double3{0, 0, 0}, Double3{1, 0, 0}, Double3{0.5, 1, 0});
    last_incident_dir = Double3{NaN}; // Force regeneration of intersection!
  }
  
  
  double ProbabilityDensity(const Double3 &wi, const Double3 &wo) const override
  {
    double ret = NaN;
    auto &intersection = MakeIntersection(wi);
    sh.EvaluateBSDF(wi, intersection, wo, context, &ret);
    return ret;
  }
  
  Spectral3 ScatterFunction(const Double3 &wi, const Double3 &wo) const override
  {
    auto &intersection = MakeIntersection(wi);
    return UndoneIorScaling(
      sh.EvaluateBSDF(wi, intersection, wo, context, nullptr), 
      wi, intersection, wo);
  }
  
  Spectral3 ReflectedRadiance(const Double3 &wi, const Double3 &wo) const override
  {
    auto &intersection = MakeIntersection(wi);
    auto value = sh.EvaluateBSDF(wi, intersection, wo, context, nullptr);
    value *= std::abs(Dot(wo, intersection.shading_normal));
    value = UndoneIorScaling(value, wi, intersection, wo);
    return value;
  }
  
  ScatterSample MakeScatterSample(const Double3 &wi, Sampler &sampler) const override
  {
    auto &intersection = MakeIntersection(wi);
    auto smpl = sh.SampleBSDF(wi, intersection, sampler, context);
    smpl.value *= std::abs(Dot(smpl.coordinates, intersection.shading_normal));
    smpl.value = UndoneIorScaling(smpl.value, wi, intersection, smpl.coordinates);
    return smpl;
  }
  
  bool IsSymmetric() const override
  {
    return false;
  }
};


class ShaderTests : public ScatterTests
{
    ShaderScatterer sh_wrapper;
  public:
    ShaderTests(const Shader &_sh, double ior_below_over_above = 1.)
      :  ScatterTests{sh_wrapper}, sh_wrapper{_sh}
    {
      sh_wrapper.SetIorRatio(ior_below_over_above);
    }
    
    void RunAllCalculations(const Double3 &_reverse_incident_dir, const Double3 shading_normal, int _num_samples)
    {
      sh_wrapper.SetShadingNormal(shading_normal);
      ScatterTests::RunAllCalculations(_reverse_incident_dir, _num_samples);
    }
};



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
      rj::Value json_smpl = rj::ScatterSampleToJSON(smpl, alloc);
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
            Eigen::Array<double, 5, 1> integral = MultivariateIntegral2D(integrand, std::get<0>(bounds), std::get<1>(bounds), 1.e-3, 1.e-2, 1000);
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


rj::Value ScatterTests::CubemapToJSON(rj::Alloc& alloc)
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


void ScatterTests::DumpVisualization(const std::string filename)
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




TEST(PhasefunctionTests, Uniform)
{
  PhaseFunctions::Uniform pf{};
  PhaseFunctionTests test(pf);
  test.RunAllCalculations(Double3{0,0,1}, 5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, Rayleigh)
{
  PhaseFunctions::Rayleigh pf{};
  PhaseFunctionTests test(pf);
  test.RunAllCalculations(Double3{0,0,1}, 5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, HenleyGreenstein)
{
  PhaseFunctions::HenleyGreenstein pf(0.4);
  PhaseFunctionTests test(pf);
  test.RunAllCalculations(Double3{0,0,1}, 5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, Combined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::Combined pf(Spectral3{1., 1., 1.}, Spectral3{.1, .2, .3}, pf1, Spectral3{.3, .4, .5}, pf2);
  PhaseFunctionTests test(pf);
  test.RunAllCalculations(Double3{0,0,1}, 5000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}


TEST(PhasefunctionTests, SimpleCombined)
{
  PhaseFunctions::HenleyGreenstein pf1{0.4};
  PhaseFunctions::Uniform pf2;
  PhaseFunctions::SimpleCombined pf{Spectral3{.1, .2, .3}, pf1, Spectral3{.3, .4, .5}, pf2};
  PhaseFunctionTests test(pf);
  test.sampler.Seed(12356);
  test.RunAllCalculations(Double3{0,0,1}, 10000);
  test.TestCountsDeviationFromSigma(3);
  test.TestChiSqr();
  test.CheckIntegral();
}



// Materials ...
// TODO: Certainly all that repetition is not very nice ...

///////////////////////////////////////
static const double DEFAULT_IOR = 1.3;
static const Double3 NORMAL_VERTICAL = Double3{0,0,1};
static const Double3 NORMAL_45DEG    = Normalized(Double3{0,1,1});
static const Double3 NORMAL_MUCH_DEFLECTED = Normalized(Double3{0,10,1});
static const Double3 INCIDENT_VERTICAL = NORMAL_VERTICAL;
static const Double3 INCIDENT_45DEG = NORMAL_45DEG;
static const Double3 INCIDENT_MUCH_DEFLECTED = NORMAL_MUCH_DEFLECTED;


TEST(ShaderTests, DiffuseShader1)
{
  DiffuseShader sh(Color::SpectralN{1.0}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_VERTICAL, 2000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral(Spectral3{1.});
}

TEST(ShaderTests, DiffuseShader2)
{
  DiffuseShader sh(Color::SpectralN{0.5}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_45DEG, NORMAL_VERTICAL, 2000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral(Spectral3{0.5});
}

TEST(ShaderTests, DiffuseShader3)
{
  // NOTE: High variant because the pdf distributes samples according to geometric normal. Hence many samples needed for integral to converge!
  // TODO: I should probably use another sampling scheme.
  DiffuseShader sh(Color::SpectralN{0.5}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_MUCH_DEFLECTED, 50000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral(Spectral3{0.5});
}

TEST(ShaderTests, DiffuseShader4)
{
  DiffuseShader sh(Color::SpectralN{0.5}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_MUCH_DEFLECTED, NORMAL_45DEG, 50000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral(Spectral3{0.5});
}




TEST(ShaderTests, SpecularReflectiveShader1)
{
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_VERTICAL, 100);
  test.TestCountsDeviationFromSigma(3.);
  test.CheckIntegral(Spectral3{0.5});
}

TEST(ShaderTests, SpecularReflectiveShader2)
{
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_45DEG, NORMAL_VERTICAL, 100);
  test.TestCountsDeviationFromSigma(3.);
  test.CheckIntegral(Spectral3{0.5});
}

TEST(ShaderTests, SpecularReflectiveShader3)
{
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_MUCH_DEFLECTED, 100);
  test.TestCountsDeviationFromSigma(3.);
  // This is a pathological case, where due to the strongly perturbed
  // shading normal, the incident direction is specularly reflected
  // below the geometric surface. I don't know what I can do but to
  // assign zero reflected radiance. Thus sane rendering algorithms
  // would not consider this path any further.
  test.CheckIntegral(Spectral3{0.0});
}

TEST(ShaderTests, SpecularReflectiveShader4)
{
  SpecularReflectiveShader sh(Color::SpectralN{0.5});
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_MUCH_DEFLECTED, NORMAL_45DEG, 100);
  test.TestCountsDeviationFromSigma(3.);
  test.CheckIntegral(Spectral3{0.5});
}



// Test case for SpecularDenseDielectric selected with almost unnaturally
// high albedo of the diffuse layer. That is to uncover potential violation
// in energy conservation. (Too high re-emission might otherwise be masked
// by reflectivity factor.
TEST(ShaderTests, SpecularDenseDielectricShader1)
{
  SpecularDenseDielectricShader sh(0.2, Color::SpectralN{0.99}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_VERTICAL, 10000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral();
}

TEST(ShaderTests, SpecularDenseDielectricShader2)
{
  SpecularDenseDielectricShader sh(0.2, Color::SpectralN{0.99}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_45DEG, NORMAL_VERTICAL, 10000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral();
}

TEST(ShaderTests, SpecularDenseDielectricShader3)
{
  SpecularDenseDielectricShader sh(0.2, Color::SpectralN{0.99}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_MUCH_DEFLECTED, 10000);
  test.TestCountsDeviationFromSigma(3.5);
  test.TestChiSqr();
  test.CheckIntegral();
}

TEST(ShaderTests, SpecularDenseDielectricShader4)
{
  SpecularDenseDielectricShader sh(0.2, Color::SpectralN{0.99}, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_MUCH_DEFLECTED, NORMAL_45DEG, 10000);
  test.TestCountsDeviationFromSigma(3.5);
  test.TestChiSqr();
  test.CheckIntegral();
}



TEST(ShaderTests, MicrofacetShader1)
{
  MicrofacetShader sh(Color::SpectralN{0.3}, 0.4, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_VERTICAL, 10000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral();
  //test.DumpVisualization("/tmp/MicrofacetShader1.json");
}

TEST(ShaderTests, MicrofacetShader2)
{
  MicrofacetShader sh(Color::SpectralN{0.3}, 0.4, nullptr);
  ShaderTests test(sh);
  // Don't have the point where the PDF diverges coincide with the boundaries of some bins.
  test.RunAllCalculations(Normalized(Double3{0,1,11}), NORMAL_VERTICAL, 20000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral();
  //test.DumpVisualization("/tmp/MicrofacetShader2.json");
}


TEST(ShaderTests, MicrofacetShader2b)
{
  MicrofacetShader sh(Color::SpectralN{0.3}, 0.4, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_MUCH_DEFLECTED, NORMAL_VERTICAL, 20000);
  test.TestCountsDeviationFromSigma(3.);
  test.TestChiSqr();
  test.CheckIntegral();
  //test.DumpVisualization("/tmp/MicrofacetShader2b.json");
}


TEST(ShaderTests, MicrofacetShader3)
{
  MicrofacetShader sh(Color::SpectralN{0.3}, 0.4, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_MUCH_DEFLECTED, 20000);
  test.TestCountsDeviationFromSigma(3.5);
  test.TestChiSqr();
  test.CheckIntegral();
  //test.DumpVisualization("/tmp/MicrofacetShader3.json");
}

TEST(ShaderTests, MicrofacetShader4)
{
  MicrofacetShader sh(Color::SpectralN{0.3}, 0.4, nullptr);
  ShaderTests test(sh);
  test.RunAllCalculations(INCIDENT_MUCH_DEFLECTED, NORMAL_45DEG, 10000);
  test.TestCountsDeviationFromSigma(3.5);
  test.TestChiSqr();
  test.CheckIntegral();
  //test.DumpVisualization("/tmp/MicrofacetShader4.json");
}



TEST(ShaderTests, SpecularTransmissiveDielectricShader1)
{ 
  SpecularTransmissiveDielectricShader sh(DEFAULT_IOR);
  ShaderTests test(sh, DEFAULT_IOR);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_VERTICAL, 100);
  test.CheckIntegral(Spectral3{1.});
}

TEST(ShaderTests, SpecularTransmissiveDielectricShader2)
{
  SpecularTransmissiveDielectricShader sh(DEFAULT_IOR);
  ShaderTests test(sh, DEFAULT_IOR);
  test.RunAllCalculations(Normalized(Double3{0,1,101}), NORMAL_VERTICAL, 100);
  test.CheckIntegral(Spectral3{1.});
}

TEST(ShaderTests, SpecularTransmissiveDielectricShader3)
{
  SpecularTransmissiveDielectricShader sh(DEFAULT_IOR);
  ShaderTests test(sh, DEFAULT_IOR);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_MUCH_DEFLECTED, 100);
  // Not checking the total scattered radiance here???
  // If reflected, the exit direction is below the goemetrical surface, in case of which the contribution is canceled.
  // However if transmitted, the exit direction is correctly below the geom. surface. So this makes a contribution to
  // the integral check. But since the energy from reflection is missing, the total scattered radiance is hard to predict.
  test.CheckIntegral();
}

TEST(ShaderTests, SpecularTransmissiveDielectricShader4)
{
  SpecularTransmissiveDielectricShader sh(DEFAULT_IOR);
  ShaderTests test(sh, DEFAULT_IOR);
  test.RunAllCalculations(INCIDENT_MUCH_DEFLECTED, NORMAL_VERTICAL, 100);
  test.CheckIntegral(Spectral3{1.});
}

TEST(ShaderTests, SpecularTransmissiveDielectricShader5)
{ 
  SpecularTransmissiveDielectricShader sh(1./DEFAULT_IOR);
  ShaderTests test(sh, 1./DEFAULT_IOR);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_VERTICAL, 100);
  test.CheckIntegral(Spectral3{1.});
}

TEST(ShaderTests, SpecularTransmissiveDielectricShader6)
{
  SpecularTransmissiveDielectricShader sh(1./DEFAULT_IOR);
  ShaderTests test(sh, 1./DEFAULT_IOR);
  test.RunAllCalculations(INCIDENT_45DEG, NORMAL_VERTICAL, 100);
  test.CheckIntegral(Spectral3{1.});
}

TEST(ShaderTests, SpecularTransmissiveDielectricShader7)
{
  SpecularTransmissiveDielectricShader sh(1./DEFAULT_IOR);
  ShaderTests test(sh, 1./DEFAULT_IOR);
  test.RunAllCalculations(INCIDENT_VERTICAL, NORMAL_MUCH_DEFLECTED, 100);
  test.CheckIntegral(Spectral3{0.}); // Scattered below geometric surface, Best I can do is zero out the contribution.
}

TEST(ShaderTests, SpecularTransmissiveDielectricShader8)
{
  SpecularTransmissiveDielectricShader sh(1./DEFAULT_IOR);
  ShaderTests test(sh, 1./DEFAULT_IOR);
  test.RunAllCalculations(INCIDENT_MUCH_DEFLECTED, NORMAL_VERTICAL, 100); // Total reflection!
  test.CheckIntegral(Spectral3{1.});
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

