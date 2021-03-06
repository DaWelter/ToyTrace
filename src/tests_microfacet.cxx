#include "gtest/gtest.h"

#ifdef HAVE_JSON
#include <rapidjson/document.h>
#endif

#include "cubature_wrapper.hxx"
#include "cubemap.hxx"
#include "tests_stats.hxx"

#include "normaldistributionfunction.hxx"
#include "util.hxx"
#include "sampler.hxx"
#include "shader_physics.hxx"

namespace {
static const Double3 VERTICAL = Double3{0,0,1};
static const Double3 EXACT_45DEG    = Normalized(Double3{0,1,1});
static const Double3 ALMOST_45DEG    = Normalized(Double3{0,1,2});
static const Double3 MUCH_DEFLECTED = Normalized(Double3{0,10,1});
static const Double3 BELOW          = Normalized(Double3{0,5,-1});
}


template<class Func>
auto IntegralOverCubemap(Func func, const CubeMap &cubemap, double absError, double relError, int max_eval = 0) //, std::vector<decltype(func(Double2{}))> *errors = nullptr)
{
  using R = decltype(func(Double2{}));
  std::vector<R> result;  result.reserve(cubemap.TotalNumBins());
  for (int idx=0; idx<cubemap.TotalNumBins(); ++idx)
  {
    auto [side, i, j] = cubemap.IndexToCell(idx);
    // Integrating this gives me the probability of a sample falling
    // into the current bin. The integration is carried out on a square in
    // R^2, hence the scale factor J from the Jacobian of the variable transform.
    auto probabilityDensityTimesJ = [side = side,&cubemap,&func](const Double2 x) -> double
    {
      Double3 omega = cubemap.UVToOmega(side, x);
      R val = func(omega)*cubemap.UVtoJ(x);
      return val;
    };
    auto [start, end] = cubemap.CellToUVBounds(i, j);
    //R err = NaN;
    const R prob = Integral2D(probabilityDensityTimesJ, start, end, absError, relError, max_eval, nullptr); //, errors ? &err : nullptr);
    result.emplace_back(prob);
    //if (errors)
    //  errors->emplace_back(err);
  }
  return result;
}


template<class Func>
std::vector<int> SampleOverCubemap(Func generator, const CubeMap &cubemap, int count)
{
  std::vector<int> result(cubemap.TotalNumBins(), 0);
  for (int k=0; k<count; ++k)
  {
    const Double3 w = generator();
    const auto [side, i, j] = cubemap.OmegaToCell(w);
    const int idx = cubemap.CellToIndex(side, i, j);
    result[idx]++;
  }
  return result;
}

template<class Func, class T>
auto Map(const std::vector<T> &v, Func &&f)
{
  std::vector<decltype(f(v[0]))> ret; ret.reserve(v.size());
  for (const auto &x : v)
    ret.emplace_back(f(x));
  return ret;
}



#ifdef HAVE_JSON
namespace rj
{
using namespace rapidjson;

using Alloc = rapidjson::Document::AllocatorType; // For some reason I must supply an "allocator" almost everywhere. Why? Who knows?!

rapidjson::Value Array3ToJSON(const Eigen::Array<double, 3, 1> &v, Alloc &alloc);
void Write(rj::Document &doc, const std::string &filename);

rj::Value CubemapToJSON(const CubeMap &cubemap, rj::Alloc& alloc)
{
  rj::Value json_side;
  json_side.SetArray();
  for (int side = 0; side < 6; ++side)
  {
    rj::Value json_i(rj::kArrayType);
    for (int i = 0; i < cubemap.BinsPerAxis(); ++i)
    {
      rj::Value json_j(rj::kArrayType);
      for (int j = 0; j < cubemap.BinsPerAxis(); ++j)
      {
        rj::Value json_bin(rj::kObjectType);
        // Store index by which to look up data for current cell.
        int idx = cubemap.CellToIndex(side, i, j);
        json_bin.AddMember("index", idx, alloc);
        // Store coordinates of cell center. As direction on the unit sphere.
        auto [uv_min, uv_max] = cubemap.CellToUVBounds(i, j);
        Double2 centeruv = (uv_min+uv_max)*0.5;
        json_bin.AddMember("center", rj::Array3ToJSON(cubemap.UVToOmega(side, centeruv).array(), alloc), alloc);
        // Store coordinates of the corners
        Double3 w00 = cubemap.UVToOmega(side, uv_min);
        Double3 w11 = cubemap.UVToOmega(side, uv_max);
        json_bin.AddMember("w00", rj::Array3ToJSON(w00.array(), alloc), alloc);
        json_bin.AddMember("w11", rj::Array3ToJSON(w11.array(), alloc), alloc);
        json_bin.AddMember("J", cubemap.UVtoJ(centeruv), alloc);
        json_j.PushBack(json_bin, alloc);
      }
      json_i.PushBack(json_j, alloc);
    }
    json_side.PushBack(json_i, alloc);
  }
  return json_side;
}


template<class T>
rapidjson::Value ToJSON(const T &v, Alloc &alloc)
{
  return rj::Value(v);
}

template<>
rapidjson::Value ToJSON<Eigen::Array<double, 3, 1>>(const Eigen::Array<double, 3, 1> &v, Alloc &alloc)
{
  return Array3ToJSON(v, alloc);
}


template<class Container>
rj::Value ContainerToJSON(const Container &c, rj::Alloc &alloc)
{
  rj::Value json_container(rj::kArrayType);
  for (const auto &x : c)
  {
    json_container.PushBack(ToJSON(x, alloc), alloc);
  }
  return json_container;
}


}

#endif




TEST(Microfacet, Beckmann)
{
  using NDF = BeckmanDistribution;
  CubeMap cubemap(8);
  const double alpha = 0.05;
  Sampler sampler;
  
  NDF ndf{alpha};

  auto density_func = [&](const Double3 &w)
  {
    // Argument is w.dot.n, where n is aligned with the z-axis.
    // So it simplifies to w.z. The second term comes from the fact
    // that we test NDF.|w.n|, which is a proper probability density.
    // Just NDF is not.
    return ndf.EvalByHalfVector(w[2])*std::abs(w[2]);
  };
  
  auto sample_gen = [&]() 
  {
    return ndf.SampleHalfVector(sampler.UniformUnitSquare());
  };
  
  std::vector<double> probs = IntegralOverCubemap(density_func, cubemap, 1.e-3, 1.e-2, 100000);
  std::vector<int> sample_counts = SampleOverCubemap(sample_gen, cubemap, 1000);
  
//   for (int i=0; i<cubemap.TotalNumBins(); ++i)
//   {
//     std::cout << strconcat(i,": p=", probs[i], ", cnt=", sample_counts[i]) << std::endl;
//   }
  double chi_sqr_probability = ChiSquaredProbability(sample_counts.data(), probs.data(), isize(probs));
  EXPECT_GE(chi_sqr_probability, 0.05);
}


using NDF_Params = std::tuple<double, Double3>;
using RefractParams = std::tuple<double, Double3, double>;

class ReflectTest : public testing::TestWithParam<NDF_Params>
{
public:
  using NDF = BeckmanDistribution;
  Sampler sampler;
};

class RefractTest : public testing::TestWithParam<RefractParams>
{
public:
  using NDF = BeckmanDistribution;
  Sampler sampler;
};


TEST_P(ReflectTest, Outdirection)
{
  /* Test the mapping from microfacet normal distribution to outgoing direction.
   * The incident ray is reflected specularly at the microfacet normal.
   * Thus the exitant ray is a random variable with some distribution on the unit sphere.
   * So here, I test the consistency of sampling and density of the existant direction.
   */
  
  const auto [alpha, wi] = this->GetParam();
  CubeMap cubemap(8); 
  NDF ndf{alpha};

  auto density_func = [wi=wi, &ndf](const Double3 &wo)
  {
    Double3 wh = Normalized(wo + wi);
    // The abs(...) in the argument to the NDF is required for
    // when wi lies below the ground plane. If it was not there,
    // the NDF would return 0, which is wrong!. Computation of the
    // half-vector from wi and wo, can produce half-vectors which also lie below
    // the ground plane. But the plain NDF is not defined there.
    // With the abs(...) we have the correct distribution for wo!
    double ndf_val = ndf.EvalByHalfVector(std::abs(wh[2]))*std::abs(wh[2]);
    return HalfVectorPdfToReflectedPdf(ndf_val, Dot(wh, wi));
  };
  
  auto sample_gen = [wi = wi, this, &ndf]()
  {
    Double3 wh = ndf.SampleHalfVector(sampler.UniformUnitSquare());
    return Reflected(wi,wh);
  };
  
  std::vector<double> probs = IntegralOverCubemap(density_func, cubemap, 1.e-3, 1.e-2, 100000);
  std::vector<int> sample_counts = SampleOverCubemap(sample_gen, cubemap, 1000);
  double chi_sqr_probability = ChiSquaredProbability(sample_counts.data(), probs.data(), isize(probs));
  EXPECT_GE(chi_sqr_probability, 0.05);
  // Integral of the density over all solid angles should be normalized to 1.
  double total_prob = std::accumulate(probs.begin(), probs.end(), 0.);
  EXPECT_NEAR(total_prob, 1., 1.e-2);
}



TEST_P(RefractTest, OutDirection)
{ 
  const auto [alpha, wi, eta_ground] = this->GetParam();
  CubeMap cubemap(8); 
  NDF ndf{alpha};
  double eta_i_over_t = 1.0/eta_ground;
  TransmissiveMicrofacetDensity density{wi, eta_i_over_t, ndf};
  
  std::vector<double> probs = IntegralOverCubemap(
    [&](const Double3 &wo) { return density.Pdf(wo); },
    cubemap, 1.e-3, 1.e-2, 100000);
  std::vector<int> sample_counts = SampleOverCubemap(
    [&]() { return density.Sample(sampler.UniformUnitSquare(), sampler.Uniform01()); }, 
    cubemap, 10000);

  double chi_sqr_probability = ChiSquaredProbability(sample_counts.data(), probs.data(), isize(probs));
  EXPECT_GE(chi_sqr_probability, 0.05);
  // Integral of the density over all solid angles should be normalized to 1.
  double total_prob = std::accumulate(probs.begin(), probs.end(), 0.);
  EXPECT_NEAR(total_prob, 1., 1.e-2);
}



TEST_P(ReflectTest, VNDFSampling)
{
  const auto [alpha, wi] = this->GetParam();
  CubeMap cubemap(8); 
  NDF ndf{alpha};
  
  auto density_func = [wi=wi, &ndf](const Double3 &wh)
  {
    double ndf_val = ndf.EvalByHalfVector(wh[2]);
    return VisibleNdfVCavity::Pdf(ndf_val, wh, wi);
  };
  
  auto sample_gen = [wi = wi, this, &ndf]() 
  {
    // The method by Heitz et. al (2014), Algorithm 3, for VCavity model.
    Double3 wh = ndf.SampleHalfVector(sampler.UniformUnitSquare());
    VisibleNdfVCavity::Sample(wh, wi, sampler.Uniform01());
    return wh;
  };
  
  std::vector<double> probs = IntegralOverCubemap(density_func, cubemap, 1.e-3, 1.e-2, 100000);
  std::vector<int> sample_counts = SampleOverCubemap(sample_gen, cubemap, 1000);
  double chi_sqr_probability = ChiSquaredProbability(sample_counts.data(), probs.data(), isize(probs));
  EXPECT_GE(chi_sqr_probability, 0.05);
  double total_prob = std::accumulate(probs.begin(), probs.end(), 0.);
  EXPECT_NEAR(total_prob, 1., 1.e-2);
}


TEST_P(ReflectTest, VNDFSamplingOutDirection)
{
  const auto [alpha, wi] = this->GetParam();
  CubeMap cubemap(8); 
  NDF ndf{alpha};
  
  auto density_func = [wi=wi, &ndf](const Double3 &wo)
  {
    Double3 wh = Normalized(wo + wi);
    if (wh[2] < 0)
      wh = -wh;
    double ndf_val = ndf.EvalByHalfVector(wh[2]);
    double vndf = VisibleNdfVCavity::Pdf(ndf_val, wh, wi);
    return HalfVectorPdfToReflectedPdf(vndf, Dot(wh, wi));
  };
  
  auto sample_gen = [wi = wi, this, &ndf]() 
  {
    // The method by Heitz et. al (2014), Algorithm 3, for VCavity model.
    Double3 wh = ndf.SampleHalfVector(sampler.UniformUnitSquare());
    VisibleNdfVCavity::Sample(wh, wi, sampler.Uniform01());
    return Reflected(wi,wh);
  };
  
  std::vector<double> probs = IntegralOverCubemap(density_func, cubemap, 1.e-3, 1.e-2, 100000);
  std::vector<int> sample_counts = SampleOverCubemap(sample_gen, cubemap, 1000);
  double chi_sqr_probability = ChiSquaredProbability(sample_counts.data(), probs.data(), isize(probs));
  EXPECT_GE(chi_sqr_probability, 0.05);
  double total_prob = std::accumulate(probs.begin(), probs.end(), 0.);
  EXPECT_NEAR(total_prob, 1., 1.e-2);
}


// // Only use when desparate.

// void RunVisualization(const Double3 &wi, double alpha, double eta_ground, const std::string &output_filename)
// {
//   std::cout << output_filename << std::endl;
//   using NDF = BeckmanDistribution;
//   CubeMap cubemap(32); 
//   NDF ndf{alpha};
//   Sampler sampler;
//   double eta_i_over_t = 1.0/eta_ground;
//   TransmissiveMicrofacetDensity density{wi, eta_i_over_t, ndf};
//   
//   std::vector<double> probs = IntegralOverCubemap(
//     [&](const Double3 &wo) { return density.Pdf(wo); },
//     cubemap, 1.e-3, 1.e-2, 100000);
//   std::vector<int> sample_counts = SampleOverCubemap(
//     [&]() { return density.Sample(sampler); }, 
//     cubemap, 1000000);
//   
//   double total_prob = std::accumulate(probs.begin(), probs.end(), 0.);
//   std::cout << "Prob computed = " << total_prob << std::endl;
//   
//   rj::Document doc;
//   auto &alloc = doc.GetAllocator();
//   doc.SetObject();
//   rj::Value node = rj::CubemapToJSON(cubemap, alloc);
//   doc.AddMember("cubemap", node, alloc);
//   node = rj::ContainerToJSON(probs, alloc);
//   doc.AddMember("probs", node, alloc);
//   node = rj::ContainerToJSON(sample_counts, alloc);
//   doc.AddMember("counts", node, alloc);
//   rj::Write(doc, output_filename);
// }
// 
// 
// TEST(Microfacet, VNDFViz)
// {
//   const char* prefix = "";
//   double ior_ground = 1.0/1.5;
//   char* ior_str = "D-L"; // Dense->light transmission
//   RunVisualization(BELOW, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_below_sharp_", ior_str,".json"));
//   RunVisualization(VERTICAL, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_vertical_sharp_", ior_str,".json"));
//   RunVisualization(MUCH_DEFLECTED, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_deflected_sharp_", ior_str,".json"));
//   RunVisualization(ALMOST_45DEG, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_45_sharp_", ior_str,".json"));
//   RunVisualization(BELOW, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_below_rough_",ior_str,".json"));
//   RunVisualization(VERTICAL, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_vertical_rough_",ior_str,".json"));
//   RunVisualization(MUCH_DEFLECTED, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_deflected_rough_",ior_str,".json"));
//   RunVisualization(ALMOST_45DEG, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_45_rough_",ior_str,".json"));
//   ior_ground = 1.5;
//   ior_str = "L-D"; // Light -> dense transmission
//   RunVisualization(BELOW, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_below_sharp_", ior_str,".json"));
//   RunVisualization(VERTICAL, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_vertical_sharp_", ior_str,".json"));
//   RunVisualization(MUCH_DEFLECTED, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_deflected_sharp_", ior_str,".json"));
//   RunVisualization(ALMOST_45DEG, 0.05, ior_ground, strconcat("/tmp/",prefix,"cubemap_45_sharp_", ior_str,".json"));
//   RunVisualization(BELOW, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_below_rough_",ior_str,".json"));
//   RunVisualization(VERTICAL, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_vertical_rough_",ior_str,".json"));
//   RunVisualization(MUCH_DEFLECTED, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_deflected_rough_",ior_str,".json"));
//   RunVisualization(ALMOST_45DEG, 0.5, ior_ground, strconcat("/tmp/",prefix,"cubemap_45_rough_",ior_str,".json"));
// }



namespace Microfacet
{

INSTANTIATE_TEST_CASE_P(Microfacet, ReflectTest, ::testing::Values(
  NDF_Params(0.05, VERTICAL),
  NDF_Params(0.08, MUCH_DEFLECTED), // Needs more roughness. Probably because numerical error of integration of the probability density.
  NDF_Params(0.05, BELOW),
  NDF_Params(0.5, VERTICAL),
  NDF_Params(0.5, MUCH_DEFLECTED),
  NDF_Params(0.5, BELOW)
));


INSTANTIATE_TEST_CASE_P(Microfacet, RefractTest, 
  ::testing::Combine(
    ::testing::Values(0.08, 0.5),  // Roughness
    ::testing::Values(VERTICAL, MUCH_DEFLECTED, BELOW, ALMOST_45DEG),  // Wi
    ::testing::Values(1.5, 1./1.5)  // IOR
  ));


}
