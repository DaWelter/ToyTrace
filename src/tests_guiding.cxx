#include "gtest/gtest.h"

#include "path_guiding.hxx"
#include "json.hxx"

using namespace guiding;

TEST(Guiding, KdTree)
{
  using namespace kdtree;
  using namespace rapidjson_util;

  std::array<Double3, 4> points{ {
    { -1., -1., 0. },
    { 1., -1., 0. },
    { -1., 1., 0. },
    { 1., 1., 0. },
  }};

  Tree tree{};

  for (int iter = 0; iter < 3; ++iter)
  {
    for (auto p : points)
    {
      tree.Lookup(p)->leaf_stats += p;
    }

    AdaptParams params{
      1, 0, // At most 1 point per leaf
    };
    TreeAdaptor(params).AdaptInplace(tree);

    for (auto *payload : tree.GetData())
    {
      payload->leaf_stats = LeafStatistics{};
    }
  }

#ifdef HAVE_JSON
  rj::Document doc(rj::kObjectType);
  tree.DumpTo(doc, doc);
  std::cerr << ToString(doc) << std::endl;
#endif

  ASSERT_EQ(tree.NumLeafs(), 4);
}


namespace 
{

void TestProjection(const Double2 &p)
{
  Double3 q = InvStereographicProjection(p);
  Double2 r = StereographicProjection(q);
  ASSERT_GE(q[2], 0.);
  ASSERT_NEAR(q.norm(), 1.0, 1.e-3);
  ASSERT_NEAR(p[0], r[0], 1.e-3);
  ASSERT_NEAR(p[1], r[1], 1.e-3);
}


} // anonymous


TEST(Guiding, StereographicProjection)
{
  TestProjection({0.999999, 0.});
  TestProjection({0., 0.999999});
  TestProjection({0.123, 0.456});
}


TEST(MOVMF, Sampling)
{
  using namespace vmf_fitting;
  VonMisesFischerMixture m;
  m.weights.setZero();
  m.weights <<
    4.92805e-08, 2.67356e-07, 2.18258e-05, 0.000542076, 7.16188e-08, 3.20299e-09, 0.997971, 5.77399e-06, 1.43747e-08, 2.79681e-07, 0.000585669, 0.000586032, 1.68318e-06, 6.35961e-08, 0.000193427, 9.20341e-05;
  //m.weights[0] = 1.f;
  m.means.setZero();
  m.means <<
    0.831414, -0.233603, 0.504163,
0.249612, 0.968303, 0.00910629,
0.0467859, 0.998452, 0.0300901,
-0.0082946, 0.999932, 0.00814839,
0.922677, 0.204906, -0.32662,
0.142239, 0.985678, 0.0905901,
0.410785, 0.892285, -0.187303,
0.0882308, 0.996085, 0.00556397,
0.684904, 0.727549, -0.0397407,
0.906872, 0.361539, 0.216503,
0.070648, 0.995644, -0.0608361,
0.0624232, 0.998019, -0.00781325,
-0.229731, -0.702705, 0.673372,
-0.154327, 0.98799, 0.00761821,
0.0524023, 0.998402, -0.0211653,
-0.0295177, 0.995306, -0.0921707;
  //m.means.row(0) = Eigen::Vector3f{1.f, 0, 0}.transpose();
  m.concentrations.setZero();
  //m.concentrations[0] = 0.291107f;
  m.concentrations << 1.89335e+06, 80052.3, 1.89335e+06, 1.96851e+06, 1.96817e+06, 5229.48, 0.291107, 1.96851e+06, 47.8656, 4.95319, 1.89369e+06, 1.9734e+06, 5236.94, 5229.57, 1.89336e+06, 1.97374e+06;
  std::array<double, 3> rnd{ 0.119874, 0.395788, 8.51584e-08 };
  auto v = vmf_fitting::Sample(m, rnd);
  std::cout << v << std::endl;
  ASSERT_TRUE(v.array().isFinite().all());
  ASSERT_NORMALIZED(v);
}