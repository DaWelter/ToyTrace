#include "gtest/gtest.h"

//#include "path_guiding.hxx"
#include "path_guiding_tree.hxx"
#include "distribution_mixture_models.hxx"

using namespace guiding;

namespace guiding { namespace kdtree {

std::pair<Box, Box> ChildrenBoxes(const Tree &tree, Handle node, const Box &node_box)
{
  auto [axis, pos] = tree.Split(node);

  Box leftbox{ node_box };
  leftbox.max[axis] = pos;
  

  Box rightbox{ node_box };
  rightbox.min[axis] = pos;
  
  return std::make_pair(leftbox, rightbox);
}


void ComputeLeafBoxes(const Tree &tree, Handle node, const Box &node_box, ToyVector<Box> &out)
{
  if (node.is_leaf)
  {
    out[node.idx] = node_box;
  }
  else
  {
    auto [left, right] = tree.Children(node);
    auto [bl, br] = ChildrenBoxes(tree, node, node_box);
    ComputeLeafBoxes(tree, left, bl, out);
    ComputeLeafBoxes(tree, right, br, out);
  }
}


}}  // guiding::tree


TEST(Guiding, KdTreeAdaptation)
{
  using namespace kdtree;

  // Tree with splits in the cell center and alternating dimension.
  // Set up so each of these points ends up in the center of a cell.
  std::array<Double3, 8> points{ {
    { -1., -1., -1. },
    { 1., -1., -1. },
    { -1., 1., -1. },
    { 1., 1., -1. },
    { -1., -1., 1. },
    { 1., -1., 1. },
    { -1., 1., 1. },
    { 1., 1., 1. },
  }};
  Box rootbox;
  rootbox.min = Double3::Constant(-2.);
  rootbox.max = Double3::Constant(2.);

  ToyVector<Box> leafboxes;
  Tree tree{};

  for (int iter = 0; iter < 3; ++iter)
  {
    leafboxes.resize(tree.NumLeafs());
    ComputeLeafBoxes(tree, tree.GetRoot(), rootbox, leafboxes);

    auto SplitDecision = [&](int cellidx)
    {
      const int axis = 2 - iter; // Due to the point ordering. Makes the tests below more convenient.
      // Split at center of cell.
      const double pos = (leafboxes[cellidx].max[axis]+leafboxes[cellidx].min[axis])*0.5;
      return std::make_pair(axis, pos);
    };

    TreeAdaptor adaptor(SplitDecision);
    tree = adaptor.Adapt(tree);

    // Since this builds a balanced tree ...
    int leaf_idx = 0;
    for (const auto &m : adaptor.GetNodeMappings())
    {
      //std::cout << strconcat("map ", leaf_idx, " -> ", m.new_first, ", ", m.new_second, "\n");
      EXPECT_EQ(m.new_first, leaf_idx*2);
      EXPECT_EQ(m.new_second, leaf_idx*2+1);
      leaf_idx++;
    }
  }

  ASSERT_EQ(tree.NumLeafs(), 8);

  leafboxes.resize(tree.NumLeafs());
  ComputeLeafBoxes(tree, tree.GetRoot(), rootbox, leafboxes);

  int idx = 0;
  for (auto &b : leafboxes)
  {
    const Double3 d = b.max - b.min;
    const Double3 c = (b.max + b.min)*0.5;
    EXPECT_NEAR(d[0], 2., 1.e-3);
    EXPECT_NEAR(d[1], 2., 1.e-3);
    EXPECT_NEAR(d[2], 2., 1.e-3);
    //std::cout << strconcat("Box", idx, " = ", c, " +/- ", d, "\n");
    EXPECT_NEAR(c[0], points[idx][0], 1.e-3);
    EXPECT_NEAR(c[1], points[idx][1], 1.e-3);
    EXPECT_NEAR(c[2], points[idx][2], 1.e-3);
    ++idx;
  }
}


TEST(Guiding, KdTreeNoopAdaptation)
{
  using namespace kdtree;

  Tree tree{};

  auto SplitDecision = [&](int cellidx)
  {
    return std::make_pair(-1, NaN);
  };

  TreeAdaptor adaptor(SplitDecision);
  tree = adaptor.Adapt(tree);

  ASSERT_EQ(tree.NumLeafs(), 1);
  auto nodemap = adaptor.GetNodeMappings();
  ASSERT_EQ(nodemap.size(), 1);
  ASSERT_EQ(nodemap[0].new_first, 0);
  ASSERT_EQ(nodemap[0].new_second, -1);
}


#if 0
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
#endif


TEST(Guiding, MovmfSampling)
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