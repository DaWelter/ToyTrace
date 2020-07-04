#include "gtest/gtest.h"

//#include <range/v3/view/enumerate.hpp>

#include "path_guiding.hxx"
#include "path_guiding_tree.hxx"
#include "path_guiding_quadtree.hxx"
#include "distribution_mixture_models.hxx"
#include "json.hxx"

using namespace guiding;


namespace {

using namespace guiding::kdtree;


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


auto BuildTree(ToyVector<int> split_axes)
{
  Box rootbox;
  rootbox.min = Double3::Constant(-4.);
  rootbox.max = Double3::Constant(4.);

  ToyVector<Box> leafboxes;
  Tree tree{};
  for (int axis : split_axes)
  {
    leafboxes.resize(tree.NumLeafs());
    ComputeLeafBoxes(tree, tree.GetRoot(), rootbox, leafboxes);

    auto SplitDecision = [&](int cellidx)
    {
      const double pos = (leafboxes[cellidx].max[axis]+leafboxes[cellidx].min[axis])*0.5;
      return std::make_pair(axis, pos);
    };

    TreeAdaptor adaptor(SplitDecision);
    tree = adaptor.Adapt(tree);
  }

  leafboxes.resize(tree.NumLeafs());
  ComputeLeafBoxes(tree, tree.GetRoot(), rootbox, leafboxes);

  return std::make_pair(std::move(tree), std::move(leafboxes));
}



void CheckIterResult(const LeafIterator::ReturnValue &value, 
                     const LeafIterator::ReturnValue &expected)
{
  EXPECT_EQ(value.idx, expected.idx);
  EXPECT_NEAR(value.tnear, expected.tnear, 1.e-3);
  EXPECT_NEAR(value.tfar, expected.tfar, 1.e-3);
}

std::ostream& operator<<(std::ostream& os, const LeafIterator::ReturnValue &value)
{
  os << "{ " << value.idx << ", " << value.tnear << ", " << value.tfar << ", }";
  return os;
}


void CheckIterator(
  LeafIterator&& iter, 
  const ToyVector<LeafIterator::ReturnValue> &expected, ToyVector<int> &iterated_leafs)
{
  iterated_leafs.clear();
  int i = 0;
  for (; iter; ++iter, ++i)
  {
    ASSERT_LT(i, isize(expected));
    CheckIterResult(*iter, expected[i]);
    iterated_leafs.push_back((*iter).idx);
    std::cout << *iter << ",\n";
  }
  ASSERT_EQ(i, isize(expected));
}


void PrintBoxes(const ToyVector<Box> &boxes, const ToyVector<int> selection)
{
  std::printf("| xmin, xmax | ymin, ymax | zmin, zmax |\n");
  for (int i:selection)
  {
    const auto &b = boxes[i];
    std::printf("| %6.1f, %6.1f | %6.1f, %6.1f | %6.1f, %6.1f |\n",
      b.min[0], b.max[0], b.min[1], b.max[1], b.min[2], b.max[2]);
  }
}

}


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


TEST(Guiding, KdTreeBuilder)
{
  using namespace kdtree;
  using P = Eigen::Vector3d;

  std::array<P, 8> points{{
    { -1., -1., -1. },
    { 1., -1., -1. },
    { -1., 1., -1. },
    { 1., 1., -1. },
    { -1., -1., 1. },
    { 1., -1., 1. },
    { -1., 1., 1. },
    { 1., 1., 1. },
  }};

  auto builder = kdtree::MakeBuilder<P>(20, 1, [&](const P& p) { return p; });
  Tree tree = builder.Build(AsSpan(points));

  ASSERT_EQ(tree.NumLeafs(), points.size());
  // For now, I know that data ranges should be reordered to the depth first traversal order of the leafs
  // which equals the order in which the leafs are iterated.
  const P* expected_begin = points.data();
  for (int i=0; i<tree.NumLeafs(); ++i)
  {
    auto span = builder.DataRangeOfLeaf(i);
    EXPECT_EQ(span.begin(), expected_begin);
    expected_begin = span.end();
  }
  // Check end of last cell.
  EXPECT_EQ(
    builder.DataRangeOfLeaf(tree.NumLeafs()-1).end(),
    points.data() + points.size());
}


TEST(Guiding, KdTreeBuilderHandlesDegeneracy)
{
  using namespace kdtree;
  using P = Eigen::Vector3d;

  std::array<P, 4> points{{
    {0., 0., 0.},
    {0., 0., 0.},
    {0., 0., 0.},
    {0., 0., 0.},
  }};

  auto builder = kdtree::MakeBuilder<P>(2, 1, [&](const P& p) { return p; });
  Tree tree = builder.Build(AsSpan(points));

  ASSERT_EQ(tree.NumLeafs(), 2);
}


TEST(Guiding, KdTreeBuilderMaxDepth)
{
  using namespace kdtree;
  using P = Eigen::Vector3d;

  std::array<P, 8> points{{
    { -1., -1., -1. },
    { 1., -1., -1. },
    { -1., 1., -1. },
    { 1., 1., -1. },
    { -1., -1., 1. },
    { 1., -1., 1. },
    { -1., 1., 1. },
    { 1., 1., 1. },
  }};

  auto builder = kdtree::MakeBuilder<P>(2, 1, [&](const P& p) { return p; });
  Tree tree = builder.Build(AsSpan(points));

  ASSERT_EQ(tree.NumLeafs(), 2);
}


TEST(Guiding, KdTreeBuilderNumPoints)
{
  using namespace kdtree;
  using P = Eigen::Vector3d;

  std::array<P, 8> points{{
    { -1., -1., -1. },
    { 1., -1., -1. },
    { -1., 1., -1. },
    { 1., 1., -1. },
    { -1., -1., 1. },
    { 1., -1., 1. },
    { -1., 1., 1. },
    { 1., 1., 1. },
  }};

  auto builder = kdtree::MakeBuilder<P>(20, 4, [&](const P& p) { return p; });
  Tree tree = builder.Build(AsSpan(points));

  ASSERT_EQ(tree.NumLeafs(), 2);
}


TEST(Guiding, KdTreeIterator1)
{
  using namespace kdtree;
  
  auto [tree, leafboxes] = BuildTree({0, 0, 0});

  // using namespace rapidjson_util;
  // rj::Document doc;
  // doc.SetObject();
  // tree.DumpTo(doc, doc);
  // std::cout << ToString(doc);
  ToyVector<int> leafs;
  {
    std::cout << "going right --->" << std::endl;
    Ray r{{-2.5, 0., 0.,}, {1., 0., 0.}};
    CheckIterator(
      LeafIterator{tree, r, 0., 5.}, 
      {
        { 1, 0, 0.5, },
        { 2, 0.5, 1.5, },
        { 3, 1.5, 2.5, },
        { 4, 2.5, 3.5, },
        { 5, 3.5, 4.5, },
        { 6, 4.5, 5, }
      },
      leafs);
    PrintBoxes(leafboxes, leafs);
  }

  {
    std::cout << "going left <---" << std::endl;
    Ray r{{2.5, 0., 0.,}, {-1., 0., 0.}};
    CheckIterator(
      LeafIterator{tree, r, 0., 5.}, 
      {
        { 6, 0, 0.5, },
        { 5, 0.5, 1.5, },
        { 4, 1.5, 2.5, },
        { 3, 2.5, 3.5, },
        { 2, 3.5, 4.5, },
        { 1, 4.5, 5, }
      }, leafs);
    PrintBoxes(leafboxes, leafs);
  }
}


TEST(Guiding, KdTreeIterator2)
{
  using namespace kdtree;
  
  auto [tree, leafboxes] = BuildTree({0, 0, 0});
  ToyVector<int> leafs;

  {
    std::cout << "left side -----" << std::endl;
    Ray r{{-0.1, 0., 0.,}, {0., 1., 0.}};
    CheckIterator(
      LeafIterator{tree, r, 0., 5.}, 
      {
        { 3, 0, 5, }
      }, leafs);
    PrintBoxes(leafboxes, leafs);
  }

  {
    std::cout << "center -----" << std::endl;
    Ray r{{0., 0., 0.,}, {0., 1., 0.}};
    CheckIterator(
      LeafIterator{tree, r, 0., 5.}, 
      {
        { 3, 0, 5, }
      }, leafs);
    PrintBoxes(leafboxes, leafs);
  }

  {
    std::cout << "right side -----" << std::endl;
    Ray r{{0.1, 0., 0.,}, {0., 1., 0.}};
    CheckIterator(
      LeafIterator{tree, r, 0., 5.}, 
      {
        { 4, 0, 5, },
      }, leafs);
    PrintBoxes(leafboxes, leafs);
  }
}


TEST(Guiding, KdTreeIterator3)
{
  using namespace kdtree;
  
  auto [tree, leafboxes] = BuildTree({1, 2, 0, 0, 0});
  ToyVector<int> leafs;

  {
    std::cout << " straight" << std::endl;
    ToyVector<int> iterated_leafs;
    
    Ray r{{-2., 0., 0.,}, {1., 0., 0.}};
    CheckIterator(
      LeafIterator{tree, r, 0., 5.}, 
      {
        { 2, 0, 1, },
        { 3, 1, 2, },
        { 4, 2, 3, },
        { 5, 3, 4, },
        { 6, 4, 5, }
      }, leafs);
    PrintBoxes(leafboxes, leafs);
  }

  {
    std::cout << " diagonal" << std::endl;
    ToyVector<int> iterated_leafs;
    
    Ray r{{-2., -1., 0.,}, {1., 0.5, 0.}};
    CheckIterator(
      LeafIterator{tree, r, 0., 5.}, 
      {
        { 2, 0, 1, },
        { 3, 1, 2, },
        { 20, 2, 3, },
        { 21, 3, 4, },
        { 22, 4, 5, },
      }, leafs);
    PrintBoxes(leafboxes, leafs);
  }
}


TEST(Guiding, CombinedIntervalsIterator)
{
  class DummyIter
  {
      Span<double> boundaries;
      int end_idx = 1;
    public:
      explicit DummyIter(Span<double> boundaries)
        : boundaries{boundaries} {}
      
      operator bool() const { return end_idx < isize(boundaries); }

      auto Interval() const { return std::make_pair(boundaries[end_idx-1], boundaries[end_idx]); }

      auto operator*() const {
        return end_idx-1;
      }

      void operator++() {
        ++end_idx;
      }
  };

  ToyVector<double> boundaries1{1., 2.,     4.,              8., 9., 10., 11. };
  ToyVector<double> boundaries2{0., 2., 3.,     5., 6., 7.,          10.      };
  ToyVector<std::tuple<double, double, int , int>> expected({
    { 1., 2. , 0, 0},
    { 2., 2. , 1, 0},
    { 2., 3. , 1, 1},
    { 3., 4. , 1, 2},
    { 4., 5. , 2, 2},
    { 5., 6. , 2, 3},
    { 6., 7. , 2, 4},
    { 7., 8. , 2, 5},
    { 8., 9. , 3, 5},
    { 9., 10., 4, 5},
    { 10.,10., 5, 5},
  });

  CombinedIntervalsIterator<DummyIter, DummyIter> it{
    DummyIter{AsSpan(boundaries1)},
    DummyIter{AsSpan(boundaries2)}
  };

  int n = 0;
  auto iter_expected = expected.begin();
  for (; it; ++it, ++iter_expected, ++n)
  {
    auto i = it.Interval();
    auto v1 = it.DereferenceFirst();
    auto v2 = it.DereferenceSecond();
    std::string state = fmt::format("@  state N={} I0={} I1={}", n, v1, v2);
    EXPECT_EQ(i.first , std::get<0>(*iter_expected)) << state;
    EXPECT_EQ(i.second, std::get<1>(*iter_expected)) << state;
    EXPECT_EQ(v1, std::get<2>(*iter_expected)) << state;
    EXPECT_EQ(v2, std::get<3>(*iter_expected)) << state;
  }
}



TEST(Guiding, QuadTreeBuilderTrivial)
{
  using Vector = Eigen::Vector2f;
  quadtree::Builder builder{ Span<Vector>{}, Span<float>{}, 1. };
  auto tree = builder.ExtractTree();
  EXPECT_EQ(tree.NumNodes(), 1);
  EXPECT_TRUE(tree.GetRoot().is_leaf);
}

// TODO:
// Test that nodes are ordered in DFS order.

TEST(Guiding, QuadTreeDivideIntoChildRanges)
{
  using Vector = Eigen::Vector2f;
  ToyVector<Vector> points;
  ToyVector<float> weights;
  Sampler sampler{};

  points = { {0.25, 0.25}, { 0.75, 0.25 }, {0.25, 0.75 }, { 0.75, 0.75 } };
  for (int i = 0; i < 4; ++i)
  {
    // Add more points, copied from the initial ones. Add different amounts.
    for (int j = 0; j < i; ++j)
      points.push_back(points[i]);
  }

  RandomShuffle(points.begin(), points.end(), sampler);
  weights.resize(points.size(), 1.f);

  {
    quadtree::Region r;
    auto childranges = r.DivideIntoChildRange(AsSpan(points), AsSpan(weights));
    EXPECT_EQ(childranges[0].second, 1);
    EXPECT_EQ(childranges[1].second, 2);
    EXPECT_EQ(childranges[2].second, 3);
    EXPECT_EQ(childranges[3].second, 4);

    EXPECT_EQ(childranges[0].first, 0);
    EXPECT_EQ(childranges[1].first, 4);
    EXPECT_EQ(childranges[2].first, 1);
    EXPECT_EQ(childranges[3].first, 6);
  }
}


TEST(Guiding, QuadTreeBuilderConsistency)
{
  using Vector = Eigen::Vector2f;
  ToyVector<Vector> points;
  ToyVector<float> weights;
  Sampler sampler{};

  constexpr int N = 100;
  for (int i = 0; i < N; ++i)
  {
    points.push_back(sampler.UniformUnitSquare().cast<float>());
    weights.push_back(1.);
  }

  const float thres = 1.00001f / points.size();

  quadtree::Builder builder{ AsSpan(points), AsSpan(weights), thres };
  auto tree = builder.ExtractTree();
  EXPECT_FALSE(tree.GetRoot().is_leaf);

  // Generate this map ...
  std::unordered_map<int, ToyVector<int>> node_to_points;
  int point_idx = 0;
  for (auto p : points)
  {
    quadtree::DecentHelper d{ p, tree };
    for(;;)
    {
      node_to_points[d.GetNode().idx].push_back(point_idx);
      if (d.IsLeaf())
        break;
      d.TraverseInplace();
    }
    ++point_idx;
  }

  //// Compare with point ranges from the builder
  //for (int node_idx=0; node_idx<tree.NumNodes(); ++node_idx)
  //{
  //  auto[offset, size] = builder.DataRangeOfNode(node_idx);
  //  ToyVector<int> expected(size); std::iota(expected.begin(), expected.end(), offset);
  //  auto in_node = node_to_points[node_idx];
  //  std::sort(in_node.begin(), in_node.end());
  //  ASSERT_EQ(in_node.size(), expected.size());
  //  ASSERT_TRUE(std::equal(in_node.begin(), in_node.end(), expected.begin()));
  //}
}


TEST(Guiding, TreeAdaptor)
{
  quadtree::Tree tree;
  ASSERT_EQ(tree.NumNodes(), 1);
  ToyVector<float> node_weights(tree.NumNodes(), 16.f); 

  {
    quadtree::TreeAdaptor adaptor{ tree, AsSpan(node_weights), 5.f / 16.f };
    tree = adaptor.ExtractTree();
    node_weights = adaptor.ExtractWeights();
    ASSERT_EQ(tree.NumNodes(), 1 + 4);
    ASSERT_EQ(node_weights.size(), tree.NumNodes());
  }

  {
    quadtree::TreeAdaptor adaptor{ tree, AsSpan(node_weights), 2.f / 16.f };
    tree = adaptor.ExtractTree();
    node_weights = adaptor.ExtractWeights();
    ASSERT_EQ(tree.NumNodes(), 1 + 4 + 16 );
    ASSERT_EQ(node_weights.size(), tree.NumNodes());
  }

  quadtree::PushWeight(tree, AsSpan(node_weights), { 0.75, 0.75 }, 10.f);

  const float expected_total_weight = 16.f + 10.f;
  EXPECT_NEAR(node_weights[tree.GetRoot().idx], expected_total_weight, 1.e-3f);
}


TEST(Guiding, MovmfSampling)
{
  using namespace vmf_fitting;
  VonMisesFischerMixture<16> m;
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
  const double vmag = v.norm();
  ASSERT_LE(0.99, vmag);
  ASSERT_LE(vmag, 1.01);
}


TEST(Guiding, PolarMap)
{
  using namespace guiding::quadtree_radiance_distribution;
  static constexpr int N = 10;
  Sampler sampler;
  for (int i=0; i<N; ++i)
  {
    const Eigen::Vector2f x = sampler.UniformUnitSquare().cast<float>();
    const Eigen::Vector2f y = MapSphereToTree(MapTreeToSphere(x));
    EXPECT_NEAR(x[0], y[0], 1.e-3);
    EXPECT_NEAR(x[1], y[1], 1.e-3);
  }
}