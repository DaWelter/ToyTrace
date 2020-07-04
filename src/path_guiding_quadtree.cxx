#include "path_guiding_quadtree.hxx"
#include "sampler.hxx"

#ifdef HAVE_JSON
#include "json.hxx"
#include "rapidjson/document.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/writer.h"
#endif

#include <numeric>


namespace guiding::quadtree::detail
{


Tree::Tree(const Eigen::ArrayX4i &children, int root_idx)
{
  storage.reserve(children.rows());
  for (int i=0; i<children.rows(); ++i)
  {
    Node nd;
    for (int j=0; j<4; ++j)
      nd.children[j] = children(i,j);
    storage.push_back(nd);
  }
  root = {root_idx, IsLeaf(storage[root_idx])};
}


Handle Builder::BuildRecursive(Points points, Weights weights, const Region & region, int depth)
{
  float weight_sum = util::AsEigenArray(weights).sum();
  if (weight_sum > weight_threshold && depth < MAX_DEPTH)
  {
    std::array<std::pair<ptrdiff_t, ptrdiff_t>, 4> ranges = region.DivideIntoChildRange(points, weights);

    auto node = AllocateWeightedNode(points, false, weight_sum);

    std::array<Handle, 4> children = util::GenerateArray<4>([&](size_t i) -> Handle {
      return BuildRecursive(Subspan(points, ranges[i]), Subspan(weights, ranges[i]), region.Child(static_cast<int>(i)), depth+1);
    });

    tree.SetChildren(node, children.data());
    return node;
  }
  else {
    return AllocateWeightedNode(points, true, weight_sum);
  }
}

Handle Builder::AllocateWeightedNode(Points points, bool leaf, float weight)
{
  Handle node = tree.AllocateNode(leaf);
  if (node_weights.size() <= node.idx)
    util::PushBackToEnsureSize(node_weights, node.idx + 1, 0.f);
  node_weights[node.idx] = weight;
  return node;
}

std::array<std::pair<ptrdiff_t, ptrdiff_t>, 4> Region::DivideIntoChildRange(Points points, Weights weights) const
{
  const auto refp = this->Center();
  ToyVector<ptrdiff_t> indices(points.size()); std::iota(indices.begin(), indices.end(), 0);
  auto it_xsplit = std::partition(indices.begin(), indices.end(), [&](int i) {
    return points[i][0] < refp[0];
  });
  auto it_ysplit_left = std::partition(indices.begin(), it_xsplit, [&](int i) {
    return points[i][1] < refp[1];
  });
  auto it_ysplit_right = std::partition(it_xsplit, indices.end(), [&](int i) {
    return points[i][1] < refp[1];
  });

  ReorderInplace(points, AsSpan(indices));
  ReorderInplace(weights, AsSpan(indices));

  std::array<std::pair<ptrdiff_t, ptrdiff_t>, 4> ranges = { {
      // Must be in the correct order.
    { 0l,                              it_ysplit_left - indices.begin() }, // bottom left
  { it_xsplit - indices.begin(),    it_ysplit_right - it_xsplit }, // bottom right
  { it_ysplit_left - indices.begin(), it_xsplit - it_ysplit_left }, // top left
  { it_ysplit_right - indices.begin(), indices.end() - it_ysplit_right } // top right
    } };
  return ranges;
}

void PushWeight(const Tree & tree, Span<float> node_weights, const Point & p, float w)
{
  DecentHelper d{ p, tree };
  for (;;)
  {
    node_weights[d.GetNode().idx] += w;
    if (d.IsLeaf())
      break;
    d.TraverseInplace();
  }
}

void PropagateLeafWeightsToParents(const Tree & tree, Span<float> node_weights, Handle node)
{
  if (node.is_leaf)
    return;
  auto& w = node_weights[node.idx] = 0.f;
  for (int i = 0; i < 4; ++i)
  {
    auto c = tree.Child(node, i);
    PropagateLeafWeightsToParents(tree, node_weights, c);
    w += node_weights[c.idx];
  }
}

ToyVector<Eigen::Array<double, 4, 1>> GenerateQuads(const Tree &tree)
{
  ToyVector<Eigen::Array<double, 4, 1>> result;
  result.reserve(tree.NumNodes());

  std::function<void(const Tree &, Handle, const Region &)> recursion;
  recursion = [&](const Tree &tree, Handle n, const Region &r) 
  {
    auto *b = r.Bounds();
    result.push_back({ b[0][0], b[0][1], b[1][0], b[1][1] }); // first min then max
    if (n.is_leaf)
      return;
    for (int i=0; i<4; ++i)
    {
      recursion(tree, tree.Child(n, i), r.Child(i));
    }
  };

  recursion(tree, tree.GetRoot(), Region{});
  return result;
}

ToyVector<int> GenerateLevels(const Tree &tree)
{
  ToyVector<int> result; result.reserve(tree.NumNodes());

  std::function<void(const Tree &, Handle, int depth)> recursion;
  recursion = [&](const Tree &tree, Handle n, int depth)
  {
    result.push_back(depth);
    if (n.is_leaf)
      return;
    for (int i = 0; i < 4; ++i)
    {
      recursion(tree, tree.Child(n, i), depth+1);
    }
  };

  recursion(tree, tree.GetRoot(), 0);
  return result;
}


Handle GetLeaf(const Tree &tree, Point pt)
{
  DecentHelper d{pt, tree};
  for (;!d.IsLeaf();)
  {
    d.TraverseInplace();
  }
  return d.GetNode();
}


std::pair<Point, float> Sample(const Tree &tree, Span<const float> node_weights, Sampler &sampler)
{
#if 1
  /* from Section 5.1. in
  Clarberg et al. (2005) "Wavelet Importance Sampling: Efficiently Evaluating Products of Complex Functions".
  */
  auto node = tree.GetRoot();
  const float less_than_1 = std::nextafter(1.f, 0.f);

  Eigen::Array2f rndnums = sampler.UniformUnitSquare().cast<float>();
  assert(rndnums[0] >= 0.f && rndnums[0] <= 1.001f);
  assert(rndnums[1] >= 0.f && rndnums[1] <= 1.001f);
  rndnums[0] = std::min(rndnums[0], less_than_1);
  rndnums[1] = std::min(rndnums[1], less_than_1);

  Region region;

  for (;!node.is_leaf;)
  {
    std::array<float, 4> weights = util::GenerateArray<4>([&](size_t i) -> float {
      return node_weights[tree.Child(node, (int)i).idx];
    });

    // marginalized over the vertical direction
    const float w_left = (weights[0] + weights[2]);
    const float w_right = (weights[1] + weights[3]);
    const float scaled_rnd0 = rndnums[0] * (w_left+w_right);
    int child_idx = (scaled_rnd0 < w_left) ? 0 : 1;
    // For the next round the random number is scaled back to lie in [0,1]
    rndnums[0] = (scaled_rnd0 < w_left) ?
      (scaled_rnd0 / (w_left + EpsilonFloat)) :
      ((scaled_rnd0 - w_left) / (w_right + EpsilonFloat));
    // Down below I consider conditional probabilities
    // w.r.t either left or right side. Here I have the "normalization"
    // constant.

    // Now we know if left or right side.
    // Hence determine if top or bottom
    const float w_bottom = child_idx ? weights[1] : weights[0];
    const float w_top = child_idx ? weights[3] : weights[2];
    const float scaled_rnd1 = rndnums[1] * (w_bottom + w_top);
    child_idx |= (scaled_rnd1 < w_bottom) ? 0 : 2;
    // Again the scaling
    rndnums[1] = (scaled_rnd1 < w_bottom) ?
      (scaled_rnd1 / (w_bottom + EpsilonFloat)) :
      ((scaled_rnd1 - w_bottom) / (w_top + EpsilonFloat));
    
    assert(rndnums[0] >= 0.f && rndnums[0] <= 1.001f);
    assert(rndnums[1] >= 0.f && rndnums[1] <= 1.001f);

    rndnums[0] = std::min(rndnums[0], less_than_1);
    rndnums[1] = std::min(rndnums[1], less_than_1);

    assert(weights[child_idx] > 0.f);
    node = tree.Child(node, child_idx);
    region.SetToChild(child_idx);
  }

  const Point area_size = region.Bounds()[1]-region.Bounds()[0];
  const float pdf = node_weights[node.idx]/(node_weights[tree.GetRoot().idx]*area_size.prod());

  assert (pdf > 0. && std::isfinite(pdf));
  const Eigen::Vector2f pt = 
    (area_size.cwiseProduct(rndnums.matrix()) + region.Bounds()[0])
    .cwiseMin(region.Bounds()[1]*less_than_1)
    .cwiseMax(region.Bounds()[0]);

  assert (GetLeaf(tree, pt).idx == node.idx);
  return { pt, pdf };
#else

  auto node = tree.GetRoot();
  Region region;

  for (;!node.is_leaf;)
  {
    std::array<float,4> weights = util::GenerateArray<4>([&](size_t i) {
      return node_weights[tree.Child(node, (int)i).idx];
    });
    int child = TowerSampling<4>(weights.data(), (float)sampler.Uniform01()*node_weights[node.idx]);
    node = tree.Child(node, child);
    region.SetToChild(child);
  }

  const Point area_size = region.Bounds()[1]-region.Bounds()[0];
  const float pdf = node_weights[node.idx]/(node_weights[tree.GetRoot().idx]*area_size.prod());
  const Eigen::Vector2f pt = area_size.cwiseProduct(sampler.UniformUnitSquare().cast<float>()) + region.Bounds()[0];
  return { pt, pdf };
#endif
}


float Pdf(const Tree &tree, Span<const float> node_weights, Point pt)
{
  DecentHelper d{pt, tree};

  for (;!d.IsLeaf();)
  {
    d.TraverseInplace();
  }

  const Point area_size = d.Bounds()[1]-d.Bounds()[0];
  const float pdf = node_weights[d.GetNode().idx]/(node_weights[tree.GetRoot().idx]*area_size.array().prod());
  return pdf;
}


Handle FindNode(const Tree &tree, const Point &pt)
{
  DecentHelper d{pt, tree};
  for (;!d.IsLeaf();)
  {
    d.TraverseInplace();
  }
  return d.GetNode();
}


rapidjson::Value Tree::ToJSON(rapidjson_util::Alloc &a) const
{
  using namespace rapidjson_util;
  
  rj::Value jchildren(rj::kArrayType);
  for (int i=0; i<4; ++i)
  {
    ToyVector<int> nodes = util::TransformVector(storage, [&](const Node &n) { return (int)n.children[i]; });
    jchildren.PushBack(rapidjson_util::ToJSON(nodes, a), a);
  }
  rj::Value jtree(rj::kObjectType);
  jtree.AddMember("children", jchildren, a);
  jtree.AddMember("root", (int)root.idx, a);
  return jtree;
}


} //namespace guiding::quadtree::detail