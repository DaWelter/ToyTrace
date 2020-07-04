#pragma once

#include "util.hxx"
#include "vec3f.hxx"
#include "box.hxx"
#include "span.hxx"
#include "json_fwd.hxx"
#include "ray.hxx"


namespace guiding
{


namespace quadtree
{


namespace detail
{

static constexpr int MAX_NODES = (1 << 16) - 2; // -2 because one value is used to indicate that no child exists.
static constexpr int MAX_DEPTH = 8;
static constexpr std::uint64_t LEAF_CODE = ~0ul;
static constexpr std::uint16_t CHILD_LEAF = (std::uint16_t)~0ul;
namespace detail { class Builder; };

struct Node
{
  // Type punning, used to check if node is leaf. code
  // UB by the standard, I guess. But every compiler should support it.
  union {
    std::uint16_t children[4];
    std::uint64_t code = LEAF_CODE;
  };
};

static_assert(sizeof(Node) == 8);

struct Handle
{
  int idx = -1;
  bool is_leaf = false;
};


class Tree
{
public:
  using Handle = guiding::quadtree::detail::Handle;

private:
  friend class TreeAdaptor;
  friend class Builder;
  ToyVector<Node> storage;
  Handle root;

  Handle AllocateNode(bool leaf=true)
  {
    assert(storage.size() < MAX_NODES);
    storage.push_back(Node{});
    return Handle{ isize(storage) - 1, leaf };
  }

  void SetChildren(Handle &node, Handle* children)
  {
    for (int i = 0; i < 4; ++i)
    {
      assert(children[i].idx < MAX_NODES);
      storage[node.idx].children[i] = children[i].idx;
    }
  }

  struct TagUninitialized {};

  explicit Tree(TagUninitialized)
  {
  }

  static bool IsLeaf(const Node &nd)
  {
    return nd.code == LEAF_CODE;
  }

public:
  Tree() :
    Tree(TagUninitialized{})
  {
    root = AllocateNode();
  }

  Tree(const Eigen::ArrayX4i &children, int root_idx);

  int NumNodes() const { return isize(storage); }

  bool IsLeaf(int node_idx) const 
  {
    return IsLeaf(storage[node_idx]);
  }

  Handle GetRoot() const
  {
    return root;
  }

  auto Child(Handle node, int num) const
  {
    assert(!node.is_leaf);
    const auto& b = storage[node.idx];
    const auto& c = storage[b.children[num]];
    return Handle{ static_cast<int>(b.children[num]), IsLeaf(c) };
  }

  auto GetChildren(Handle node) const
  {
    return util::GenerateArray<4>([this, node](size_t i) -> Handle {
      return Child(node, (int)i);
    });
  }

  rapidjson::Value ToJSON(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> &a) const;
};


using Point = Eigen::Vector2f;
using Points = Span<Point>;
using Weights = Span<float>;


class Region
{
  Point region[2]{ Eigen::zero, Eigen::ones }; // min and max coordinate values
public:
  /*
  Child ids in binary:
  00 : bottom left
  01 : bottom right
  10 : top left
  11 : top right
  */
  void SetToChild(int child_id)
  {
    const auto refp = Center();
    region[1 - (child_id & 1)][0] = refp[0];
    region[1 - ((child_id & 2) >> 1)][1] = refp[1];
  }

  int GetChildNum(const Point &p) const
  {
    const auto refp = Center();
    return (p[0] < refp[0] ? 0 : 1) | (p[1] < refp[1] ? 0 : 2);
  }

  Point Center() const {
    return 0.5*(region[0] + region[1]);
  }

  Region Child(int child_id) const
  {
    Region tmp{ *this };
    tmp.SetToChild(child_id);
    return tmp;
  }

  const Point* Bounds() const
  {
    return region;
  }

  std::array<std::pair<ptrdiff_t, ptrdiff_t>, 4> DivideIntoChildRange(Points points, Weights weights) const;
};


 class DecentHelper
 {
   Point p; // query point
   Handle node{};
   const Tree* tree{nullptr};
   Region region;

 public:
   explicit DecentHelper(const Point p, const Tree &tree) : p{p}, node{tree.GetRoot()}, tree{&tree} {}

   void TraverseInplace()
   {
     assert(!node.is_leaf);
     const int child_num = region.GetChildNum(p);
     region.SetToChild(child_num);
     node = tree->Child(node, child_num);
   }

   bool IsLeaf() const
   {
     return node.is_leaf;
   }

   Handle GetNode() const
   {
     return node;
   }

   auto Bounds() const {
     return region.Bounds();
   }
 };


class TreeAdaptor
{
public:
  TreeAdaptor(const Tree &tree, Weights weights, float weight_fraction_threshold) :
    all_weights{ weights }, tree{ &tree }
  {
    weight_threshold = weight_fraction_threshold * weights[tree.GetRoot().idx];
    new_weights.reserve(tree.NumNodes());
    new_tree.root = AdaptRecursive(tree.root, 1);
  }

  Tree ExtractTree()
  {
    return std::move(new_tree);
  }

  ToyVector<float> ExtractWeights()
  {
    return std::move(new_weights);
  }

  struct NewToOld
  {
    int idx = -1;
    float fractional_size = 1.f; // How much of the original nodes width. Less than 1 in case a leaf was split.
  };

  Span<const NewToOld> GetNewToOldMapping() const
  {
    return AsSpan(new_to_old);
  }

private:
  Tree new_tree{ Tree::TagUninitialized() };
  ToyVector<float> new_weights;
  ToyVector<NewToOld> new_to_old;
  Weights all_weights;
  const Tree *tree = nullptr;
  float weight_threshold{ NaN };

  // Takes the weight of the original leaf as input
  Handle SplitLeaf(float weight, int depth, NewToOld old)
  {
    // Subdivides until weights are lower than threshold. 
    // Divides weight evently among children. This is like in the paper
    if (weight < weight_threshold || depth >= MAX_DEPTH)
    {
      return AllocateWeightedNode(weight, old);
    }
    else
    {
      auto branch = AllocateWeightedNode(weight, old, false);
      auto children = util::GenerateArray<4>([&](size_t i) {
        return SplitLeaf(weight * 0.25f, depth+1, { old.idx, 0.5f*old.fractional_size });
      });
      new_tree.SetChildren(branch, children.data());
      return branch;
    }
  }

  Handle SplitBranch(Tree::Handle node, int depth)
  {
    auto branch = AllocateWeightedNode(all_weights[node.idx], {node.idx, 1.f}, /*leaf=*/false);
    auto children = util::GenerateArray<4>([&](size_t i) {
      return AdaptRecursive(tree->Child(node, static_cast<int>(i)), depth + 1);
    });
    new_tree.SetChildren(branch, children.data());
    return branch;
  }

  Handle AdaptRecursive(Tree::Handle node, int depth)
  {
    const float weight = all_weights[node.idx];
    if (depth < MAX_DEPTH && weight > weight_threshold)
    {
      if (node.is_leaf)
        return SplitLeaf(weight, depth, { node.idx, 1.f});
      else
        return SplitBranch(node, depth);
    }
    else
    {
      auto leaf = AllocateWeightedNode(weight, { node.idx, 1.f});
      return leaf;
    }
  }

  Handle AllocateWeightedNode(float weight, NewToOld old, bool leaf=true)
  {
    auto node = new_tree.AllocateNode(leaf);
    if (new_weights.size() <= node.idx)
    {
      util::PushBackToEnsureSize<float>(new_weights, node.idx+1, 0.);
      util::PushBackToEnsureSize(new_to_old, node.idx+1, {});
    }
    new_weights[node.idx] = weight;
    new_to_old[node.idx] = old;
    return node;
  }
};



class Builder
{
  Tree tree;
  //ToyVector<std::pair<ptrdiff_t, ptrdiff_t>> ranges; // Offset and size into points/weights arrays.
  Points all_points;
  Weights all_weights;
  ToyVector<float> node_weights;
  float weight_threshold = NaN;

  Handle BuildRecursive(Points points, Weights weights, const Region &region, int depth);

  Handle AllocateWeightedNode(Points points, bool leaf, float weight);

public:
  Builder(Points points, Weights weights, float weight_fraction_threshold) :
    tree{ Tree::TagUninitialized{} }, all_points{ points }, all_weights{ weights }
  {
    weight_threshold = weight_fraction_threshold * util::AsEigenArray(weights).sum();

    tree.root = BuildRecursive(points, weights, Region{}, 1);
  }

  Tree ExtractTree()
  {
    return std::move(tree);
  }

  ToyVector<float> ExtractWeights()
  {
    return std::move(node_weights);
  }
};

void PushWeight(const Tree &tree, Span<float> node_weights, const Point &p, float w);
void PropagateLeafWeightsToParents(const Tree &tree, Span<float> node_weights, Handle node);
ToyVector<Eigen::Array<double, 4, 1>> GenerateQuads(const Tree &tree);
ToyVector<int> GenerateLevels(const Tree &tree);
std::pair<Point, float> Sample(const Tree &tree, Span<const float> node_weights, Sampler &sampler);
float Pdf(const Tree &tree, Span<const float> node_weights, Point pt);
Handle FindNode(const Tree &tree, const Point &pt);

} // detail


using detail::Tree;
using detail::TreeAdaptor;
using detail::Builder;
using detail::Region;
using detail::DecentHelper;
using detail::PushWeight;
using detail::GenerateQuads;
using detail::GenerateLevels;
using detail::Pdf;
using detail::Sample;
using detail::PropagateLeafWeightsToParents;
using detail::FindNode;

} // quadtree

} // namespace guiding