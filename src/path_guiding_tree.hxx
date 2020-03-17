#pragma once

#include "util.hxx"
#include "vec3f.hxx"
#include "box.hxx"
#include "span.hxx"
#include "json_fwd.hxx"

namespace guiding 
{


namespace kdtree
{


struct AdaptParams
{
  std::uint64_t max_num_points = std::numeric_limits<std::uint64_t>::max();
};

static constexpr int MAX_NODES = std::numeric_limits<short>::max();

struct Node
{
  double split_pos = std::numeric_limits<double>::quiet_NaN();
  short left_idx;
  short right_idx;

  struct {
    unsigned char split_axis : 2, left_is_leaf : 1, right_is_leaf;

  } flags;
};


struct Handle 
{
  short idx = -1;
  unsigned char is_leaf = false;
};


class Tree
{
public:
  using Handle = guiding::kdtree::Handle;

private:
  friend class TreeAdaptor;
  ToyVector<Node> storage;
  Handle root;
  int num_leafs = 0;
  
  Handle AllocateLeaf()
  {
    assert (num_leafs < MAX_NODES);
    return Handle{ (short)num_leafs++, true };
  }

  Handle AllocateBranch(Handle left, Handle right, int axis, double pos)
  {
    Node nd {
      pos,
      left.idx,
      right.idx,
      {}
    };
    nd.flags.split_axis = axis;
    nd.flags.left_is_leaf = left.is_leaf;
    nd.flags.right_is_leaf = right.is_leaf;
    storage.push_back(nd);
    return { (short)(storage.size()-1), false };
  }

  struct TagUninitialized {};

  explicit Tree(TagUninitialized)
  {
  }

public:
  Tree() :
    Tree(TagUninitialized{})
  {
    root = AllocateLeaf();
  }

  Tree(const Tree &) = delete;
  Tree& operator=(const Tree&) = delete;

  Tree(Tree &&) = default;
  Tree& operator=(Tree &&) = default;

  int NumLeafs() const { return num_leafs; }

  Handle GetRoot() const
  {
    return root;
  }

  auto Children(Handle node) const
  {
    assert (!node.is_leaf);
    const auto& b = storage[node.idx];
    return std::make_pair(Handle{b.left_idx, b.flags.left_is_leaf},
                          Handle{b.right_idx, b.flags.right_is_leaf});
  }

  auto Split(Handle node) const
  {
    assert (!node.is_leaf);
    const auto& b = storage[node.idx];
    return std::make_pair(b.flags.split_axis, b.split_pos);
  }

  int Lookup(const Double3 &p)
  {
    Handle current = root;
    while (true)
    {
      if (current.is_leaf)
      {
        return current.idx;
      }
      else
      {
        const auto& b = storage[current.idx];
        const bool go_left = p[b.flags.split_axis] < b.split_pos;
        auto [left, right] = Children(current);
        current = go_left ? left : right;
      }
    }
  }

  int Lookup(const Double3 &p) const
  {
    return const_cast<Tree*>(this)->Lookup(p);
  }

#ifdef HAVE_JSON
  void DumpTo(rapidjson::Document &doc, rapidjson::Value & parent) const;
#endif
};


class TreeAdaptor
{
public:
  using SplitDecision = std::function<std::pair<int, double>(int)>;

  TreeAdaptor(SplitDecision split_decision_) :
    split_decision{split_decision_}
  {
  }

  Tree Adapt(Tree &tree)
  {
    this->tree = &tree;
    new_tree.root = AdaptRecursive(tree.root);
    Tree tmp{Tree::TagUninitialized()};
    std::swap(tmp, new_tree);
    return tmp;
  }

  struct OldToNew
  {
    //int old = -1;
    int new_first = -1;
    int new_second = -1;
  };

  Span<const OldToNew> GetNodeMappings() const
  {
    return AsSpan(node_mappings); 
  }

private:
  SplitDecision split_decision;
  Tree *tree = nullptr;
  Tree new_tree{Tree::TagUninitialized()};
  ToyVector<OldToNew> node_mappings;


  void AddToTranslationMap(Tree::Handle old, Tree::Handle new_first, Tree::Handle new_second)
  {
    node_mappings.push_back({
      /*old.idx,*/ new_first.idx, new_second.idx
    });
    assert(node_mappings.size() == old.idx+1);
    assert(old.is_leaf);
  }


  // Axis and location
  std::pair<int, double> DetermineSplit(Handle p)
  {
    assert (p.is_leaf);
    return split_decision(p.idx);
    // int axis = -1;
    // double pos = NaN;
    // const auto& stats = GetPayload(p).leaf_stats;
    // if (stats.Count())
    // {
    //   const auto mean = stats.Mean();
    //   const auto var = stats.Var();
    //   var.maxCoeff(&axis);
    //   pos = mean[axis];
    // }
    // return { axis, pos };
  }

  Handle AdaptLeafRecursive(Tree::Handle node)
  {
    if (auto [axis, pos] = DetermineSplit(node); axis>=0)
    {
      auto left = new_tree.AllocateLeaf();
      auto right = new_tree.AllocateLeaf();
      auto branch = new_tree.AllocateBranch(left, right, axis, pos);
      AddToTranslationMap(node, left, right);
      return branch;
    }
    else
    {
      auto clone = new_tree.AllocateLeaf();
      AddToTranslationMap(node, clone, {});
      return clone;
    }
  }

  Handle AdaptBranchRecursive(Tree::Handle node)
  {
    auto [old_left, old_right] = tree->Children(node);
    auto [axis, pos] = tree->Split(node);
    auto left = AdaptRecursive(old_left);
    auto right = AdaptRecursive(old_right);
    auto branch = new_tree.AllocateBranch(left, right, axis, pos);
    return branch;
  }

  Handle AdaptRecursive(Tree::Handle node)
  {
    return node.is_leaf ? AdaptLeafRecursive(node) : AdaptBranchRecursive(node);
  }
};


} // kdtree


} // guiding