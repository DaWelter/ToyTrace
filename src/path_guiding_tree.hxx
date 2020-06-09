#pragma once

#include "util.hxx"
#include "vec3f.hxx"
#include "box.hxx"
#include "span.hxx"
#include "json_fwd.hxx"
#include "ray.hxx"

#include <boost/container/static_vector.hpp>

namespace guiding 
{


namespace kdtree
{

inline static constexpr int MAX_DEPTH = 20;


struct AdaptParams
{
  std::uint64_t max_num_points = std::numeric_limits<std::uint64_t>::max();
};

static constexpr int MAX_NODES = (1<<30)-1;

struct Node
{
  double split_pos = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t 
    split_axis : 2, 
    left_is_leaf : 1, 
    right_is_leaf : 1,
    left_idx : 30,
    right_idx : 30;
};

static_assert(sizeof(Node) == 16);

struct Handle 
{
  int idx = -1;
  bool is_leaf = false;
};


class Tree
{
public:
  using Handle = guiding::kdtree::Handle;

private:
  friend class TreeAdaptor;
  template <class, class> friend class Builder;
  ToyVector<Node> storage;
  Handle root;
  int num_leafs = 0;
  
  Handle AllocateLeaf()
  {
    assert (num_leafs < MAX_NODES);
    return Handle{ num_leafs++, true };
  }

  Handle AllocateBranch(Handle left, Handle right, int axis, double pos)
  {
    Node nd;
    nd.split_pos = pos;
    nd.split_axis = axis;
    nd.left_is_leaf = left.is_leaf;
    nd.right_is_leaf = right.is_leaf;
    nd.left_idx = left.idx;
    nd.right_idx = right.idx;
    storage.push_back(nd);
    return { static_cast<int>(storage.size()-1), false };
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
    return std::make_pair(Handle{static_cast<int>(b.left_idx), static_cast<bool>(b.left_is_leaf)},
                          Handle{static_cast<int>(b.right_idx), static_cast<bool>(b.right_is_leaf)});
  }

  auto Split(Handle node) const
  {
    assert (!node.is_leaf);
    const auto& b = storage[node.idx];
    return std::make_pair(static_cast<int>(b.split_axis), b.split_pos);
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
        const bool go_left = p[b.split_axis] < b.split_pos;
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
    new_tree.root = AdaptRecursive(tree.root, 1);
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
  }

  Handle AdaptLeafRecursive(Tree::Handle node, int depth)
  {
    if (auto [axis, pos] = DetermineSplit(node); axis>=0 && depth<MAX_DEPTH)
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

  Handle AdaptBranchRecursive(Tree::Handle node, int depth)
  {
    auto [old_left, old_right] = tree->Children(node);
    auto [axis, pos] = tree->Split(node);
    auto left = AdaptRecursive(old_left, depth+1);
    auto right = AdaptRecursive(old_right, depth+1);
    auto branch = new_tree.AllocateBranch(left, right, axis, pos);
    return branch;
  }

  Handle AdaptRecursive(Tree::Handle node, int depth)
  {
    return node.is_leaf ? AdaptLeafRecursive(node, depth) : AdaptBranchRecursive(node, depth);
  }
};


class LeafIterator
{
  using H = Tree::Handle;
  const Tree* tree;
  Ray ray;

  struct Entry
  {
    H node;
    double tnear;
    double tfar;
  };

  boost::container::static_vector<Entry, MAX_DEPTH> stack;

public:
  LeafIterator(const Tree &tree_, const Ray &ray_, double tnear_init, double tfar_init)  noexcept
    : tree{&tree_}, ray{ray_}
    {
      stack.push_back({tree->GetRoot(), tnear_init, tfar_init});
      DecentToNextLeaf();
    }

  void operator++()  noexcept
  {
    stack.pop_back();
    if (!stack.empty())
      DecentToNextLeaf();
  }

  operator bool() const  noexcept
  {
    return !stack.empty();
  }

  struct ReturnValue {
    int idx;
    double tnear, tfar;
  };

  ReturnValue operator*() const  noexcept
  {
    auto e = stack.back();
    return ReturnValue{e.node.idx, e.tnear, e.tfar};
  }

  std::pair<double, double> Interval() const  noexcept
  {
    const auto& e = stack.back();
    return std::make_pair(e.tnear, e.tfar);
  }

  int Payload() const  noexcept
  {
    const auto& e = stack.back();
    return e.node.idx;
  }

private:

  void DecentToNextLeaf() noexcept;
};


template<class Point, class GetCoordinates>
class Builder
{
  int max_depth, min_num_points;
  Tree new_tree{ Tree::TagUninitialized() };
  GetCoordinates get_coordinates;
  ToyVector <Span<Point>> leaf_ranges;

  auto MakeCompareCoordinates(int axis) const
  {
    auto compare = [this, axis](const Point &a, const Point &b) 
    { 
      return get_coordinates(a)[axis] < get_coordinates(b)[axis]; 
    };

    return compare;
  }

  Eigen::Array<double, 3, 2> ComputeBounds(const Span<Point> pts) const
  {
    Eigen::Array<double, 3, 2> result;
    for (int axis = 0; axis < 3; ++axis)
    {  
      auto[imin, imax] = std::minmax_element(pts.begin(), pts.end(), MakeCompareCoordinates(axis));
      const double pmin = get_coordinates(*imin)[axis];
      const double pmax = get_coordinates(*imax)[axis];
      result(axis, 0) = pmin;
      result(axis, 1) = pmax;
    }
    return result;
  }

  Handle BuildRecursive(Span<Point> points, int depth)
  {
    if (points.size() < 2*min_num_points || depth >= max_depth)
    {
      return BuildLeaf(points);
    }
    else
    {
      return TryBuildBranchRecursive(points, depth);
    }
  }

  Handle BuildLeaf(Span<Point> points)
  {
      Handle h = new_tree.AllocateLeaf();
      assert(h.idx == isize(leaf_ranges));
      leaf_ranges.push_back(points);
      return h;
  }

  static double AverageIfNonDegenerate(double a, double b)
  {
    assert (a < b);
    double x = 0.5*(a+b);
    // Use b because taking a would be an error.
    return x==a ? b : x;
  }

  /* Returns
       Coordinate for which all elements in the left interval are smaller.
       Size of the left interval.
  */
  std::tuple<double, int> SplitRange(Span<Point> pts, int axis)
  {
    const auto n = pts.size();
    if (n <= 0)
      return { NaN, 0 };
    
    const auto n_split_guess = n/2;
    std::nth_element(
      pts.begin(), 
      pts.begin()+n_split_guess, 
      pts.end(),
      MakeCompareCoordinates(axis));
    
    return {
      get_coordinates(pts[n_split_guess])[axis],
      n_split_guess
    };
  }

  Handle TryBuildBranchRecursive(Span<Point> branch_points, int depth)
  {
    const Eigen::Array<double, 3, 2> bounds = ComputeBounds(branch_points);

    int axis = -1;
    (bounds.col(1) - bounds.col(0)).maxCoeff(&axis);
    assert(axis >= 0);

    const auto n = branch_points.size();
    assert(n >= 2*min_num_points);

    auto [pos, count_left] = SplitRange(branch_points, axis);
    const auto count_right = n - count_left;

    //std::cout << "split axis " << axis << " range " << bounds.row(axis) << " count " << branch_points.size() << " nleft " << count_left << std::endl;

    if (count_left >= min_num_points && count_right >= min_num_points)
    {
      auto left_range = Subspan(branch_points, 0, count_left);
      auto right_range = Subspan(branch_points, count_left, count_right);
      Handle left = BuildRecursive(left_range, depth+1);
      Handle right = BuildRecursive(right_range, depth+1);
      auto branch = new_tree.AllocateBranch(left, right, axis, pos);
      return branch;
    }
    else
    {
      Handle leaf = BuildLeaf(branch_points);
      return leaf;
    }
  }

public:
  Builder(int max_depth, int min_num_points, GetCoordinates get_coordinates)
    : max_depth{ max_depth }, min_num_points{ min_num_points }, get_coordinates{ get_coordinates }
  {
    assert (max_depth > 0);
  }

  Tree Build(Span<Point> points)
  {
    new_tree.root = BuildRecursive(points, 1);
    return std::move(new_tree);
  }

  Span<Point> DataRangeOfLeaf(int idx) const
  {
    return leaf_ranges[idx];
  }
};

template<class Point, class GetCoordinates>
inline Builder<Point, GetCoordinates> MakeBuilder(int max_depth, int min_num_points, GetCoordinates get_coordinates)
{
  return Builder<Point, GetCoordinates>(max_depth, min_num_points, get_coordinates);
}


} // kdtree


} // guiding