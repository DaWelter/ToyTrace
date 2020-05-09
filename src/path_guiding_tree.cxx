#include "path_guiding_tree.hxx"

#ifdef HAVE_JSON
#include "json.hxx"
#include "rapidjson/document.h"
#endif


namespace guiding
{


namespace kdtree
{


void LeafIterator::DecentToNextLeaf()  noexcept
{
  const Double3 o = ray.org;
  const Double3 d = ray.dir;

  while (true)
  {
    auto [node, tnear, tfar] = stack.back();

    if (node.is_leaf)
    {
      break;
    }
    
    stack.pop_back();

    auto [axis, pos] = tree->Split(node);
    auto [left, right] = tree->Children(node);

    const double tentative_t = (pos - o[axis]) / d[axis];
    const double t = d[axis]==0. ? std::numeric_limits<double>::infinity() : tentative_t;

    if (unlikely(d[axis] == 0.))
    {
      if (o[axis] <= pos)
      {
        stack.push_back({left, tnear, tfar});
      }
      else
      {
        stack.push_back({right, tnear, tfar});
      }
    }
    else
    {
      if (d[axis] < 0.)
        std::swap(left, right);

      if (tfar > t)
      {
        stack.push_back({right, std::max(t, tnear), tfar});
      }

      if (tnear < t)
      {
        stack.push_back({left, tnear, std::min(t, tfar)});
      }
    }
  }
}



#ifdef HAVE_JSON

namespace dump_rj_tree
{
using namespace rapidjson_util;
using Alloc = rj::Document::AllocatorType;

rj::Value Build(const Tree &tree, Handle node, Alloc &alloc)
{
  if (node.is_leaf)
  {
    rj::Value v(rj::kObjectType);
    v.AddMember("kind", "leaf", alloc);
    v.AddMember("id", node.idx, alloc);
    return v;
  }
  else
  {
    auto [left, right] = tree.Children(node);
    auto [axis, pos] = tree.Split(node);
    rj::Value v_left = Build(tree, left, alloc);
    rj::Value v_right = Build(tree, right, alloc);
    rj::Value v{ rj::kObjectType };
    v.AddMember("id", node.idx, alloc);
    v.AddMember("kind", "branch", alloc);
    v.AddMember("split_axis", axis, alloc);
    v.AddMember("split_pos", pos, alloc);
    v.AddMember("left", v_left.Move(), alloc);
    v.AddMember("right", v_right.Move(), alloc);
    return v;
  }
}


}


void Tree::DumpTo(rapidjson::Document &doc, rapidjson::Value & parent) const
{
  using namespace rapidjson_util;
  auto &alloc = doc.GetAllocator();
  parent.AddMember("tree", dump_rj_tree::Build(*this, this->root, alloc), alloc);
}
#endif

} // namespace kdtree


} // namespace guiding