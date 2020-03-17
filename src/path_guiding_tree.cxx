#include "path_guiding_tree.hxx"

#ifdef HAVE_JSON
#include "json.hxx"
#include "rapidjson/document.h"
#endif


namespace guiding
{


namespace kdtree
{

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