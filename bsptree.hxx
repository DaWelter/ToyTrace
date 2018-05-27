#pragma once

#include<vector>
#include<iterator>

#include "primitive.hxx"
#include "util.hxx"


class IntersectionRecorder
{
public:
  double ray_length;
  HitId hit;
  HitVector &all_hits;
  
//   using FilterCallback = void (double, HitId &, void *);
//   FilterCallback *cb;
//   void* user;
  
  IntersectionRecorder(double _ray_length, HitVector &_all_hits /*, FilterCallback *_cb, void *_user*/)
    : ray_length {_ray_length}, all_hits(_all_hits)
  {
    _all_hits.reserve(16);
  }

  bool intersect(const Ray &ray, const Primitive &p)
  {
    int original_size = all_hits.size();
    p.Intersect(ray, ray_length, all_hits);
    if (all_hits.size() > original_size)
    {
      auto p_ptr = &p;
      auto the_end = all_hits.begin() + original_size;
      auto it = std::find_if(all_hits.begin(), the_end,
        [p_ptr](const HitRecord &r) -> bool { return r.primitive == p_ptr; }
      );
      if (it != the_end) // If the primitive is already in the hit array, we know that all potential hits are registered.
        all_hits.resize(original_size);
    }
    return false;
  }
};


class FirstIntersection
{
public:
  double &ray_length;
  HitId &hit;
  
  FirstIntersection(double &_ray_length, HitId &_hit)
    : ray_length {_ray_length}, hit(_hit)
  {
  }

  bool intersect(const Ray &ray, const Primitive &p)
  {
    return p.Intersect(ray, ray_length, hit);
  }
};


class TreeNode
{
  TreeNode* child[2];
  std::vector<Primitive *> primitive;
  char splitaxis;
  double splitpos;
public:
  TreeNode(const TreeNode &) = delete;
  TreeNode& operator=(const TreeNode &other) = delete;
  
  ~TreeNode()
  {
    if (child[0]) delete child[0];
    if (child[1]) delete child[1];
  }

  TreeNode() : splitaxis(0)
  {
    child[0] = child[1] = 0;
  }

  inline void SplitBox(const Box &nodebox, Box &box0, Box &box1)
  {
    box0 = box1 = nodebox;
    box0.max[splitaxis] = splitpos;
    box1.min[splitaxis] = splitpos;
  }

  void Split(int level, std::vector<Primitive *> list, const Box &nodebox);

  template<class IntersectionChecker>
  bool Intersect(const Ray &ray, double min, double max, IntersectionChecker &intersectionChecker) const;
};


std::unique_ptr<TreeNode> BuildBspTree(const std::vector<Primitive *> &list, const Box &box);


/* Stateful scene intersection calculations.
 * 
 * This class is intended to be used in the rendering threads. One instance per thread.
 * Internal storage is used to hold things like all intersections along a ray.
 * The class itself is ofc. not thread safe. Hence one instance per thread.
 */
class IntersectionCalculator
{
  const TreeNode &root;
  HitVector temporary_hit_storage;
public:
  IntersectionCalculator(const TreeNode &_root)
    : root(_root)
  {
  }
  
  HitId First(const Ray &ray, double &ray_length);

//   void All(const Ray &ray, double ray_length);
  
//   auto Hits() const 
//   {
//     return iter_pair<decltype(temporary_hit_storage.cbegin())>(
//       temporary_hit_storage.cbegin(),
//       temporary_hit_storage.cend());
//   }
};
