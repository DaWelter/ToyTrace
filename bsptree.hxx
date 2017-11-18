#pragma once

#include<vector>
#include<iterator>

#include "primitive.hxx"


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



class BSPTree
{
  TreeNode root;
  Box boundingBox;
public:
  BSPTree() {}

  void Build(const std::vector<Primitive *> &list, const Box &box)
  {
    std::cout << "building bsp tree ..." << std::endl;
    boundingBox = box;
    root.Split(0, list, box);
    std::cout << std::endl;
    std::cout << "bsp tree finished" << std::endl;
  }

  bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const
  {
    FirstIntersection intersectionChecker {ray_length, hit};
    return root.Intersect(ray, 0, ray_length, intersectionChecker);
  }

  void IntersectAll(const Ray &ray, double ray_length, HitVector &all_hits) const
  {
    IntersectionRecorder intersectionChecker {ray_length, all_hits};
    root.Intersect(ray, 0, ray_length, intersectionChecker);
    using RecordType = std::remove_reference<decltype(all_hits[0])>::type;
    std::sort(all_hits.begin(), all_hits.end(),
      [](const RecordType &a, const RecordType &b) -> bool { return a.t < b.t; }
    );
    // Reject hits occuring within Eps of the previous hit.
    auto it = std::unique(all_hits.begin(), all_hits.end(),
      [](const RecordType &a, const RecordType &b) -> bool { assert(a.t <= b.t); return b.t-a.t < RAY_EPSILON; }
    );
    all_hits.resize(it - all_hits.begin());
  }
};

