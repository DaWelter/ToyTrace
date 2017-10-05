#ifndef _BSPTREE_H
#define _BSPTREE_H

#include "primitive.hxx"


class IntersectionChecker
{
  double &ray_length;
  HitId &hit;
  HitId to_ignore[2];
public:
  IntersectionChecker(double &_ray_length, HitId &_hit, HitId igno1, HitId igno2)
    : ray_length{_ray_length}, hit(_hit), to_ignore{ igno1, igno2 }
  {
  }
//   void addToIgnore(const HitId &igno)
//   {
//     if (to_ignore[0])
//     {
//       assert((bool)to_ignore[1]==false);
//       to_ignore[1] = igno;
//     }
//     else
//       to_ignore[0] = igno;
//   }
  double rayLength() const 
  { 
    return ray_length; 
  }
  HitId hitId() const 
  { 
    return hit; 
  }
  bool intersect(const Ray &ray, const Primitive &p)
  {
    if (to_ignore[0].primitive == &p || to_ignore[1].primitive == &p)
      return p.Intersect(ray, ray_length, hit, to_ignore[0], to_ignore[1]);
    else
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
		if(child[0]) delete child[0];
		if(child[1]) delete child[1];
	}

	TreeNode() : splitaxis(0)
	{ child[0]=child[1]=0; }

	inline void SplitBox(const Box &nodebox,Box &box0,Box &box1) 
  {
		box0=box1=nodebox;
		box0.max[splitaxis] = splitpos;
		box1.min[splitaxis] = splitpos;
	}

	void Split(int level, std::vector<Primitive *> list,const Box &nodebox);
	bool Intersect(const Ray &ray, double min, double max, IntersectionChecker &intersectionChecker) const;
};



class BSPTree
{
	TreeNode root;
	Box boundingBox;
public:
	BSPTree() {}

	void Build(const std::vector<Primitive *> &list,const Box &box) 
  {
		std::cout << "building bsp tree ..."<<std::endl;
		boundingBox=box;
		root.Split(0,list,box);
		std::cout << std::endl;
		std::cout << "bsp tree finished" <<std::endl;
	}

	bool Intersect(const Ray &ray, double &ray_length, HitId &hit, const HitId &to_ignore1, const HitId &to_ignore2) const
  {
    IntersectionChecker intersectionChecker{ray_length, hit, to_ignore1, to_ignore2};
    return root.Intersect(ray, 0, ray_length, intersectionChecker);
	}
};


#endif