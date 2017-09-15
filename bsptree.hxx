#ifndef _BSPTREE_H
#define _BSPTREE_H

#include "primitive.hxx"


#define MAX_LEVEL 10
#define MIN_PRIMITIVES 4


static int nprim=0;


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
    if ((to_ignore[0].primitive == &p) || 
       (to_ignore[1].primitive == &p)) 
     return false;;
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

	inline void SplitBox(const Box &nodebox,Box &box0,Box &box1) {
		box0=box1=nodebox;
		box0.max[splitaxis] = splitpos;//0.5*(nodebox.max[splitaxis]+nodebox.min[splitaxis]);
		box1.min[splitaxis] = splitpos;//0.5*(nodebox.max[splitaxis]+nodebox.min[splitaxis]);
	}

	void Split(int level,std::vector<Primitive *> &list,const Box &nodebox)
	{
		if(level>=MAX_LEVEL || list.size()<=MIN_PRIMITIVES) {
			int n = list.size();
			//Assert(n<50);
			//std::cout<<"-leaf: l:"<<level<<" s:"<<n<<"-";
			std::cout<<".";
			nprim += n;

			primitive.swap(list);
			return;
		}
		
		std::cout<<".";

		double axislen = nodebox.max[0]-nodebox.min[0];
		splitaxis = 0;
		if(nodebox.max[1]-nodebox.min[1] > axislen) {
			splitaxis = 1;
			axislen = nodebox.max[1]-nodebox.min[1];
		}
		if(nodebox.max[2]-nodebox.min[2] > axislen) {
			splitaxis = 2;
			axislen = nodebox.max[2]-nodebox.min[2];
		}
		splitpos = 0.5*(nodebox.max[splitaxis]+nodebox.min[splitaxis]);

		Box childbox[2];
		SplitBox(nodebox,childbox[0],childbox[1]);

		std::vector<Primitive *> childlist[2];
		for(unsigned int i=0;i<list.size();i++)
		{
			bool inlist=false;
			if(list[i]->InBox(childbox[0])) { inlist=true; childlist[0].push_back(list[i]); }
			if(list[i]->InBox(childbox[1])) { inlist=true; childlist[1].push_back(list[i]); }
			assert(inlist);
		}
		list.clear();

		if(childlist[0].size()>0) {
			child[0] = new TreeNode();
			child[0]->Split(level+1,childlist[0],childbox[0]);
		}
		if(childlist[1].size()>0) {
			child[1] = new TreeNode();
			child[1]->Split(level+1,childlist[1],childbox[1]);
		}
	}


	bool Intersect(const Ray &ray, double min, double max, IntersectionChecker &intersectionChecker) const
  {
    if(!child[0] && !child[1]) 
    {
      bool res=false;
      for(unsigned int i=0;i<primitive.size();i++) 
      {
        res |= intersectionChecker.intersect(ray, *primitive[i]);
      }
      return res;
    }
    
    double dist = (splitpos-ray.org[splitaxis])/ray.dir[splitaxis];
    
    char first,last;
    if(ray.org[splitaxis]<=splitpos) 
    {
      first=0; last=1;
    } 
    else 
    {
      first=1; last=0;
    }
    
    if(dist<0 || dist>max) 
    {
      // Facing away from split plane or distance is further than max checking distance.
      if(child[first]) return child[first]->Intersect(ray, min, max, intersectionChecker);
      else return false;
    } 
    else if(dist<min) 
    {
      // Facing going towards the split plane but min checking distance is beyond the plane, i.e. within second node.
      if(child[last]) return child[last]->Intersect(ray, min, max, intersectionChecker);
      else return false;
    } 
    else 
    {
      // Intersecting both subvolumes.
      bool bhit;
      if(child[first]) 
        bhit = child[first]->Intersect(ray, min, dist, intersectionChecker);
      else 
        bhit = false;
      // Because primitives overlapping both nodes, we have to check if the hit location is actually in the first node.
      char hitside = ((ray.org+intersectionChecker.rayLength()*ray.dir)[splitaxis]<splitpos) ? 0 : 1;
      if ((!bhit || hitside!=first) && child[last]) 
      {
         bhit |= child[last]->Intersect(ray, dist, max, intersectionChecker);
      } 
      return bhit;
    }
  }
};


class BSPTree
{
	TreeNode root;
	Box boundingBox;
public:
	BSPTree() {}

	void Build(std::vector<Primitive *> &list,const Box &box) {
		std::cout << "building bsp tree: "<<list.size()<<" primitives"<<std::endl;
		boundingBox=box;
		root.Split(0,list,box);
		std::cout << std::endl;

		std::cout << "number of references: " << nprim << std::endl;
		std::cout << "bsp tree finished" <<std::endl;
	}

	bool Intersect(const Ray &ray, double &ray_length, HitId &hit, const HitId &to_ignore1, const HitId &to_ignore2) const
  {
    IntersectionChecker intersectionChecker{ray_length, hit, to_ignore1, to_ignore2};
		return root.Intersect(ray, 0, ray_length, intersectionChecker);
	}
};


#endif