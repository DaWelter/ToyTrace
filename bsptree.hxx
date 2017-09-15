#ifndef _BSPTREE_H
#define _BSPTREE_H

#include "primitive.hxx"


#define MAX_LEVEL 10
#define MIN_PRIMITIVES 4


static int nprim=0;


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
	
	bool Intersect(const Ray &ray, double &ray_length, double min, double max, HitId &hit, const HitId *last_hit) const
  {
    if(!child[0] && !child[1]) 
    {
      bool res=false;
      for(unsigned int i=0;i<primitive.size();i++) 
      {
        if (last_hit && last_hit->primitive == primitive[i]) continue;
        res |= primitive[i]->Intersect(ray, ray_length, hit);
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
      if(child[first]) return child[first]->Intersect(ray, ray_length, min,max, hit, last_hit);
      else return false;
    } 
    else if(dist<min) 
    {
      if(child[last]) return child[last]->Intersect(ray, ray_length, min,max, hit, last_hit);
      else return false;
    } 
    else 
    {
      bool bhit;
      if(child[first]) bhit = child[first]->Intersect(ray, ray_length, min,dist, hit, last_hit);
      else bhit = false;
      
      char hitside = ((ray.org+ray_length*ray.dir)[splitaxis]<splitpos)?0:1;
      if((!bhit || hitside!=first) && child[last]) {
         bhit |= child[last]->Intersect(ray, ray_length, dist,max, hit, last_hit);
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

	bool Intersect(const Ray &ray, double &ray_length, HitId &hit, const HitId *last_hit) const
  {
		return root.Intersect(ray, ray_length, 0, ray_length, hit, last_hit);
	}
};


#endif