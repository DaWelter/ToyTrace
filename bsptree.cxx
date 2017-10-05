
#include "bsptree.hxx"

#define MAX_LEVEL 10
#define MIN_PRIMITIVES 4


void TreeNode::Split(int level, std::vector<Primitive *> list,const Box &nodebox)
{
  if(level>=MAX_LEVEL || list.size()<=MIN_PRIMITIVES) 
  {
    int n = list.size();
    std::cout<<"leaf " << this << " l:"<<level<<" s:"<<n<<std::endl;
    std::cout<<"  "<<nodebox.min<<std::endl;
    std::cout<<"  "<<nodebox.max<<std::endl;
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
  {
    decltype(list) begone{};
    list.swap(begone);
  }

  if(childlist[0].size()>0) {
    child[0] = new TreeNode();
    child[0]->Split(level+1,childlist[0],childbox[0]);
  }
  if(childlist[1].size()>0) {
    child[1] = new TreeNode();
    child[1]->Split(level+1,childlist[1],childbox[1]);
  }
}


bool TreeNode::Intersect(const Ray &ray, double min, double max, IntersectionChecker &intersectionChecker) const
{
  if(!child[0] && !child[1]) 
  {
    //std::cout << "L:" << this << primitive.size() << std::endl;
    bool res=false;
    for(unsigned int i=0;i<primitive.size();i++) 
    {
      res |= intersectionChecker.intersect(ray, *primitive[i]);
    }
    return res;
  }
  
  double dist = (splitpos-ray.org[splitaxis])/ray.dir[splitaxis];
  
  //std::cout << "N:" << this << " a=" << splitaxis << " d=" << splitpos << std::endl;
  
  char first,last;
  if((ray.org[splitaxis]<splitpos) || 
     (ray.org[splitaxis]==splitpos && ray.dir[splitaxis] > 0))
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