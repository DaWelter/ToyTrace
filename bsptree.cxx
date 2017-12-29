
#include "bsptree.hxx"

#define MAX_LEVEL 20
#define MIN_PRIMITIVES 4


void TreeNode::Split(int level, std::vector<Primitive *> list,const Box &nodebox)
{
  if(level>=MAX_LEVEL || list.size()<=MIN_PRIMITIVES) 
  {
    //std::cout<<"leaf " << this << " l:"<<level<<" s:"<<list.size()<<std::endl;
    primitive.swap(list);
    return;
  }

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

template<class IntersectionChecker>
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
  
  int first,last;
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
    char hitside = ((ray.org+intersectionChecker.ray_length*ray.dir)[splitaxis]<splitpos) ? 0 : 1;
    if ((!bhit || hitside!=first) && child[last]) 
    {
        bhit |= child[last]->Intersect(ray, dist, max, intersectionChecker);
    }
    return bhit;
  }
}


template
bool TreeNode::Intersect<IntersectionRecorder>(const Ray &ray, double min, double max, IntersectionRecorder &intersectionChecker) const;

template
bool TreeNode::Intersect<FirstIntersection>(const Ray &ray, double min, double max, FirstIntersection &intersectionChecker) const;


std::unique_ptr<TreeNode> BuildBspTree(const std::vector<Primitive *> &list, const Box &box)
{
  std::cout << "building bsp tree ..." << std::endl;
  auto root = std::make_unique<TreeNode>();
  root->Split(0, list, box);
  std::cout << std::endl;
  std::cout << "bsp tree finished" << std::endl;
  return std::move(root);
}


HitId IntersectionCalculator::First(const Ray &ray, double &ray_length)
{
  HitId hit;
  FirstIntersection intersectionChecker {ray_length, hit};
  root.Intersect(ray, 0, ray_length, intersectionChecker);
  return hit;
}


void IntersectionCalculator::All(const Ray &ray, double ray_length)
{
  temporary_hit_storage.clear();
  IntersectionRecorder intersectionChecker {ray_length, temporary_hit_storage};
  root.Intersect(ray, 0, ray_length, intersectionChecker);
  using RecordType = std::remove_reference<decltype(temporary_hit_storage[0])>::type;
  std::sort(temporary_hit_storage.begin(), temporary_hit_storage.end(),
    [](const RecordType &a, const RecordType &b) -> bool { return a.t < b.t; }
  );
  // Reject hits occuring within Eps of the previous hit.
  auto it = std::unique(temporary_hit_storage.begin(), temporary_hit_storage.end(),
    [](const RecordType &a, const RecordType &b) -> bool { assert(a.t <= b.t); return b.t-a.t < RAY_EPSILON; }
  );
  temporary_hit_storage.resize(it - temporary_hit_storage.begin());
}