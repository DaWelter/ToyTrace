#ifndef GROUP_HXX
#define GROUP_HXX

#include <vector> // for dynamic array stl template 'vector'
#include <iostream>

#include "primitive.hxx"
#include "box.hxx"

class Group : public Primitive
{
  bool BoxIntersect(Ray &ray) 
  {
    double t1, t2;
    double a  = Epsilon;
    double b  = ray.t;
    
    t1 = (boundingBox.min[0]- ray.org[0]) / ray.dir[0];
    t2 = (boundingBox.max[0]- ray.org[0]) / ray.dir[0];
    
    if (ray.dir[0]< 0.0) {
      if (t1 < b)  b  = t1;
      if (t2 > a) a= t2;
    }
    else {
      if (t2 < b)  b  = t2;
      if (t1 > a) a= t1;
    }
    
    if  (b < a)
      return false;

    t1 = (boundingBox.min[1] - ray.org[1]) / ray.dir[1];
    t2 = (boundingBox.max[1] - ray.org[1]) / ray.dir[1];
    
    if (ray.dir[1] < 0.0) {
      if (t1 < b)  b  = t1;
      if (t2 > a) a= t2;
    }
    else {
      if (t2 < b)  b  = t2;
      if (t1 > a) a= t1;
    }
    
    if  (b < a)
      return false;
    
    t1 = (boundingBox.min[2] - ray.org[2]) / ray.dir[2];
    t2 = (boundingBox.max[2] - ray.org[2]) / ray.dir[2];
    
    if (ray.dir[2] < 0.0) {
      if (t1 < b)  b  = t1;
      if (t2 > a) a= t2;
    }
    else {
      if (t2 < b)  b  = t2;
      if (t1 > a) a= t1;
    }
    
    if  (b < a)
      return false;

	return true;
  }
public:
  std::vector<Primitive *> primitive; // primitives

  // the actual bounding box of all objects in the group
  Box boundingBox; 
  
  // just because of compiler warning
           Group() {};
  virtual ~Group() {};

  // add a new primitive (or a sub group)
  void Add(Primitive *prim)
  { primitive.push_back(prim); };
  
  // intersect all contained objects
  virtual bool Intersect(Ray &ray)
  {
    /* test if ray intersects the bounding box */
	if(!BoxIntersect(ray)) return false;
    
    /* intersection with the bounding box, might have */
    /* an intersection with one of the objects !      */
    bool hit = false;
    for (unsigned int i=0;i<primitive.size();i++)
      hit |= primitive[i]->Intersect(ray);
    
    return hit;
  };
  
  virtual Double3 GetNormal(Ray &ray)
  {
    std::cerr << "Group::GetNormal is not defined!" << std::endl;
    return Double3(0,0,0); // dummy !
  };

  virtual Box CalcBounds()
  {
    boundingBox.Clear(); // initialize bounding box
    
    // extend box with every sub bounding box
    for (int i=0;i<(int)primitive.size();i++) {
      Box box = primitive[i]->CalcBounds();
      boundingBox.Extend(box);
    }

    // primitives might lie on box borders
    boundingBox.min = boundingBox.min - Double3(Epsilon,Epsilon,Epsilon);
    boundingBox.max = boundingBox.max + Double3(Epsilon,Epsilon,Epsilon);

    return boundingBox;
  };

  virtual bool Occluded(Ray &ray)
  {
     if(!BoxIntersect(ray)) return false;
	 
	 for(unsigned int i=0;i<primitive.size();i++)
		 if(primitive[i]->Occluded(ray)) return true;
	 
     return false;
  }

  virtual bool InBox(const Box &box)
  {
	return boundingBox.Intersect(box);
  }

  virtual void ListPrimitives(std::vector<Primitive *> &list)
  {
	  for(int i=0;i<(int)primitive.size();i++) {
		  primitive[i]->ListPrimitives(list);
	  }
  }
};

#endif
