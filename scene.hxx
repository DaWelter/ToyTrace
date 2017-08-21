#ifndef SCENE_HXX
#define SCENE_HXX

#include "perspectivecamera.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "bsptree.hxx"

#include <memory>

class Scene
{
  // parse an NFF file 'fileName', add all its primitives
  // to the specified group...
  void ParseNFF(FILE *file, char *fileName);
  void ParseMesh(char *filename);
  
public:
  BSPTree bsptree;
  // TODO: Manage memory ...
  std::vector<Primitive*> primitives;
  std::vector<Light *> lights;
  Camera *camera;
  Double3 bgColor;
  Box boundingBox;

  Scene()
    : bgColor(Double3(0,0,0)),
      camera(new PerspectiveCamera(Double3(0,0,0),Double3(0,0,-1),
      Double3(0,1,0),60,640,480))
  {
  };
	
  void AddLight(std::unique_ptr<Light> light) 
  {
    // TODO: Manage memory ...
	  lights.push_back(light.release());
  }
  
  void AddPrimitive(std::unique_ptr<Primitive> primitive)
  {
    // TODO: Manage memory ...
    primitives.push_back(primitive.release());
  }

  // parse an NFF file 'fileName', store all its primitives
  void ParseNFF(char *fileName);
  
  // trace the given ray and shade it and
  // return the color of the shaded ray
  Double3 RayTrace(Ray &ray)
  {
    if (bsptree.Intersect(ray))
      if(ray.hit && ray.hit->shader)
        return ray.hit->shader->Shade(ray,this);
	  else 
      return Double3(1,0,0);
    else
      return bgColor; // ray missed geometric primitives
  };

  bool Occluded(Ray &ray)
  { 
    return bsptree.Intersect(ray); 
  }

  void BuildAccelStructure()
  {   
	  this->boundingBox = CalcBounds();
	  bsptree.Build(primitives, boundingBox);
  }
  
  void PrintInfo()
  {
    std::cout << std::endl;
    std::cout << "bounding box min: "
              << boundingBox.min << std::endl;
    std::cout << "bounding box max: "
              << boundingBox.max << std::endl;
  }
  
  Box CalcBounds()
  {
    Box scenebox;
    for (int i=0; i<(int)primitives.size(); i++) 
    {
      Box box = primitives[i]->CalcBounds();
      scenebox.Extend(box);
    }
    
    // primitives might lie on box borders
    scenebox.min -= Double3(Epsilon,Epsilon,Epsilon);
    scenebox.max += Double3(Epsilon,Epsilon,Epsilon);
    return scenebox;
  };
};


#endif
