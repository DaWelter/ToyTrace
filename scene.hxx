#ifndef SCENE_HXX
#define SCENE_HXX

#include "group.hxx"
#include "perspectivecamera.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "bsptree.hxx"

class Scene
{
  // parse an NFF file 'fileName', add all its primitives
  // to the specified group...
  void ParseNFF(FILE *file, char *fileName, Group *groupToAddTo);
  
public:
  BSPTree bsptree;
  Group primitives;
  std::vector<Light *> lights;
  Double3 bgColor;
  Camera *camera; 

  Scene()
    : bgColor(Double3(0,0,0)),
      camera(new PerspectiveCamera(Double3(0,0,0),Double3(0,0,-1),
				   Double3(0,1,0),60,640,480))
  {
	  //Double3 dir(-4,-10,-7);
	  //Normalize(dir);
	  //AddLight(new DirectionalLight(Double3(1,1,1),dir));
  };
	
  void AddLight(Light *light) {
	  lights.push_back(light);
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
  { return bsptree.Intersect(ray); }

  void BuildAccelStructure()
  {   
	  primitives.CalcBounds(); 
	  std::vector<Primitive *> list;
	  primitives.ListPrimitives(list);
	  bsptree.Build(list,primitives.boundingBox);
  }
};

#endif
