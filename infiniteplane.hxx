#ifndef INFINITEPLANE_HXX
#define INFINITEPLANE_HXX

#include "vec3f.hxx"

class InfinitePlane
{
  Double3 normal;
  Double3 origin;
public:
  InfinitePlane(Double3 origin, Double3 normal)
    : normal(normal),origin(origin)
  {
    Normalize(normal);
  };

  bool Intersect(const Ray &ray, double tnear, double &ray_length, Double3 &barry) const
  {
    double s = Dot(origin,normal);
    double nv = Dot(ray.dir,normal);
    if(nv*nv < 1.0e-9)
      return false;
    double t = (s-Dot(ray.org,normal))/(nv);
    if(t<=tnear || t>ray_length)
      return false;
    barry = ray.PointAt(t);
    ray_length = t;
    return true;
  };
};

#endif
