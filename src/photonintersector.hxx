#pragma once

#include<vector>
#include<iterator>

#include <embree3/rtcore.h>

#include "primitive.hxx"
#include "util.hxx"
#include "vec3f.hxx"
#include "ray.hxx"

class PhotonIntersector
{
private:
  RTCDevice rtdevice = nullptr;
  RTCScene rtscene = nullptr;
  int geom_id = -1;
  static constexpr int HIT_LIST_LENGTH = 16;
  struct Ray2 : public RTCRayHit
  {
    // we remember up to HIT_LIST_LENGTH hits to ignore duplicate hits
    unsigned int firstHit, lastHit;
    unsigned int hit_geomIDs[PhotonIntersector::HIT_LIST_LENGTH];
    unsigned int hit_primIDs[PhotonIntersector::HIT_LIST_LENGTH];
    int *items;
    float *distances;
    int num_items;
    int buffer_size;
  };
  struct IntersectContext : public RTCIntersectContext
  {
    Ray2 *userRayExt;
  };
  
  static void occlusionFilter(const RTCFilterFunctionNArguments* args);
  friend struct PhotonIntersector::Ray2;  
public:
  PhotonIntersector(double search_radius, const ToyVector<Double3> &items);
  ~PhotonIntersector();
  PhotonIntersector(const PhotonIntersector &) = delete;
  PhotonIntersector& operator=(const PhotonIntersector&) = delete;
  
  // TODO: Change this so I don't use raw pointers.
  int Query(const Ray &ray, double length, int *items, float *distances, const int buffer_size);
};

