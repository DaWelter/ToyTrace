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
  static constexpr int FULL_HIT_LIST_LENGTH = 1024;
  struct Ray2 : public RTCRayHit
  {
    // we remember up to HIT_LIST_LENGTH hits to ignore duplicate hits
    unsigned int firstHit, lastHit;
    unsigned int hit_geomIDs[PhotonIntersector::HIT_LIST_LENGTH];
    unsigned int hit_primIDs[PhotonIntersector::HIT_LIST_LENGTH];
    int items[FULL_HIT_LIST_LENGTH];
    int num_items;
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
  
  template<class F>
  void Query(const Ray &ray, double length, F &&f);
};


template<class F>
inline void PhotonIntersector::Query(const Ray &ray, double length, F &&f)
{
  IntersectContext context;
  rtcInitIntersectContext(&context);
  Ray2 rtrayhit;
  context.userRayExt = &rtrayhit;
  RTCRay &rtray = rtrayhit.ray;
  rtray.org_x = ray.org[0];
  rtray.org_y = ray.org[1];
  rtray.org_z = ray.org[2];
  rtray.tnear = 0.f;
  rtray.dir_x = ray.dir[0];
  rtray.dir_y = ray.dir[1];
  rtray.dir_z = ray.dir[2];
  rtray.time = 0.f;
  rtray.tfar = static_cast<float>(length);
  rtray.mask = -1;
  rtray.id = 0;
  rtray.flags = 0;
  RTCHit &rthit = rtrayhit.hit;
  rthit.geomID = RTC_INVALID_GEOMETRY_ID;
  rthit.primID = RTC_INVALID_GEOMETRY_ID;
  rthit.instID[0] = RTC_INVALID_GEOMETRY_ID; 
  rtrayhit.firstHit = 0;
  rtrayhit.lastHit = 0;
  rtrayhit.num_items = 0;
  
  /* intersect ray with scene */
  rtcOccluded1(rtscene, &context, &rtrayhit.ray);
  
  for (int i=0; i<rtrayhit.num_items; ++i)
  {
    f(rtrayhit.items[i]);
  }
}
