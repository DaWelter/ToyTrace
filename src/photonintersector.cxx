#include "photonintersector.hxx"


PhotonIntersector::PhotonIntersector(double search_radius, const ToyVector<Double3> &items)
{
  rtdevice = rtcNewDevice(nullptr);
  rtscene = rtcNewScene(rtdevice);
  RTCGeometry geom = rtcNewGeometry(rtdevice, RTC_GEOMETRY_TYPE_DISC_POINT);
  float *point_vertices = (float*)rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, 4*sizeof(float), items.size());
  for (int i=0; i<items.size(); ++i)
  {
    for (int j=0; j<3; ++j)
    {
      point_vertices[i*4+j] = (float)items[i][j];
      point_vertices[i*4+3] = (float)search_radius;
    }
  }
  rtcSetGeometryOccludedFilterFunction(geom, occlusionFilter);
  rtcCommitGeometry(geom);
  geom_id = rtcAttachGeometry(rtscene, geom);
  rtcReleaseGeometry(geom);
  rtcCommitScene (rtscene);
}


PhotonIntersector::~PhotonIntersector()
{
  rtcReleaseScene(rtscene);
  rtcReleaseDevice(rtdevice);
}


void PhotonIntersector::occlusionFilter(const RTCFilterFunctionNArguments* args)
{
  /* avoid crashing when debug visualizations are used */
  if (args->context == nullptr) return;

  assert(args->N == 1);
  int* valid = args->valid;
  const IntersectContext* context = (const IntersectContext*) args->context;
  Ray* ray = (Ray*)args->ray;
  RTCHit* hit = (RTCHit*)args->hit;

  /* ignore inactive rays */
  if (valid[0] != -1) return;

  Ray2* ray2 = (Ray2*) context->userRayExt;
  assert(ray2);

  for (unsigned int i=ray2->firstHit; i<ray2->lastHit; i++) {
    unsigned slot= i%HIT_LIST_LENGTH;
    if (ray2->hit_geomIDs[slot] == hit->geomID && ray2->hit_primIDs[slot] == hit->primID) {
      valid[0] = 0; return; // ignore duplicate intersections
    }
  }
  /* store hit in hit list */
  unsigned int slot = ray2->lastHit%HIT_LIST_LENGTH;
  ray2->hit_geomIDs[slot] = hit->geomID;
  ray2->hit_primIDs[slot] = hit->primID;
  ray2->lastHit++;
  if (ray2->lastHit - ray2->firstHit >= HIT_LIST_LENGTH)
    ray2->firstHit++;

  //Vec3fa h = ray->org + ray->dir * ray->tfar;

  ray2->items[ray2->num_items++] = hit->primID;
  
//   /* calculate and accumulate transparency */
//   float T = transparencyFunction(h);
//   T *= ray2->transparency;
//   ray2->transparency = T;
//   if (T != 0.0f) 
  valid[0] = 0;
}
