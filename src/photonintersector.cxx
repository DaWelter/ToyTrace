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
  assert(args->N == 1);
  int* valid = args->valid;
  const IntersectContext* context = (const IntersectContext*) args->context;
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

  // Buffer is full. Don't report any more.
  if (ray2->num_items >= ray2->buffer_size) {
    valid[0]=1; return;
  }
  
  ray2->distances[ray2->num_items] = ray2->ray.tfar;
  ray2->items[ray2->num_items++] = hit->primID;
  
//   /* calculate and accumulate transparency */
//   float T = transparencyFunction(h);
//   T *= ray2->transparency;
//   ray2->transparency = T;
//   if (T != 0.0f) 
  valid[0] = 0;
}


int PhotonIntersector::Query(const Ray &ray, double length, int *items, float *distances, const int buffer_size)
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
  rtrayhit.items = items;
  rtrayhit.distances = distances;
  rtrayhit.num_items = 0;
  rtrayhit.buffer_size = buffer_size;
  
  /* intersect ray with scene */
  rtcOccluded1(rtscene, &context, &rtrayhit.ray);
  return rtrayhit.num_items;
}
