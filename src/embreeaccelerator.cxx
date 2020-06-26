#include "embreeaccelerator.hxx"
#include "scene.hxx"
#include "ray.hxx"
#include "vec3f.hxx"

#include <embree3/rtcore.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)  // Float to double conversion. I don't care. There are too many of them, and it is too much of a mess to try to make it right.
#endif

using namespace EmbreeAcceleratorDetail;

thread_local ToyVector<BoundaryIntersection> EmbreeAccelerator::intersections_result;

EmbreeAccelerator::EmbreeAccelerator()
{
  rtdevice = rtcNewDevice(nullptr);
  rtscene = rtcNewScene(rtdevice);
}


EmbreeAccelerator::~EmbreeAccelerator()
{
  rtcReleaseScene(rtscene);
  rtcReleaseDevice(rtdevice);
}


void EmbreeAccelerator::Build(bool enable_intersection_in_order_call)
{
  if (enable_intersection_in_order_call)
    rtcSetSceneFlags(rtscene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
  rtcCommitScene (rtscene);
}


void EmbreeAccelerator::InsertRefTo(Mesh& mesh)
{
  RTCGeometry rtmesh = rtcNewGeometry(rtdevice, RTC_GEOMETRY_TYPE_TRIANGLE);
  auto id = rtcAttachGeometry(rtscene, rtmesh);
  rtcSetGeometryUserData(rtmesh, (void*)&mesh);
  mesh.identifier = id;
  rtcSetSharedGeometryBuffer(rtmesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 
                             mesh.vertices.data(), 0, 
                             sizeof(decltype(mesh.vertices)::Scalar)*3, mesh.vertices.rows());
  rtcSetSharedGeometryBuffer(rtmesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                             mesh.vert_indices.data(), 0,
                             sizeof(decltype(mesh.vert_indices)::Scalar)*3, mesh.vert_indices.rows());
  rtcSetGeometryVertexAttributeCount(rtmesh,2);
  rtcSetSharedGeometryBuffer(rtmesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3,
                             mesh.normals.data(), 0,
                             sizeof(decltype(mesh.normals)::Scalar)*3, mesh.NumVertices());
  rtcSetSharedGeometryBuffer(rtmesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, RTC_FORMAT_FLOAT2,
                             mesh.uvs.data(), 0,
                             sizeof(decltype(mesh.uvs)::Scalar)*2, mesh.NumVertices());
  rtcCommitGeometry(rtmesh);
  rtcReleaseGeometry(rtmesh);
}


void EmbreeAccelerator::InsertRefTo(Spheres& spheres)
{
  RTCGeometry rtgeom = rtcNewGeometry(rtdevice, RTC_GEOMETRY_TYPE_USER);
  unsigned int id = rtcAttachGeometry(rtscene, rtgeom); 
  spheres.identifier = id;
  rtcSetGeometryUserData(rtgeom, (void*)&spheres);
  rtcSetGeometryUserPrimitiveCount(rtgeom, spheres.NumSpheres());
  rtcSetGeometryBoundsFunction(rtgeom,SphereBoundsFunc,nullptr);
  rtcSetGeometryIntersectFunction(rtgeom,SphereIntersectFunc);
  rtcSetGeometryOccludedFunction (rtgeom, SphereOccludedFunc);
  rtcCommitGeometry(rtgeom);
  rtcReleaseGeometry(rtgeom);
}


void EmbreeAccelerator::InsertRefTo(Geometry & geo)
{
  if (geo.type == Geometry::PRIMITIVES_SPHERES)
    InsertRefTo(static_cast<Spheres&>(geo));
  else if (geo.type == Geometry::PRIMITIVES_TRIANGLES)
    InsertRefTo(static_cast<Mesh&>(geo));
}


bool EmbreeAccelerator::FirstIntersection(const Ray &ray, double tnear, double &ray_length, SurfaceInteraction &intersection) const
{
  RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  RTCRayHit rtrayhit;
  RTCRay &rtray = rtrayhit.ray;
  rtray.org_x = ray.org[0];
  rtray.org_y = ray.org[1];
  rtray.org_z = ray.org[2];
  rtray.tnear = (float)tnear;
  rtray.dir_x = ray.dir[0];
  rtray.dir_y = ray.dir[1];
  rtray.dir_z = ray.dir[2];
  rtray.time = 0.;
  rtray.tfar = static_cast<float>(ray_length);
  rtray.mask = -1;
  rtray.id = 0;
  rtray.flags = 0;
  RTCHit &rthit = rtrayhit.hit;
  rthit.geomID = RTC_INVALID_GEOMETRY_ID;
  rthit.primID = RTC_INVALID_GEOMETRY_ID;
  rthit.instID[0] = RTC_INVALID_GEOMETRY_ID; 
  // ----
  // Here we do the work!!
  rtcIntersect1(rtscene, &context, &rtrayhit);
  // ----
  if (rthit.geomID == RTC_INVALID_GEOMETRY_ID)
    return false;
  assert(rtray.tfar > rtray.tnear);
  ray_length = rtray.tfar;
  const Geometry* prim = (const Geometry*)rtcGetGeometryUserData(rtcGetGeometry(rtscene, rthit.geomID));
  intersection.hitid.geom = prim;
  intersection.hitid.index    = rthit.primID;
  if (prim->type == Geometry::PRIMITIVES_TRIANGLES)
    FirstIntersectionTriangle(rthit, ray, intersection);
  else
  {
    assert (prim->type == Geometry::PRIMITIVES_SPHERES);
    FirstIntersectionSphere(rthit, ray, intersection);
  }
  ASSERT_NORMALIZED(intersection.geometry_normal);
  ASSERT_NORMALIZED(intersection.smooth_normal);
  intersection.SetOrientedNormals(ray.dir);
  return true;
}


struct RTCIntersectContextWithMyCallback : public RTCIntersectContext
{
  mutable ToyVector<BoundaryIntersection> hitRecords;
  float tfarMax;

  // TODO: filter duplicate intersections!
  static void MyFilterFunction(const struct RTCFilterFunctionNArguments* args)
  {
    assert(args->N == 1);
    assert(args->geometryUserPtr);
    const auto* hit = (const RTCHit*)args->hit;
    const auto* ray = (const RTCRay*)args->ray;
    const auto* context = static_cast<const RTCIntersectContextWithMyCallback*>(args->context);
    const auto previousHits = AsSpan(context->hitRecords);

    /* ignore inactive rays */
    if (args->valid[0] != -1 || ray->tfar > context->tfarMax) return;

    args->valid[0] = 0;

    const auto* geo = static_cast<const Geometry*>(args->geometryUserPtr);

    const auto current = BoundaryIntersection{
      geo->index_in_scene,
      static_cast<scene_index_t>(hit->primID),
      ray->tfar,
      { hit->Ng_x, hit->Ng_y, hit->Ng_z }
    };

    auto [it, found] = FindPlaceIfUnique(context->hitRecords.begin(), context->hitRecords.end(), current, 
      /*less = */ [](const auto &a, const auto &b) { return a.t < b.t;  },
      /*eq   = */ [](const auto &a, const auto &b) {
      // Should I compare the primitive, too, or is this enough.
      // If I compare the primitive naively it won't work near edges with adjacent triangles.
      // There the intersector registers both triangles.
      return (a.geom == b.geom) & (a.t == b.t);
      //return (a.t == b.t) & (a.prim == b.prim) & (a.geom == b.geom); }
    }
    );

    if (!found)
    {
      context->hitRecords.insert(it, current);
    }
  }
};


Span<BoundaryIntersection> EmbreeAccelerator::IntersectionsInOrder(const Ray &ray, double tnear, double tfar) const
{
  RTCIntersectContextWithMyCallback context;
  rtcInitIntersectContext(&context);
  context.hitRecords = std::move(this->intersections_result);
  context.hitRecords.clear();
  context.filter = RTCIntersectContextWithMyCallback::MyFilterFunction;
  context.tfarMax = tfar;

  RTCRay rtray; // = rtrayhit.ray;
  rtray.org_x = ray.org[0];
  rtray.org_y = ray.org[1];
  rtray.org_z = ray.org[2];
  rtray.tnear = static_cast<float>(tnear);
  rtray.dir_x = ray.dir[0];
  rtray.dir_y = ray.dir[1];
  rtray.dir_z = ray.dir[2];
  rtray.time = 0.;
  rtray.tfar = static_cast<float>(tfar);
  rtray.mask = -1;
  rtray.id = 0;
  rtray.flags = 0;
  // ----
  // Here we do the work!!
  rtcOccluded1(rtscene, &context, &rtray);

  // TODO: Since the memory returned comes from internal storage, I can
  // never hold on to it and call the IntersectionsInOrder function again.
  // Improve this!
  this->intersections_result = std::move(context.hitRecords);
  return AsSpan(this->intersections_result);
}


void EmbreeAccelerator::FirstIntersectionTriangle(const RTCHit &rthit, const Ray &ray, SurfaceInteraction &intersection) const
{
  auto geom = rtcGetGeometry(rtscene, rthit.geomID);
  Float3 pos, n_sh;
  rtcInterpolate1(
    geom,
    rthit.primID, rthit.u, rthit.v, RTC_BUFFER_TYPE_VERTEX, 0, pos.data(), nullptr, nullptr, 3);
  rtcInterpolate1(
    geom,
    rthit.primID, rthit.u, rthit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, n_sh.data(), nullptr, nullptr, 3);
  rtcInterpolate1(
    geom,
    rthit.primID, rthit.u, rthit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, intersection.tex_coord.data(), nullptr, nullptr, 2);
  HitId &hit = intersection.hitid;
  hit.barry[0] = rthit.u;
  hit.barry[1] = rthit.v;
  hit.barry[2] = 1 - hit.barry[0] - hit.barry[1];   
  intersection.pos = pos.cast<double>();
  intersection.smooth_normal = n_sh.cast<double>();
  Normalize(intersection.smooth_normal);
  intersection.geometry_normal[0] = rthit.Ng_x;
  intersection.geometry_normal[1] = rthit.Ng_y;
  intersection.geometry_normal[2] = rthit.Ng_z;
  Normalize(intersection.geometry_normal);
  
  const auto &mesh = static_cast<const Mesh&>(*hit.geom);
  const auto tri = mesh.vert_indices.row(hit.index);
  FillPosBoundsTriangle(intersection, mesh.vertices.row(tri[0]), mesh.vertices.row(tri[1]), mesh.vertices.row(tri[2]));
}


void EmbreeAccelerator::FirstIntersectionSphere(const RTCHit &rthit, const Ray &ray, SurfaceInteraction &intersection) const
{
  const auto* spheres = static_cast<const Spheres*>(intersection.hitid.geom);
  HitId &hit = intersection.hitid;
  Float3 delta; delta << rthit.Ng_x, rthit.Ng_y, rthit.Ng_z;
  Normalize(delta);
  Float3 pos; float radius; std::tie(pos, radius) = spheres->Get(rthit.primID);
  intersection.pos = (pos + radius*delta).cast<double>();
  hit.barry = delta.cast<double>();
  intersection.geometry_normal = hit.barry;
  intersection.smooth_normal = intersection.geometry_normal;
  intersection.tex_coord = Projections::SphericalToUv(Projections::KartesianToSpherical(delta));
  FillPosBoundsSphere(intersection);
}


//Adapted from Embree tutorial user_geometry_device.cpp
void EmbreeAccelerator::SphereBoundsFunc(const struct RTCBoundsFunctionArguments* args)
{
  const Spheres* spheres = (const Spheres*) args->geometryUserPtr;
  RTCBounds* bounds_o = args->bounds_o;
  Float3 pos; float radius; std::tie(pos, radius) = spheres->Get(args->primID);
  bounds_o->lower_x = pos[0]-radius;
  bounds_o->lower_y = pos[1]-radius;
  bounds_o->lower_z = pos[2]-radius;
  bounds_o->upper_x = pos[0]+radius;
  bounds_o->upper_y = pos[1]+radius;
  bounds_o->upper_z = pos[2]+radius;
  bounds_o->lower_x = std::nextafter(bounds_o->lower_x, -std::numeric_limits<float>::infinity());
  bounds_o->lower_y = std::nextafter(bounds_o->lower_y, -std::numeric_limits<float>::infinity());
  bounds_o->lower_z = std::nextafter(bounds_o->lower_z, -std::numeric_limits<float>::infinity());
  bounds_o->upper_x = std::nextafter(bounds_o->upper_x,  std::numeric_limits<float>::infinity());
  bounds_o->upper_y = std::nextafter(bounds_o->upper_y,  std::numeric_limits<float>::infinity());
  bounds_o->upper_z = std::nextafter(bounds_o->upper_z,  std::numeric_limits<float>::infinity());
}


bool EmbreeAccelerator::IsOccluded(const Ray & ray, double tnear, double tfar) const
{
  RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  RTCRay rtray;
  rtray.org_x = ray.org[0];
  rtray.org_y = ray.org[1];
  rtray.org_z = ray.org[2];
  rtray.tnear = (float)tnear;
  rtray.dir_x = ray.dir[0];
  rtray.dir_y = ray.dir[1];
  rtray.dir_z = ray.dir[2];
  rtray.time = 0.;
  rtray.tfar = static_cast<float>(tfar);
  rtray.mask = -1;
  rtray.id = 0;
  rtray.flags = 0;

  rtcOccluded1(rtscene, &context, &rtray);
  return rtray.tfar <= 0.;
}


void EmbreeAccelerator::SphereIntersectFunc(const RTCIntersectFunctionNArguments* args)
{
  int* valid = args->valid;
  assert(args->N == 1);
  if (!valid[0]) return;
  
  auto* rtrayn = RTCRayHitN_RayN(args->rayhit, args->N);
  auto* rthitn = RTCRayHitN_HitN(args->rayhit, args->N);
  RTCRay rtray = rtcGetRayFromRayN(rtrayn, args->N, 0);

  unsigned int primID = args->primID;
  const Spheres* spheres = (const Spheres*) args->geometryUserPtr;
  auto &sphere = spheres->spheres[primID];
  
  Eigen::Map<const Eigen::Vector3f> ray_dir{&rtray.dir_x};
  Eigen::Map<const Eigen::Vector3f> ray_org{&rtray.org_x};
  Float3 sphere_p; float sphere_r; std::tie(sphere_p, sphere_r) = spheres->Get(args->primID);  
  Float3 v = ray_org - sphere_p;
  ASSERT_NORMALIZED(ray_dir);
  const float A = 1.f; //Dot(ray_dir,ray_dir);
  const float B = 2.0f*Dot(v,ray_dir);
  const float C = Dot(v,v) - Sqr(sphere_r);
  const float Berr = DotAbs(v, ray_dir)*util::Gamma<float>(3);
  const float Cerr = (2.f*Dot(v,v) + Sqr(sphere_r))*util::Gamma<float>(3);
  float t0, t1, err0, err1;
  const bool has_solution = util::Quadratic(A, B, C, 0.f, Berr, Cerr, t0, t1, err0, err1);
//   float t0, t1;
//   float err0 = 0.f, err1 = 0.f;
//   const bool has_solution = Quadratic(A, B, C, t0, t1);
  if (!has_solution)
    return;
  RTCHit potentialHit;
  potentialHit.u = 0.0f;
  potentialHit.v = 0.0f;
  potentialHit.instID[0] = args->context->instID[0];
  potentialHit.geomID = spheres->identifier;
  potentialHit.primID = primID;

  // First check the smaller one of the t-values. 
  // See if it is within the [tnear,tfar] interval.
  // If it is then this is our first hit.
  if ((rtray.tnear < t0-err0) & (t0+err0 < rtray.tfar))
  {
    int imask;
    bool mask = 1;
    {
      imask = mask ? -1 : 0;
    }
    
    const Float3 Ng = ray_org+t0*ray_dir-sphere_p;
    potentialHit.Ng_x = Ng[0];
    potentialHit.Ng_y = Ng[1];
    potentialHit.Ng_z = Ng[2];

    RTCFilterFunctionNArguments fargs;
    fargs.valid = (int*)&imask;
    fargs.geometryUserPtr = args->geometryUserPtr;
    fargs.context = args->context;
    fargs.ray = (RTCRayN *)args->rayhit;
    fargs.hit = (RTCHitN*)&potentialHit;
    fargs.N = 1;

    const float old_t = rtray.tfar;
    RTCRayN_tfar(rtrayn, args->N, 0) = t0;
    rtray.tfar = t0;
    rtcFilterIntersection(args,&fargs);

    // Did the filter function accept the hit?
    if (imask == -1)
    {
      rtcCopyHitToHitN(rthitn, &potentialHit, args->N, 0);
    }
    else
    {
      rtray.tfar = old_t;
      RTCRayN_tfar(rtrayn, args->N, 0) = old_t;
    }
  }

  // Check the t-value which is further away.
  // This can still be a hit, e.g. if the first hit was behind the ray origin.
  if ((rtray.tnear < t1-err1) & (t1+err1 < rtray.tfar))
  {
    int imask;
    bool mask = 1;
    {
      imask = mask ? -1 : 0;
    }
    
    const Float3 Ng = ray_org+t1*ray_dir-sphere_p;
    potentialHit.Ng_x = Ng[0];
    potentialHit.Ng_y = Ng[1];
    potentialHit.Ng_z = Ng[2];

    RTCFilterFunctionNArguments fargs;
    fargs.valid = (int*)&imask;
    fargs.geometryUserPtr = args->geometryUserPtr;
    fargs.context = args->context;
    fargs.ray = (RTCRayN *)args->rayhit;
    fargs.hit = (RTCHitN*)&potentialHit;
    fargs.N = 1;

    const float old_t = rtray.tfar;
    RTCRayN_tfar(rtrayn, args->N, 0) = t1;
    rtray.tfar = t1;
    rtcFilterIntersection(args,&fargs);

    if (imask == -1)
    {
      rtcCopyHitToHitN(rthitn, &potentialHit, args->N, 0);
    }
    else
    {
      rtray.tfar = old_t;
      RTCRayN_tfar(rtrayn, args->N, 0) = old_t;
    }
  }
  
  assert(rtray.tfar > rtray.tnear);
}


void EmbreeAccelerator::SphereOccludedFunc(const RTCOccludedFunctionNArguments *args)
{
  int* valid = args->valid;
  assert(args->N == 1);
  if (!valid[0]) return;

  RTCRay rtray = rtcGetRayFromRayN(args->ray, args->N, 0);

  unsigned int primID = args->primID;
  const Spheres* spheres = (const Spheres*)args->geometryUserPtr;
  auto &sphere = spheres->spheres[primID];

  Eigen::Map<const Eigen::Vector3f> ray_dir{ &rtray.dir_x };
  Eigen::Map<const Eigen::Vector3f> ray_org{ &rtray.org_x };
  Float3 sphere_p; float sphere_r; std::tie(sphere_p, sphere_r) = spheres->Get(args->primID);
  Float3 v = ray_org - sphere_p;
  ASSERT_NORMALIZED(ray_dir);
  const float A = 1.f; //Dot(ray_dir,ray_dir);
  const float B = 2.0f*Dot(v, ray_dir);
  const float C = Dot(v, v) - Sqr(sphere_r);
  const float Berr = DotAbs(v, ray_dir)*util::Gamma<float>(3);
  const float Cerr = (2.f*Dot(v, v) + Sqr(sphere_r))*util::Gamma<float>(3);
  float t0, t1, err0, err1;
  const bool has_solution = util::Quadratic(A, B, C, 0.f, Berr, Cerr, t0, t1, err0, err1);
  //   float t0, t1;
  //   float err0 = 0.f, err1 = 0.f;
  //   const bool has_solution = Quadratic(A, B, C, t0, t1);
  if (!has_solution)
    return;
  RTCHit potentialHit;
  potentialHit.u = 0.0f;
  potentialHit.v = 0.0f;
  potentialHit.instID[0] = args->context->instID[0];
  potentialHit.geomID = spheres->identifier;
  potentialHit.primID = primID;

  auto checkPotentialHit = [&potentialHit, &rtray, &sphere_p, args](float t)
  {
    int imask = -1;

    potentialHit.Ng_x = rtray.org_x + t * rtray.dir_x - sphere_p[0];
    potentialHit.Ng_y = rtray.org_y + t * rtray.dir_y - sphere_p[1];
    potentialHit.Ng_z = rtray.org_z + t * rtray.dir_z - sphere_p[2];

    RTCFilterFunctionNArguments fargs;
    fargs.valid = (int*)&imask;
    fargs.geometryUserPtr = args->geometryUserPtr;
    fargs.context = args->context;
    fargs.ray = args->ray;
    fargs.hit = (RTCHitN*)&potentialHit;
    fargs.N = 1;

    const float old_t = rtray.tfar;
    RTCRayN_tfar(args->ray, args->N, 0) = t;
    rtray.tfar = t;
    rtcFilterOcclusion(args, &fargs);

    // Did the filter function accept the hit?
    if (imask == -1)
    {
      rtray.tfar = -std::numeric_limits<float>::infinity();
      RTCRayN_tfar(args->ray, args->N, 0) = -std::numeric_limits<float>::infinity();
    }
    else
    {
      rtray.tfar = old_t;
      RTCRayN_tfar(args->ray, args->N, 0) = old_t;
    }
  };

  if ((rtray.tnear < t0 - err0) & (t0 + err0 < rtray.tfar))
  {
    checkPotentialHit(t0);
  }

  if ((rtray.tnear < t1 - err1) & (t1 + err1 < rtray.tfar))
  {
    checkPotentialHit(t1);
  }
}


Box EmbreeAccelerator::GetSceneBounds() const
{
  RTCBounds rtb;
  rtcGetSceneBounds(rtscene, &rtb);
  Box b;
  if (std::isfinite(rtb.lower_x))
  {
    // If there is geometry, all of these should be finite.
    b.min[0] = rtb.lower_x;
    b.max[0] = rtb.upper_x;
    b.min[1] = rtb.lower_y;
    b.max[1] = rtb.upper_y;
    b.min[2] = rtb.lower_z;
    b.max[2] = rtb.upper_z;
  }
  else
  {
    // If the scene is empty
    b.min = b.max = decltype(b.max)::Zero();
  }
  return b;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif