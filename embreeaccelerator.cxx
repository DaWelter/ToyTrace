#include "embreeaccelerator.hxx"
#include "scene.hxx"
#include "ray.hxx"
#include "vec3f.hxx"

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


void EmbreeAccelerator::Build()
{
  rtcCommitScene (rtscene);
}


void EmbreeAccelerator::Add(Mesh& mesh)
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


void EmbreeAccelerator::Add(Spheres& spheres)
{
  RTCGeometry rtgeom = rtcNewGeometry(rtdevice, RTC_GEOMETRY_TYPE_USER);
  unsigned int id = rtcAttachGeometry(rtscene, rtgeom); 
  spheres.identifier = id;
  rtcSetGeometryUserData(rtgeom, (void*)&spheres);
  rtcSetGeometryUserPrimitiveCount(rtgeom, spheres.NumSpheres());
  rtcSetGeometryBoundsFunction(rtgeom,SphereBoundsFunc,nullptr);
  rtcSetGeometryIntersectFunction(rtgeom,SphereIntersectFunc);
  //rtcSetGeometryOccludedFunction (geom,sphereOccludedFunc);
  rtcCommitGeometry(rtgeom);
  rtcReleaseGeometry(rtgeom);
}


bool EmbreeAccelerator::FirstIntersection(const Ray &ray, double tnear, double &ray_length, RaySurfaceIntersection &intersection) const
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
  rtray.tfar = std::numeric_limits<float>::infinity();
  rtray.mask = -1;
  rtray.id = 0;
  rtray.flags = 0;
  RTCHit &rthit = rtrayhit.hit;
  rthit.geomID = RTC_INVALID_GEOMETRY_ID;
  rthit.primID = RTC_INVALID_GEOMETRY_ID;
  rthit.instID[0] = RTC_INVALID_GEOMETRY_ID;
  
  rtcIntersect1(rtscene, &context, &rtrayhit);
  if (rthit.geomID == RTC_INVALID_GEOMETRY_ID)
    return false;
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


void EmbreeAccelerator::FirstIntersectionTriangle(const RTCHit &rthit, const Ray &ray, RaySurfaceIntersection &intersection) const
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


void EmbreeAccelerator::FirstIntersectionSphere(const RTCHit &rthit, const Ray &ray, RaySurfaceIntersection &intersection) const
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
  intersection.tex_coord = ToSphericalCoordinates(delta);
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
  const float Berr = DotAbs(v, ray_dir)*Gamma<float>(3);
  const float C = Dot(v,v) - Sqr(sphere_r);
  const float Cerr = (2.f*Dot(v,v) + Sqr(sphere_r))*Gamma<float>(3);
  float t0, t1, err0, err1;
  const bool has_solution = Quadratic(A, B, C, 0.f, Berr, Cerr, t0, t1, err0, err1);
  if (!has_solution)
    return;
  RTCHit potentialHit;
  potentialHit.u = 0.0f;
  potentialHit.v = 0.0f;
  potentialHit.instID[0] = args->context->instID[0];
  potentialHit.geomID = spheres->identifier;
  potentialHit.primID = primID;
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
    fargs.geometryUserPtr = nullptr;
    fargs.context = args->context;
    fargs.ray = (RTCRayN *)args->rayhit;
    fargs.hit = (RTCHitN*)&potentialHit;
    fargs.N = 1;

    const float old_t = rtray.tfar;
    RTCRayN_tfar(rtrayn, args->N, 0) = t0;
    rtray.tfar = t0;
    rtcFilterIntersection(args,&fargs);

    if (imask == -1)
      rtcCopyHitToHitN(rthitn, &potentialHit, args->N, 0);
    else
      rtray.tfar = old_t;
  }

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
    fargs.geometryUserPtr = nullptr;
    fargs.context = args->context;
    fargs.ray = (RTCRayN *)args->rayhit;
    fargs.hit = (RTCHitN*)&potentialHit;
    fargs.N = 1;

    const float old_t = rtray.tfar;
    RTCRayN_tfar(rtrayn, args->N, 0) = t1;
    rtray.tfar = t1;
    rtcFilterIntersection(args,&fargs);

    if (imask == -1)
      rtcCopyHitToHitN(rthitn, &potentialHit, args->N, 0);
    else
      rtray.tfar = old_t;
  }
}


Box EmbreeAccelerator::GetSceneBounds() const
{
  RTCBounds rtb;
  rtcGetSceneBounds(rtscene, &rtb);
  Box b;
  b.min[0] = rtb.lower_x;
  b.max[0] = rtb.upper_x;
  b.min[1] = rtb.lower_y;
  b.max[1] = rtb.upper_y;
  b.min[2] = rtb.lower_z;
  b.max[2] = rtb.upper_z;
  return b;
}
