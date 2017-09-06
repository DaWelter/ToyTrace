#include "ray.hxx"
#include "primitive.hxx"


RaySurfaceIntersection::RaySurfaceIntersection(const HitId& hitid, const RaySegment& inbound)
  : primitive(hitid.primitive),
    barry(hitid.barry),
    shader(hitid.primitive ? hitid.primitive->shader : nullptr),
    dir_out(-inbound.ray.dir),
    pos(inbound.EndPoint())
{
  Double3 n = primitive ? primitive->GetNormal(hitid) : Double3();
  double sign = Dot(dir_out, n);
  this->normal = sign > 0 ? n : (-n).eval();
}