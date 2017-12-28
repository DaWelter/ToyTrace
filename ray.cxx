#include "ray.hxx"
#include "primitive.hxx"


RaySurfaceIntersection::RaySurfaceIntersection(const HitId& _hitid, const RaySegment &_incident_segment)
  : hitid(_hitid)
{
  assert(hitid.primitive);
  hitid.primitive->GetLocalGeometry(hitid, this->pos, this->normal, this->shading_normal);
  double sign = Dot(-_incident_segment.ray.dir, normal) > 0. ? 1. : -1;
  volume_normal = normal;
  normal *= sign;
  shading_normal *= sign;
  assert(LengthSqr(volume_normal)>0.9);
  assert(LengthSqr(shading_normal)>0.9);
}


const Primitive& RaySurfaceIntersection::primitive() const
{ 
  assert((bool)hitid); 
  return *hitid.primitive; 
}

const Shader& RaySurfaceIntersection::shader() const
{ 
  assert((bool)hitid && hitid.primitive->shader); 
  return *hitid.primitive->shader; 
}
  