#include "ray.hxx"
#include "primitive.hxx"
#include "util.hxx"


SurfaceInteraction::SurfaceInteraction(const HitId& _hitid)
  : hitid(_hitid)
{
  assert(hitid.primitive);
  hitid.primitive->GetLocalGeometry(hitid, this->pos, this->geometry_normal, this->smooth_normal);
}



RaySurfaceIntersection::RaySurfaceIntersection(const HitId& _hitid, const RaySegment &_incident_segment)
  : SurfaceInteraction(_hitid)
{
  double sign = -Sign(Dot(_incident_segment.ray.dir, geometry_normal));
  normal = sign*geometry_normal;
  shading_normal = sign*smooth_normal;
  assert(LengthSqr(normal)>0.9);
  assert(LengthSqr(shading_normal)>0.9);
}


const Primitive& SurfaceInteraction::primitive() const
{ 
  assert((bool)hitid); 
  return *hitid.primitive; 
}

const Shader& SurfaceInteraction::shader() const
{ 
  assert((bool)hitid && hitid.primitive->shader); 
  return *hitid.primitive->shader; 
}
  