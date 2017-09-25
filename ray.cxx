#include "ray.hxx"
#include "primitive.hxx"


RaySurfaceIntersection::RaySurfaceIntersection(const HitId& _hitid, const RaySegment& _inbound)
  : hitid(_hitid),
    //primitive(hitid.primitive),
    
    //barry(hitid.barry),
    //shader(hitid.primitive ? hitid.primitive->shader : nullptr),
    //dir_out(-inbound.ray.dir),
    pos(_inbound.EndPoint())
{
  Double3 n = hitid.primitive ? hitid.primitive->GetNormal(hitid) : Double3();
  double sign = Dot(-_inbound.ray.dir, n);
  this->normal = sign > 0 ? n : (-n).eval();
  this->volume_normal = n;
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
  