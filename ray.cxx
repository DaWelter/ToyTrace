#include "ray.hxx"
#include "primitive.hxx"
#include "util.hxx"
#include "scene.hxx"


RaySegment RaySegment::FromTo(const Double3 &src, const Double3 &dest) 
{
  Double3 delta = dest-src;
  assert(delta.allFinite());
  double l = Length(delta);
  if (std::isinf(l))
  {
    l = delta.cwiseAbs().maxCoeff();
    delta /= l;
    double ll = Length(delta);
    l *= ll;
    delta /= ll;
  }
  else
    delta = l>0 ? (delta / l).eval() : Double3(NaN, NaN, NaN);
  return RaySegment{{src, delta}, l};
}


SurfaceInteraction::SurfaceInteraction(const HitId& _hitid)
  : hitid(_hitid)
{
  assert((bool)hitid);
  hitid.geom->GetLocalGeometry(*this);
}


void RaySurfaceIntersection::SetOrientedNormals(const Double3 &incident)
{
  double sign = -Sign(Dot(incident, geometry_normal));
  normal = sign*geometry_normal;
  shading_normal = sign*smooth_normal;
}


RaySurfaceIntersection::RaySurfaceIntersection(const HitId& _hitid, const RaySegment &_incident_segment)
  : SurfaceInteraction(_hitid)
{
  SetOrientedNormals(_incident_segment.ray.dir);
  assert(LengthSqr(normal)>0.9);
  assert(LengthSqr(shading_normal)>0.9);
}

// const Shader& SurfaceInteraction::GetShader() const
// {
//   assert(scene && hitid);
//   const auto &mat = scene->GetMaterialOf(hitid);
//   return scene->GetShader(mat.shader_index);
// }
// 
// 
// const Medium& SurfaceInteraction::GetMedium() const
// {
//   assert(scene && hitid);
//   const auto &mat = scene->GetMaterialOf(hitid);
//   return scene->GetMedium(mat.medium_index);  
// }
// 
// 
// const RadianceOrImportance::AreaEmitter* SurfaceInteraction::GetEmitter() const
// {
//   assert(scene && hitid);
//   const auto &mat = scene->GetMaterialOf(hitid);
//   return mat.emitter;
// }
