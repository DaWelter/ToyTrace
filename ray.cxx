#include "ray.hxx"
#include "primitive.hxx"


Double3 SurfaceHit::Normal() const
{
  return primitive ? primitive->GetNormal(*this) : Double3();
}