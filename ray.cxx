#include "ray.hxx"
#include "primitive.hxx"


Double3 SurfaceHit::Normal(const Double3 &up_dir) const
{
  Double3 n = primitive ? primitive->GetNormal(*this) : Double3();
  double sign = Dot(up_dir, n);
  n = sign > 0 ? n : (-n).eval();
  return n;
}