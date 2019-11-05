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
