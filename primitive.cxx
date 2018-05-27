#include "triangle.hxx"
#include "sampler.hxx"
#include "sphere.hxx"

bool Triangle::Intersect(const Ray &ray, double tnear, double &tfar, HitId &hit) const
{
  Double3 n;
  Double3 edge[3];
  edge[0] = p[1]-p[0];
  edge[1] = p[2]-p[1];
  edge[2] = p[0]-p[2];
  n = Cross(edge[0],-edge[2]);
  Normalize(n);

  double d = Dot(p[0],n);
  double r = Dot(ray.dir,n);
  if(fabs(r)<Epsilon) return false;
  double s = (d-Dot(ray.org,n))/r; // s is the fraction of ray dir, where the hit point is.
  if(s<=tnear || s>tfar) return false;
  Double3 q = ray.org+s*ray.dir;

  double edge_func[3];
  for(int i=0; i<3; i++)
  {
    Double3 w = Cross(edge[i], q-p[i]);
    edge_func[i] = Dot(n, w);
    if(edge_func[i]<0.) return false;
  }
  double normalization = 1./(edge_func[0]+edge_func[1]+edge_func[2]);
  hit.barry[0] = edge_func[1] * normalization;
  hit.barry[1] = edge_func[2] * normalization;
  hit.barry[2] = edge_func[0] * normalization;
  assert(hit.barry.allFinite());
  tfar = s;
  hit.primitive = this;
  return true;
}


void Triangle::GetLocalGeometry(
  const HitId &hit,
  Double3 &hit_point,
  Double3 &normal,
  Double3 &shading_normal) const
{
  hit_point = hit.barry[0] * p[0] +
              hit.barry[1] * p[1] +
              hit.barry[2] * p[2];
  shading_normal = normal =
      Normalized(Cross(p[1]-p[0],p[2]-p[0]));
  assert  (hit_point.allFinite());
  assert  (normal.allFinite());
}


Box Triangle::CalcBounds() const
{
  Box box;
  for(int i=0; i<3; i++) box.Extend(p[i]);
  return box;
}

HitId Triangle::SampleUniformPosition(Sampler &sampler) const
{
  HitId hit{
    this,
    SampleTrafo::ToTriangleBarycentricCoords(sampler.UniformUnitSquare())
  };
  return hit;
}

double Triangle::Area() const
{
  return 0.5*Length(Cross(p[1]-p[0],p[2]-p[0]));
}




HitId Sphere::SampleUniformPosition(Sampler &sampler) const
{
  Double3 pos = center + radius*SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
  return HitId{
    this,
    pos
  };
}


double Sphere::Area() const
{
  return Sqr(radius)*UnitSphereSurfaceArea;
}
