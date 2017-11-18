#ifndef SPHERE_HXX
#define SPHERE_HXX

#include "primitive.hxx"

class Sphere : public Primitive
{
  Double3 center;
  double radius;
public:
	Sphere(Double3 _center,double _radius)
	: center(_center),radius(_radius),Primitive()
  {}

  inline bool PotentialDistances(const Ray &ray, double ray_length, double &t1, double &t2) const
  {
    Double3 q = ray.org-center;
    double C = Dot(q,q) - radius * radius;
    double A = Dot(ray.dir,ray.dir);
    double B = 2.*Dot(ray.dir,q);
    double t = -B*0.5/A;
    double under_the_sqrt = B*B - 4.*A*C;
    if (under_the_sqrt < 0.)
      return false;
    double dt = std::sqrt(under_the_sqrt)*0.5/A;
    t1=t+dt;
    t2=t-dt;
    // The cases where 1) hit point lies behind the start point, 2) hit point lies beyond the end of the ray.
    if (t1-RAY_EPSILON<0. || t2+RAY_EPSILON>ray_length) return false;
    return true;
  }

  void Intersect(const Ray &ray, double ray_length, HitVector &hits) const override
  {
    double t1, t2;
    if(!PotentialDistances(ray, ray_length, t1, t2))
      return;
    if (t2-RAY_EPSILON > 0.) // Hit in front of sphere.
    {
      HitRecord r{{this, ray.PointAt(t2)}, t2};
      ReProjectHitPoint(r.barry);
      hits.push_back(r);
    }
    if (t1+RAY_EPSILON < ray_length)
    {
      HitRecord r{{this, ray.PointAt(t1)}, t1};
      ReProjectHitPoint(r.barry);
      hits.push_back(r);
    }
  }
  
  bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const override
  {
    double t1, t2;
    if(!PotentialDistances(ray, ray_length, t1, t2))
      return false;
    double t = t2;
    if (t-RAY_EPSILON <= 0.)
    {
      t = t1;
      if (t + RAY_EPSILON > ray_length)
        return false;
    }
    hit = HitId{ this, ray.PointAt(t) };
    ReProjectHitPoint(hit.barry);
    ray_length = t;
    return true;
  }
  
  inline void ReProjectHitPoint(Double3 &pos) const
  {
    // The error estimate is probably fine. However it neglects the
    // error due to computing org + t * dir. Thus it can only be
    // used as a measure of the offset from the true sphere surface.
    // It does not account for the error of the distance from the true hit point.
    Double3 q = pos - center;
    q *= (radius / Length(q));
    //error = Gamma(5)*q.array().abs().maxCoeff();
    pos = q + center;
    //error += Gamma(1)*(pos.array().abs().maxCoeff() + error); // PBRT pg 219. (Error of addition).
  }
  
  virtual bool CompareEqual(const HitId &hit_this, const HitId &hit_other) const
  {
    return LengthSqr(hit_this.barry - hit_other.barry) < RAY_EPSILON*RAY_EPSILON;
  }

  virtual Double3 GetUV(const HitId &hit) const override
  {
    // From kartesian to spherical coordinates.
    double z = hit.barry[2];
    double y = hit.barry[1];
    double x = hit.barry[0];
    double r = Length(hit.barry);
    double theta = std::acos(y/r);
    double phi;
    double ax = std::abs(x);
    double az = std::abs(z);
    if (ax > az)
    {
      phi = std::atan2(z,ax);
      phi = (x > 0.) ? phi : Pi - phi;
    }
    else
    {
      phi = std::atan2(x,az);
      phi = (z > 0.) ? Pi/2.-phi : 3./2.*Pi + phi;
    }
    // To UV
    theta /= Pi;
    phi   /= 2.*Pi;
    return Double3{phi, theta, 0.};
  }

  virtual void GetLocalGeometry(
      const HitId &hit,
      Double3 &hit_point,
      Double3 &normal,
      Double3 &shading_normal) const override
  {
    hit_point = hit.barry;
    shading_normal = normal = Normalized(hit.barry-center);
  }

	virtual Box CalcBounds() const
	{
		Box box;
		box.min = center - Double3(radius+Epsilon);
		box.max = center + Double3(radius+Epsilon);
		return box;
	}
};

#endif
