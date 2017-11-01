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
    ASSERT_NORMALIZED(ray.dir);
    Double3 q = center-ray.org;
    double t = Dot(q,ray.dir);
    if(t>ray_length+Epsilon+radius || t<-radius+Epsilon) return false;
    Double3 p = ray.org + t*ray.dir;
    double dd = LengthSqr(p-center);
    if(dd>=radius*radius) return false;
    double dt = std::sqrt(radius*radius-dd);
    t1=t+dt;
    t2=t-dt;
    if (t1<0. || t2>ray_length) return false;
    return true;
  }
  

  /* To fix precision issues I tried to move the intersection point closer
   * to the sphere surface by refining t via Newton iterations.
   * That worked. However since the introduction of Duff's
   * orthonormal basis code the precision problems are gone either way. 
     To use like this:
   t1 = Refine(t1, Dot(q,q), -Dot(q, ray.dir), Dot(ray.dir, ray.dir));
   t2 = Refine(t2, Dot(q,q), -Dot(q, ray.dir), Dot(ray.dir, ray.dir));

  inline double Refine(double t, double qq, double qv, double vv) const
  {
    // Newton iterations.
    for (int n = 0; n < 5; ++n)
    {
      t = (t*t*vv - qq + radius*radius) / (2.*qv + 2.*t*vv);
    }
    return t;
  }
  */  
  
  bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const override
  {
    double t1, t2;
    if(!PotentialDistances(ray, ray_length, t1, t2))
      return false;
    double t = t2>0. ? t2 : t1;
    if (t < 0. || t > ray_length)
      return false;
    hit.primitive = this;
    ray_length = t;
    hit.barry = ray.org + t * ray.dir;
    return true;
  }

  bool CheckIsNearHit(const Ray &ray, double t, const Double3 &p, const HitId &to_ignore) const
  {
    // TODO: use error estimates of t and to_ignore.barry. Because this is really unreliable.
    const double tol = 100. * Epsilon * radius;
    double uu = LengthSqr(to_ignore.barry - p);
    return uu < tol*tol;
  }
  
  bool IntersectIfNotIgnored(const Ray &ray, double t, double &ray_length, HitId &hit, const HitId &to_ignore1, const HitId &to_ignore2) const
  {
    if (t < 0. || t>ray_length) return false;
    Double3 p = ray.PointAt(t);
    if (to_ignore1.primitive == this && CheckIsNearHit(ray, t, p, to_ignore1)) return false;
    if (to_ignore2.primitive == this && CheckIsNearHit(ray, t, p, to_ignore2)) return false;
    hit.primitive = this;
    ray_length = t;
    hit.barry = p;
    return true;
  }
  
  virtual bool Intersect(const Ray &ray, double &ray_length, HitId &hit, const HitId &to_ignore1, const HitId &to_ignore2) const override
  {
    double t, t1, t2;
    if (!PotentialDistances(ray, ray_length, t1, t2))
      return false;
    if (IntersectIfNotIgnored(ray, t2, ray_length, hit, to_ignore1, to_ignore2)) return true;
    if (IntersectIfNotIgnored(ray, t1, ray_length, hit, to_ignore1, to_ignore2)) return true;
    return false;
  }
  
	virtual Double3 GetNormal(const HitId &hit) const
	{
	  Double3 n(hit.barry-center);
	  Normalize(n);
	  return n;
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
