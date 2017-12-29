#ifndef TRIANGLE_HXX
#define TRIANGLE_HXX

#include"primitive.hxx"
#include "sampler.hxx"

class Triangle : public Primitive
{
protected:
	Double3 p[3];
public:
	Triangle(const Double3 &a,const Double3 &b,const Double3 &c) 
	{
		p[0]=a; p[1]=b; p[2]=c;
	}

  bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const override;
  virtual void GetLocalGeometry(
      const HitId &hit,
      Double3 &hit_point,
      Double3 &normal,
      Double3 &shading_normal) const override;
	virtual Box CalcBounds() const override;
	virtual HitId SampleUniformPosition(Sampler &sampler) const override;
  virtual double Area() const override;
};





class TexturedSmoothTriangle : public Triangle
{
  Double3 n[3];
	Double3 uv[3];
public:
  TexturedSmoothTriangle(
      const Double3 &a,const Double3 &b,const Double3 &c,
      const Double3 &na,const Double3 &nb,const Double3 &nc,
      const Double3 &uva,const Double3 &uvb,const Double3 &uvc)
    : Triangle(a,b,c),
      n{na, nb, nc},
      uv{uva, uvb, uvc}
  {
  }

	virtual Double3 GetUV(const HitId &hit) const override
	{
    Double3 res = uv[0]*hit.barry[0]+
           uv[1]*hit.barry[1] +
           uv[2]*hit.barry[2];
		return res;
	}

  virtual void GetLocalGeometry(
      const HitId &hit,
      Double3 &hit_point,
      Double3 &normal,
      Double3 &shading_normal) const override
  {
    hit_point = hit.barry[0] * p[0] +
                hit.barry[1] * p[1] +
                hit.barry[2] * p[2];
    normal = Normalized(Cross(p[1]-p[0],p[2]-p[0]));
    shading_normal = Normalized(
                     n[0]*hit.barry[0]+
                     n[1]*hit.barry[1]+
                     n[2]*hit.barry[2]);
  }
};


#endif