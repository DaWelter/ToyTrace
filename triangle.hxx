#ifndef TRIANGLE_HXX
#define TRIANGLE_HXX

#include"primitive.hxx"

class Triangle : public Primitive
{
protected:
	Double3 p[3];
public:
	Triangle(const Double3 &a,const Double3 &b,const Double3 &c) 
	{
		p[0]=a; p[1]=b; p[2]=c;
	}

  bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const override
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
    double s = (d-Dot(ray.org,n))/r;
    if(s<Epsilon || s>ray_length+Epsilon) return false;
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
    ray_length = s;
    hit.primitive = this;
    return true;
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
    shading_normal = normal =
        Normalized(Cross(p[1]-p[0],p[2]-p[0]));
  }

	virtual Box CalcBounds() const override
	{
		Box box;
		for(int i=0; i<3; i++) box.Extend(p[i]);
		return box;
	}
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