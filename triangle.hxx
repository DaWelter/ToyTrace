#ifndef TRIANGLE_HXX
#define TRIANGLE_HXX

#include"primitive.hxx"

class Triangle : public Primitive
{
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
		for(int i=0; i<3; i++) 
    {
			Double3 w = Cross(n,edge[i]);
			double t = Dot(q-p[i],w);
			if(t<Epsilon) return false;
		}
		ray_length = s;
    hit.primitive = this;
    return true;
	}

	virtual Double3 GetNormal(const HitId &hit) const override
	{
		Double3 n = Cross(p[1]-p[0],p[2]-p[0]);
		Normalize(n);
		return n;
	}
	
	virtual Box CalcBounds() const override
	{
		Box box;
		for(int i=0; i<3; i++) box.Extend(p[i]);
		return box;
	}
};




class SmoothTriangle : public Primitive
{
	Double3 p[3],n[3];
public:
	SmoothTriangle(const Double3 &a,const Double3 &b,const Double3 &c,
				   const Double3 &na,const Double3 &nb,const Double3 &nc)
	{
		p[0]=a;  p[1]=b;  p[2]=c;
		n[0]=na; n[1]=nb; n[2]=nc;
	}

	bool Intersect(const Ray &ray, double &ray_length, HitId &hit) const override
	{
		Double3 n;
		Double3 ab,ac,a;
		a = p[0];
		ab = p[1]-p[0];
		ac = p[2]-p[0];
		n = Cross(ab,ac);
		Normalize(n);

		double d = Dot(p[0],n);
		double r = Dot(ray.dir,n); 
		if(fabs(r)<Epsilon) return false;
		double s = (d-Dot(ray.org,n))/r;
		if(s<Epsilon || s>ray_length+Epsilon) return false;
		Double3 q = ray.org+s*ray.dir;
		
		double nmax;
		int plane = 0;
		nmax = n[0]*n[0];
		if(n[1]*n[1] > nmax) {  nmax = n[1]*n[1]; plane=1; }
		if(n[2]*n[2] > nmax) {  nmax = n[2]*n[2]; plane=2; }
		double qx,qy,abx,aby,acx,acy;
		switch(plane)
		{
			case 0: qx=q[1]-a[1]; qy=q[2]-a[2]; abx=ab[1]; aby=ab[2]; acx=ac[1]; acy=ac[2]; break;
			case 1: qx=q[0]-a[0]; qy=q[2]-a[2]; abx=ab[0]; aby=ab[2]; acx=ac[0]; acy=ac[2]; break;
			case 2: qx=q[0]-a[0]; qy=q[1]-a[1]; abx=ab[0]; aby=ab[1]; acx=ac[0]; acy=ac[1]; break;
		}
		double det  = abx*acy-aby*acx;
		double detu = qx *acy-qy *acx;
		double detv = abx*qy -aby*qx;
		if(fabs(det)<Epsilon) {
			std::cerr << "Error: Triangle Degenerate" << std::endl;
			return false;
		}
		double u = detu/det;
		double v = detv/det;
		if(u<-Epsilon || v<-Epsilon || u+v>1.+Epsilon) return false;
		hit.barry = Double3(u,v,1.-u-v);
		ray_length = s;
		hit.primitive = this;
		return true;
	}

	Double3 GetNormal(const HitId &hit) const override
	{
		Double3 normal = n[0]*hit.barry[0]+
					   n[1]*hit.barry[1] +
				       n[2]*hit.barry[2];
// 		Double3 normal = Cross(p[1]-p[0],p[2]-p[0]) ;
		Normalize(normal);
		return normal;
	}

	Box CalcBounds() const override
	{
		Box box;
		for(int i=0; i<3; i++) box.Extend(p[i]);
		return box;
	}
};


class TexturedSmoothTriangle : public SmoothTriangle
{
	Double3 uv[3];
public:
	TexturedSmoothTriangle(	const Double3 &a,const Double3 &b,const Double3 &c,
							const Double3 &na,const Double3 &nb,const Double3 &nc,
							const Double3 &uva,const Double3 &uvb,const Double3 &uvc)
								: SmoothTriangle(a,b,c,na,nb,nc)
	{ uv[0]=uva; uv[1]=uvb; uv[2]=uvc; }

	virtual Double3 GetUV(const HitId &hit) const override
	{
		Double3 res = uv[0]*hit.barry[0]+
				   uv[1]*hit.barry[1] + 
				   uv[2]*hit.barry[2];
		return res;
	}
};


#endif