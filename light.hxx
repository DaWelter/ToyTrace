#ifndef LIGHT_HXX
#define LIGHT_HXX

#include"vec3f.hxx"

class Light
{
public:
	virtual bool Illuminate(Ray &ray,Double3 &intensity) = 0;
};



class DirectionalLight : public Light
{
	Double3 dir;
	Double3 col;
public:
	DirectionalLight(const Double3 &col,const Double3 &dir) 
		: dir(dir) , col(col)
	{}
	virtual bool Illuminate(Ray &ray,Double3 &intensity)
	{
		intensity=col; 
		ray.dir=-dir;
		ray.t = 1000.;
		return true;
	}
};

class PointLight : public Light
{
	Double3 col;
	Double3 pos;
public:
	PointLight(const Double3 &col,const Double3 &pos)
		: col(col),pos(pos)
	{}

	virtual bool Illuminate(Ray &ray,Double3 &intensity)
	{
		double c1,c2,c3;
		c1 = 1.;
		c2 = 0.5;
		c3 = 0.;
		Double3 dist = pos-ray.org;
		ray.t = Length(dist)-Epsilon;
		ray.dir = dist/ray.t;
		double attenuation = 1./(c1+ray.t*c2+ray.t*ray.t*c3);
		intensity = col*attenuation;
		return true;
	}
};


class SpotLight : public Light
{
	Double3 pos,dir,col;
	double min,max;
public:
	SpotLight(	const Double3 &_col,
				const Double3 &_pos,
				const Double3 &_dir,
				double _min,
				double _max ) 
				: pos(_pos),dir(_dir),col(_col),min(_min),max(_max) 
	{
		Normalize(dir);
		min *= Pi/180.;
		max *= Pi/180.;
	}

	virtual bool Illuminate(Ray &ray,Double3 &intensity)
	{
		Double3 d = ray.org-pos;
		double l = Length(d);
		d		= d/l;
		ray.dir = -d;
		ray.t   = l-Epsilon;

		double alpha = acos(Dot(d,dir));
		if(alpha>max) return false;
		double weight = (alpha-min)/(max-min);
		Clip(weight,0.,1.);
		weight = 1.-weight;

		double c1,c2,c3;
		c1 = 1.;
		c2 = 0.5;
		c3 = 0.;
		double attenuation = 1./(c1+l*c2+l*l*c3);

		intensity = col * weight * attenuation;
		return true;
	}
};

#endif