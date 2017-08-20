#ifndef SHADER_HXX
#define SHADER_HXX

#include"vec3f.hxx"
#include"texture.hxx"

class Ray;
class Scene;

class Shader
{
public:
	virtual Double3 Shade(Ray &ray,Scene *scene) = 0;
};

class FlatShader : public Shader
{
	Double3 col;
public:
	FlatShader(const Double3 &col) : col(col) {}
	virtual Double3 Shade(Ray &ray,Scene *scene) { return col; }
};

class EyeLightShader : public Shader
{
	Double3 col;
public:
	EyeLightShader(const Double3 &col) : col(col) {}
	virtual Double3 Shade(Ray &ray,Scene *scene);
};


class TexturedEyeLightShader : public EyeLightShader
{
	Texture tex;
public:
	TexturedEyeLightShader(const Double3 &col,const char *filename) :EyeLightShader(col),tex(filename)  {}
	virtual Double3 Shade(Ray &ray,Scene *scene);
};


class ReflectiveEyeLightShader : public Shader
{
	Double3 col;
	double reflectivity;
public:
	ReflectiveEyeLightShader(const Double3 &col,double reflectivity)
		: col(col), reflectivity(reflectivity)
	{ }
	virtual Double3 Shade(Ray &ray,Scene *scene);
};

class PhongShader : public Shader
{
	Double3 ca,cd,cs;
	double kr,ka,kd,ks,ke;
public:
	PhongShader(const Double3 &ambient,
				const Double3 &diffuse,
				const Double3 &specular,
				double ka,double kd,double ks,double ke,double kr) 
				:	ca(ambient),
					cd(diffuse),
					cs(specular),
					ka(ka),kd(kd),ks(ks),ke(ke),kr(kr)
	{}
	virtual Double3 Shade(Ray &ray,Scene *scene);
};

#endif
