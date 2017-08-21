#include "image.hxx"
#include "scene.hxx"


class SampleGenerator
{
public:
	virtual void GetSamples(int n,double *u,double *v,double *weight) = 0;
};

class RegularSampleGenerator : public SampleGenerator
{
public:
	virtual void GetSamples(int n,double *u,double *v,double *weight)
	{
		n = sqrt((double)n);
		if(n<=0) n=1;
		for(int j=0; j<n; j++)
		for(int i=0; i<n; i++)
		{
			u[j*n+i]= ((double)(i+1))/(n+1);
			v[j*n+i]= ((double)(j+1))/(n+1);
			weight[j*n+i] = 1.0/(n*n);
		}
	}
};


class RandomSampleGenerator : public SampleGenerator
{
public:
	virtual void GetSamples(int n,double *u,double *v,double *weight)
	{
		srand(124312);
		for(int i=0;i<n;i++) 
		{
			u[i] = ((double)rand())/RAND_MAX;
			v[i] = ((double)rand())/RAND_MAX;
			weight[i] = 1.0/n;
		}
	}
};


class StratifiedSampleGenerator : public SampleGenerator
{
public:
	virtual void GetSamples(int n,double *u,double *v,double *weight)
	{
		n = sqrt((double)n);
		srand(124312);
		if(n<=0) n=1;
		for(int j=0; j<n; j++)
		for(int i=0; i<n; i++)
		{
			double dx = ((double)rand())/RAND_MAX;
			double dy = ((double)rand())/RAND_MAX;
			u[j*n+i]= ((double)i+dx)/(n+1);
			v[j*n+i]= ((double)j+dy)/(n+1);
			weight[j*n+i] = 1.0/(n*n);
		}
	}
};


int main(int argc, char *argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <file.nff>" << std::endl;
    exit(0);
  }
  
  Scene scene;
  
  std::cout << "parsing input file " << argv[1]<< std::endl;
  scene.ParseNFF(argv[1]);
  
  std::cout << "building acceleration structure " << std::endl;
  scene.BuildAccelStructure();
  scene.PrintInfo();
  
  Image bm(scene.camera->xres,scene.camera->yres);
  
  SampleGenerator *sampler = new StratifiedSampleGenerator;
  const int nsmpl = 4;
  double usmpl[nsmpl];
  double vsmpl[nsmpl];
  double wsmpl[nsmpl];
  sampler->GetSamples(nsmpl,usmpl,vsmpl,wsmpl);

  Ray ray;
  std::cout << std::endl;
  std:: cout << "rendering line ";
  for (int y=0;y<scene.camera->yres;y++) 
  {
    std::cout << '.';
    std::cout << std::flush;
    for (int x=0;x<scene.camera->xres;x++) 
    {
      Double3 col(0);
      for(int i=0;i<nsmpl;i++)
      {
        scene.camera->InitRay(x+usmpl[i],y+vsmpl[i],ray);
        col += wsmpl[i]*scene.RayTrace(ray);
      }
      Clip(col[0],0.,1.); Clip(col[1],0.,1.); Clip(col[2],0.,1.);
      bm.set_pixel(x, bm.height() - y, col[0]*255.99999999,col[1]*255.99999999,col[2]*255.99999999);
    }
  }
  std::cout<<std::endl;

  bm.write("raytrace.tga");

  return 1;
}
