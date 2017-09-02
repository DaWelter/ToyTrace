#include "image.hxx"
#include "scene.hxx"
#include <chrono>


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
  ImageDisplay display;
  auto time_of_last_display = std::chrono::steady_clock::now();

  const int nsmpl = 4;
  
  Sampler sampler;

  std::cout << std::endl;
  std::cout << "Rendering ..." << std::endl;
  for (int y=0;y<scene.camera->yres;y++) 
  {
    for (int x=0;x<scene.camera->xres;x++) 
    {
      scene.camera->current_pixel_x = x;
      scene.camera->current_pixel_y = y;
      
      Double3 col(0);
      for(int i=0;i<nsmpl;i++)
      {
        col += scene.RayTrace(sampler);
      }
      col *= 1.0/nsmpl;
      Clip(col[0],0.,1.); Clip(col[1],0.,1.); Clip(col[2],0.,1.);
      bm.set_pixel(x, bm.height() - y, col[0]*255.99999999,col[1]*255.99999999,col[2]*255.99999999);
    }
    
    auto time = std::chrono::steady_clock::now();
    if (time - time_of_last_display > std::chrono::seconds(1))
    {
      display.show(bm);
      time_of_last_display = time;
    }
  }

  bm.write("raytrace.tga");

  return 1;
}
