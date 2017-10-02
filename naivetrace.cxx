#include "image.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
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
  
  int xres = scene.GetCamera().xres;
  int yres = scene.GetCamera().yres;
  
  Image bm(xres, yres);
  ImageDisplay display;
  auto time_of_last_display = std::chrono::steady_clock::now();

  //Bdpt algo(scene);
  Raytracing algo(scene);
  //NormalVisualizer algo(scene);

  SpectralImageBuffer buffer{xres, yres};
  
  const int nsmpl = 4;
  
  int image_conversion_y_start = 0;
  std::cout << std::endl;
  std::cout << "Rendering ..." << std::endl;
  for (int y=0;y<yres;y++) 
  {
    for (int x=0;x<xres;x++) 
    {    
      int pixel_index = scene.GetCamera().PixelToUnit({x,y});
      for(int i=0;i<nsmpl;i++)
      {
        Spectral smpl = algo.MakePrettyPixel(pixel_index);
        buffer.Insert(pixel_index, smpl);
      }
    }
    
    auto time = std::chrono::steady_clock::now();
    if (time - time_of_last_display > std::chrono::seconds(1))
    {
      buffer.ToImage(bm, image_conversion_y_start, y);
      image_conversion_y_start = y;
      display.show(bm);
      time_of_last_display = time;
    }
  }

  buffer.ToImage(bm, image_conversion_y_start, yres);
  
  bm.write("raytrace.tga");

  return 1;
}
