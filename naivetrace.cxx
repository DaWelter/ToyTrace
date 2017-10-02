#include "image.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
#include <chrono>
#include <atomic>
#include <thread>

static constexpr int COMM_WAITING = 0;
static constexpr int COMM_HALT = 1;
static constexpr int COMM_GO   = 2;


void ThreadWorker(std::atomic_int &shared_pixel_index, int end_index, SpectralImageBuffer &buffer, const Scene &scene, std::atomic_int &shared_comms)
{
  Raytracing algo(scene);
  while (true)
  {
    int comms = shared_comms.load();
    if (comms == COMM_HALT)
    {
      shared_comms.store(COMM_WAITING);
    }
    else if (comms == COMM_WAITING)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    else if (comms == COMM_GO)
    {
      const int pixel_stride = 64 / sizeof(Spectral);
      int pixel_index = shared_pixel_index.fetch_add(pixel_stride);
      if (pixel_index >= end_index)
        break;
      int current_end_index = std::min(pixel_index + pixel_stride, end_index);
      for (; pixel_index < current_end_index; ++pixel_index)
      {
        const int nsmpl = 64;
        for(int i=0;i<nsmpl;i++)
        {
          Spectral smpl = algo.MakePrettyPixel(pixel_index);
          buffer.Insert(pixel_index, smpl);
        }
      }
    }
  }
}


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
  int num_pixels = xres*yres;
  
  Image bm(xres, yres);
  ImageDisplay display;

  //Bdpt algo(scene);
  Raytracing algo(scene);
  //NormalVisualizer algo(scene);

  SpectralImageBuffer buffer{xres, yres};
  
  const int num_threads = 4;
  
  auto start_time = std::chrono::steady_clock::now();
  
  int image_conversion_y_start = 0;
  
  std::atomic_int shared_pixel_index(0);
  
  std::vector<std::atomic_int> thread_comm(num_threads);
  std::vector<std::thread> threads;
  
  for (int i=0; i<num_threads; ++i)
  {
    thread_comm[i].store(COMM_GO);
    threads.push_back(std::thread{
      ThreadWorker, std::ref(shared_pixel_index), num_pixels, std::ref(buffer), std::cref(scene), std::ref(thread_comm[i])
    });
  }
  std::cout << std::endl;
  std::cout << "Rendering ..." << std::endl;

  while (shared_pixel_index.load() < num_pixels)
  {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    bool ok = false;
    for (auto &comm : thread_comm)
      comm.store(COMM_HALT);
    for (auto &comm : thread_comm)
    {
      while (comm.load() != COMM_WAITING && shared_pixel_index.load() < num_pixels)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    int pixel_index = shared_pixel_index.load();
    int y = pixel_index/xres - 1;
    buffer.ToImage(bm, image_conversion_y_start, y);
    image_conversion_y_start = y;

    for (auto &comm : thread_comm)
      comm.store(COMM_GO);    
    
    display.show(bm);
  }
  
  for (int i=0; i<threads.size(); ++i)
    threads[i].join();

  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Rendering time: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

  buffer.ToImage(bm, image_conversion_y_start, yres);
  
  bm.write("raytrace.tga");

  return 0;
}
