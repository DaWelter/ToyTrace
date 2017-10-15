#include "image.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
#include <chrono>
#include <atomic>
#include <thread>
#include <memory>

static constexpr int SAMPLES_PER_PIXEL = 16;
#ifndef NDEBUG
static constexpr int NUM_THREADS = 0;
#else
static constexpr int NUM_THREADS = 4;
#endif

class Worker
{
  const Scene &scene;
  SpectralImageBuffer &buffer;
  Raytracing algo;
  //NormalVisualizer algo;
  const int pixel_stride = 64 / sizeof(Spectral);
  int num_pixels;
  std::atomic_int &shared_pixel_index;
  std::atomic_int shared_request, shared_state;
public:
  Worker(const Worker &) = delete;
  Worker(Worker &&) = delete;
  
  enum State : int {
    THREAD_WAITING = 0,
    THREAD_RUNNING = 1,
    THREAD_TERMINATED = 2,
  };

  enum Request : int {
    REQUEST_NONE      = -1,
    REQUEST_TERMINATE = 0,
    REQUEST_HALT      = 1,
    REQUEST_GO        = 2,
  };
  
  Worker(std::atomic_int &_shared_pixel_index, SpectralImageBuffer &_buffer, const Scene &_scene) :
    shared_request(REQUEST_NONE),
    shared_state(THREAD_WAITING),
    shared_pixel_index(_shared_pixel_index),
    buffer(_buffer),
    scene(_scene),
    algo(_scene)
  {
    num_pixels = scene.GetCamera().xres * scene.GetCamera().yres;
    //std::cout << "thread ctor " << this << std::endl;
  }
  
  void IssueRequest(Request cmd)
  {
    shared_request.store(cmd);
  }

  State GetState() const
  {
    return (State)shared_state.load();
  }
  
  static void Run(Worker *worker)
  {
    //std::cout << "thread running ... " << worker << std::endl;
    bool run = true;
    while (run)
    {
      worker->ProcessCommandRequest();
      worker->ExecuteState(run);
    }
  }
  
private:
  void SetState(State _state)
  {
    shared_state.store(_state);
  }
  
  void ProcessCommandRequest()
  {
    Request comms = (Request)shared_request.load();
    switch(comms)
    {
      case REQUEST_NONE:
        break;
      case REQUEST_HALT:
        SetState(THREAD_WAITING); 
        shared_request.store(REQUEST_NONE);
        break;
      case REQUEST_GO:
        SetState(THREAD_RUNNING); 
        shared_request.store(REQUEST_NONE);
        break;
      case REQUEST_TERMINATE:
        SetState(THREAD_TERMINATED); 
        shared_request.store(REQUEST_NONE);
        break;
    }
  }
  
  void ExecuteState(bool &run)
  {
    int state = GetState();
    //std::cout << "state within thread " << this << " is " << state << std::endl;
    switch (state) 
    {
      case THREAD_WAITING:
        StateWaiting();
        break;
      case THREAD_RUNNING: 
        StateRunning();
        break;
      case THREAD_TERMINATED:
        run = false;
        break;
    }
  }
  
  void StateWaiting()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  
  void StateRunning()
  {
    int pixel_index = shared_pixel_index.fetch_add(pixel_stride);
    if (pixel_index >= num_pixels)
    {
      SetState(THREAD_WAITING);
    }
    else
    {
      Render(pixel_index);
    }
  }
  
  void Render(int pixel_index)
  {
    int current_end_index = std::min(pixel_index + pixel_stride, num_pixels);
    for (; pixel_index < current_end_index; ++pixel_index)
    {
      const int nsmpl = SAMPLES_PER_PIXEL;
      for(int i=0;i<nsmpl;i++)
      {
        Spectral smpl = algo.MakePrettyPixel(pixel_index);
        buffer.Insert(pixel_index, smpl);
      }
    }
  }
};


void IssueRequest(std::vector<std::unique_ptr<Worker>> &workers, Worker::Request request)
{
  //std::cout << "Request " << request << std::endl;
  for (auto &worker : workers)
    worker->IssueRequest(request);
}


void WaitForWorkers(std::vector<std::unique_ptr<Worker>> &workers)
{
  //std::cout << "Waiting ..." << std::endl;
  for (auto &worker : workers)
  {
    while (worker->GetState() != Worker::THREAD_WAITING)
    {
      //std::cout << "worker " << w << " state = " << w->GetState() << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  //std::cout << " Threads are waiting now." << std::endl;
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
  bm.SetColor(0, 0, 128);
  bm.DrawRect(0, 0, xres-1, yres-1);
  ImageDisplay display;
  display.show(bm);
  
  auto start_time = std::chrono::steady_clock::now();
  
  SpectralImageBuffer buffer{xres, yres};
  
  const int num_threads = NUM_THREADS;

  if (num_threads > 0)
  { 
    std::atomic_int shared_pixel_index(std::numeric_limits<int>::max());
    
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::thread> threads;
    
    for (int i=0; i<num_threads; ++i)
      workers.push_back(std::make_unique<Worker>(shared_pixel_index, buffer, scene));
    for (int i=0; i<num_threads; ++i)
      threads.push_back(std::thread{Worker::Run, workers[i].get()});
      
    std::cout << std::endl;
    std::cout << "Rendering ..." << std::endl;
    
    while (display.is_open())
    {
      shared_pixel_index.store(0);
      IssueRequest(workers, Worker::REQUEST_GO);
      
      int image_conversion_y_start = 0;  
      
      while (shared_pixel_index.load() < num_pixels && display.is_open())
      {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        IssueRequest(workers, Worker::REQUEST_HALT);
        WaitForWorkers(workers);
        
        int pixel_index = shared_pixel_index.load();
        int y = pixel_index/xres;
        buffer.ToImage(bm, image_conversion_y_start, std::min(yres, y+1));

        IssueRequest(workers, Worker::REQUEST_GO);

        image_conversion_y_start = y;      
        display.show(bm);
      }

      IssueRequest(workers, Worker::REQUEST_HALT);
      WaitForWorkers(workers);
      
      buffer.ToImage(bm, image_conversion_y_start, yres);
      display.show(bm);
      bm.write("raytrace.tga");
    }
    
    for (auto &worker : workers)
      worker->IssueRequest(Worker::REQUEST_TERMINATE);
    for (auto &thread : threads)
      thread.join();
  }
  else
  {
    Raytracing algo(scene);
    for (int y = 0; y < yres; ++y)
    {
      for (int x = 0; x < xres; ++x)
      {
        //if (x != 47 || y != (128-75))
        //  continue;
        int pixel_index = scene.GetCamera().PixelToUnit({x, y});
        Spectral smpl = algo.MakePrettyPixel(pixel_index);
        buffer.Insert(pixel_index, smpl);
      }
      buffer.ToImage(bm, y, y+1);
      display.show(bm);
    }
    bm.write("raytrace.tga");
  }
  
  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Rendering time: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

  return 0;
}
