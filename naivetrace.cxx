#include "image.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
#include <chrono>
#include <atomic>
#include <thread>
#include <memory>
#include <boost/program_options.hpp>


static constexpr int SAMPLES_PER_PIXEL = 16;

class Worker
{
  const Scene &scene;
  SpectralImageBuffer &buffer;
  PathTracing algo;
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
  
  Worker(std::atomic_int &_shared_pixel_index, SpectralImageBuffer &_buffer, const Scene &_scene, RenderingParameters &render_params) :
    shared_request(REQUEST_NONE),
    shared_state(THREAD_WAITING),
    shared_pixel_index(_shared_pixel_index),
    buffer(_buffer),
    scene(_scene),
    algo(_scene, render_params)
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


void HandleCommandLineArguments(int argc, char* argv[], std::string &input_file, RenderingParameters &render_params);


int main(int argc, char *argv[])
{
  RenderingParameters render_params;
  std::string input_file;
  
  HandleCommandLineArguments(argc, argv, input_file, render_params);
  
  Scene scene;
  
  std::cout << "parsing input file " << input_file << std::endl;
  scene.ParseNFF(input_file.c_str(), &render_params);
  
  std::cout << "building acceleration structure " << std::endl;
  scene.BuildAccelStructure();
  scene.PrintInfo();
   
  int num_pixels = render_params.width*render_params.height;
  
  Image bm(render_params.width, render_params.height);
  bm.SetColor(0, 0, 128);
  bm.DrawRect(0, 0, render_params.width-1, render_params.height-1);
  ImageDisplay display;
  display.show(bm);
  
  auto start_time = std::chrono::steady_clock::now();
  
  SpectralImageBuffer buffer{render_params.width, render_params.height};

  if (render_params.num_threads > 0)
  { 
    std::atomic_int shared_pixel_index(std::numeric_limits<int>::max());
    
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::thread> threads;
    
    for (int i=0; i<render_params.num_threads; ++i)
      workers.push_back(std::make_unique<Worker>(shared_pixel_index, buffer, scene, render_params));
    for (int i=0; i<render_params.num_threads; ++i)
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
        int y = pixel_index/render_params.width;
        buffer.ToImage(bm, image_conversion_y_start, std::min(render_params.height, y+1));

        IssueRequest(workers, Worker::REQUEST_GO);

        image_conversion_y_start = y;      
        display.show(bm);
      }

      IssueRequest(workers, Worker::REQUEST_HALT);
      WaitForWorkers(workers);
      
      buffer.ToImage(bm, image_conversion_y_start, render_params.height);
      display.show(bm);
      bm.write("raytrace.jpg");
    }
    
    for (auto &worker : workers)
      worker->IssueRequest(Worker::REQUEST_TERMINATE);
    for (auto &thread : threads)
      thread.join();
  }
  else
  {
    //Raytracing algo(scene, render_params);
    NormalVisualizer algo(scene);
    if (render_params.pixel_x<0 && render_params.pixel_y<0)
    {
      for (int y = 0; y < render_params.height; ++y)
      {
        for (int x = 0; x < render_params.width; ++x)
        {
          int pixel_index = scene.GetCamera().PixelToUnit({x, y});
          Spectral smpl = algo.MakePrettyPixel(pixel_index);
          buffer.Insert(pixel_index, smpl);
        }
        buffer.ToImage(bm, y, y+1);
        display.show(bm);
      }
    }
    else
    {
      int pixel_index = scene.GetCamera().PixelToUnit(
        {render_params.pixel_x, render_params.pixel_y});
      Spectral smpl = algo.MakePrettyPixel(pixel_index);
      buffer.Insert(pixel_index, smpl);
      buffer.ToImage(bm, render_params.pixel_y, render_params.pixel_y+1);
      display.show(bm);
    }
    bm.write("raytrace.png");
  }
  
  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Rendering time: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

  return 0;
}



void HandleCommandLineArguments(int argc, char* argv[], std::string &input_file, RenderingParameters &render_params)
{
  namespace po = boost::program_options;
  try
  {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("nt", po::value<int>(), "Number of Threads")
      ("px", po::value<int>(), "X Pixel")
      ("py", po::value<int>(), "Y Pixel")
      ("w", po::value<int>(), "Width")
      ("h", po::value<int>(), "Height")
      ("rd", po::value<int>(), "Max ray depth")
      ("input-file", po::value<std::string>(), "Input file");
    po::positional_options_description pos_desc;
    pos_desc.add("input-file", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                options(desc).
                positional(pos_desc).run(), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      exit(0);
    }
    
    int n = 4;
    if (vm.count("nt"))
    {
      n = vm["nt"].as<int>();
      if (n < 0)
        throw po::error("Numer of threads must be non-negative.");
    }
    render_params.num_threads = n;
    
    bool single_pixel_render = vm.count("px") && vm.count("py");
    int px = -1, py = -1;
    if (single_pixel_render)
    {
      px = vm["px"].as<int>();
      py = vm["py"].as<int>();
    }
    
    int w = -1;
    if (vm.count("w"))
    {
      w = vm["w"].as<int>();
      if (w <= 0)
        throw po::error("Width must be positive");
    }
    
    int h = -1;
    if (vm.count("h"))
    {
      h = vm["h"].as<int>();
      if (h <= 0)
        throw po::error("Height must be positive");
    }
    
    render_params.width = w;
    render_params.height = h;
    
    int rd = 25;
    if (vm.count("rd"))
    {
      rd = vm["rd"].as<int>();
      if (rd <= 1)
        throw po::error("Max ray depth must be greater one.");
    }
    render_params.max_ray_depth = rd;
    
    if (single_pixel_render)
    {
      if (px<0 || px>w)
        throw po::error("Pixel X is out of image bounds.");
      if (py<0 || py>h)
        throw po::error("Pixel Y is out of image bounds.");
    }
    render_params.pixel_x = px;
    render_params.pixel_y = py;
    
    if (single_pixel_render)
      render_params.num_threads = 0;
    
    if (vm.count("input-file"))
      input_file = vm["input-file"].as<std::string>();
    else
      throw po::error("Input file is required.");
  }
  catch(po::error &ex)
  {
    std::cerr << ex.what() << std::endl;
  }
}