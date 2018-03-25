#include "image.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
#include "renderingalgorithms_pt.hxx"

#include <chrono>
#include <atomic>
#include <thread>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>

namespace fs = boost::filesystem;


std::unique_ptr<IRenderingAlgo> RenderAlgorithmFactory(const Scene &_scene, const RenderingParameters &_params)
{
  if (_params.algo_name == "bdpt")
    return std::make_unique<Bdpt>(_scene, _params);
  else
    return std::make_unique<PathTracing>(_scene, _params);
}


class Worker
{
  const Scene &scene;
  Spectral3ImageBuffer &buffer;
  std::unique_ptr<IRenderingAlgo> algo;
  //NormalVisualizer algo;
  const int pixel_stride = 64 / sizeof(Spectral3); // Because false sharing.
  int num_pixels;
  std::atomic_int &shared_pixel_index;
  std::atomic_int shared_request, shared_state;
public:
  Worker(const Worker &) = delete;
  Worker(Worker &&) = delete;
  int samples_per_pixel; // Only set when thread is stopped.
  
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
  
  Worker(std::atomic_int &_shared_pixel_index, Spectral3ImageBuffer &_buffer, const Scene &_scene, RenderingParameters &render_params) :
    scene(_scene),
    buffer(_buffer),
    algo(RenderAlgorithmFactory(_scene, render_params)),
    shared_pixel_index(_shared_pixel_index),
    shared_request(REQUEST_NONE),
    shared_state(THREAD_WAITING),
    samples_per_pixel(16)
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
  
  void FillInSplats()
  {
    auto &responses = algo->GetSensorResponses();
    for (const auto &r : responses)
    {
      assert (r);
      buffer.Splat(r.unit_index, r.weight);
    }
    responses.clear();
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
      const int nsmpl = samples_per_pixel;
      for(int i=0;i<nsmpl;i++)
      {
        auto smpl = algo->MakePrettyPixel(pixel_index);
        buffer.Insert(pixel_index, smpl);
      }
    }
  }
};


using WorkerSet = std::vector<std::unique_ptr<Worker>>;

void IssueRequest(WorkerSet &workers, Worker::Request request)
{
  //std::cout << "Request " << request << std::endl;
  for (auto &worker : workers)
    worker->IssueRequest(request);
}


void WaitForWorkers(WorkerSet &workers)
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


void DetermineSamplesPerPixelForNextPass(int &spp, int total_spp, const RenderingParameters &render_params)
{
  if (spp < 256)
  {
    spp *= 2;
  }
  const int max_spp = render_params.max_samples_per_pixel;
  if (max_spp > 0 && 
      total_spp + spp > max_spp)
  {
    spp = max_spp - total_spp;
  }
}


void AssignSamplesPerPixel(WorkerSet &workers, int spp)
{
  for (auto &worker : workers)
  {
    worker->samples_per_pixel = spp;
  }
}



class MaybeDisplay
{
public:
  virtual void Show(const Image &im) = 0;
  virtual bool IsOkToKeepGoing() const = 0;
};

class CIMGDisplay : public MaybeDisplay
{
  ImageDisplay display;
public:
  void Show(const Image &im) override
  {
    display.show(im);
  }
  
  bool IsOkToKeepGoing() const override
  {
    return display.is_open();
  }
};

class NoDisplay : public MaybeDisplay
{
public:
  void Show(const Image &im) override
  {
  }
  
  bool IsOkToKeepGoing() const override
  {
    return true;
  }
};

std::unique_ptr<MaybeDisplay> MakeDisplay(bool will_open_a_window);

void HandleCommandLineArguments(int argc, char* argv[], fs::path &input_file, fs::path &output_file, RenderingParameters &render_params, std::unique_ptr<MaybeDisplay> &display);


int main(int argc, char *argv[])
{
  RenderingParameters render_params;
  fs::path input_file;
  fs::path output_file;
  std::unique_ptr<MaybeDisplay> display;
  
  HandleCommandLineArguments(argc, argv, input_file, output_file, render_params, display);
  
  Scene scene;
  
  std::cout << "Parsing input file " << input_file << std::endl;
  std::cout << "Output to " << output_file << std::endl;
  if (input_file.string() != "-")
  {
    scene.ParseNFF(input_file.c_str(), &render_params);
  }
  else
  {
//     std::string scene_str;
//     constexpr std::size_t SZ = 1024;
//     std::vector<char> data; data.resize(SZ);
//     while (std::cin)
//     {
//       std::cin.read(&data[0], SZ);
//       scene_str.append(&data[0], std::cin.gcount());
//     }
    scene.ParseNFF(std::cin, &render_params);
  }
  
  if (!scene.HasCamera())
  {
    std::cout << "There is no camera. Aborting." << std::endl;
    return -1;
  }
  
  if (scene.GetNumAreaLights() <= 0 &&
      scene.GetNumEnvLights() <= 0 &&
      scene.GetNumLights() <= 0)
  {
    std::cout << "There are no lights. Aborting." << std::endl;
    return -1;
  }

  std::cout << "building acceleration structure " << std::endl;
  scene.BuildAccelStructure();
  scene.PrintInfo();
   
  int num_pixels = render_params.width*render_params.height;
  
  Image bm(render_params.width, render_params.height);
  bm.SetColor(0, 0, 128);
  bm.DrawRect(0, 0, render_params.width-1, render_params.height-1);
  display->Show(bm);
  
  auto start_time = std::chrono::steady_clock::now();
  
  Spectral3ImageBuffer buffer{render_params.width, render_params.height};

  if (render_params.num_threads > 0)
  { 
    std::atomic_int shared_pixel_index(std::numeric_limits<int>::max());
    
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::thread> threads;
    
    for (int i=0; i<render_params.num_threads; ++i)
      workers.push_back(std::make_unique<Worker>(shared_pixel_index, buffer, scene, render_params));
    for (int i=0; i<render_params.num_threads; ++i)
      threads.push_back(std::thread{Worker::Run, workers[i].get()});
    
    int total_samples_per_pixel = 0;
    int samples_per_pixel_per_iteration = 1;
    AssignSamplesPerPixel(workers, samples_per_pixel_per_iteration);
    
    std::cout << std::endl;
    std::cout << "Rendering ..." << std::endl;
        
    while (display->IsOkToKeepGoing() && samples_per_pixel_per_iteration>0)
    {
      buffer.AddSampleCount(samples_per_pixel_per_iteration);
      shared_pixel_index.store(0);
      IssueRequest(workers, Worker::REQUEST_GO);

      auto UpdateImageToCurrentLine = [&]() -> void
      {
        int pixel_index = shared_pixel_index.load();
        int y = std::max(0, pixel_index/render_params.width);
        buffer.ToImage(bm, 0, std::min(render_params.height, y));
      };
      
      while (shared_pixel_index.load() < num_pixels && display->IsOkToKeepGoing())
      {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        IssueRequest(workers, Worker::REQUEST_HALT);
        WaitForWorkers(workers);
        
        for (auto &worker : workers)
          worker->FillInSplats();

        UpdateImageToCurrentLine();
        IssueRequest(workers, Worker::REQUEST_GO);        
        display->Show(bm);
      }

      IssueRequest(workers, Worker::REQUEST_HALT);
      WaitForWorkers(workers);

      for (auto &worker : workers)
        worker->FillInSplats();
      UpdateImageToCurrentLine();
      display->Show(bm);
      bm.write(output_file.string());
      
      total_samples_per_pixel += samples_per_pixel_per_iteration;
      std::cout << "Iteration finished, spp = " << samples_per_pixel_per_iteration << ", total " << total_samples_per_pixel << std::endl;
      DetermineSamplesPerPixelForNextPass(samples_per_pixel_per_iteration, total_samples_per_pixel, render_params);
      AssignSamplesPerPixel(workers, samples_per_pixel_per_iteration);
    }
    
    for (auto &worker : workers)
      worker->IssueRequest(Worker::REQUEST_TERMINATE);
    for (auto &thread : threads)
      thread.join();
  }
  else
  {
    std::unique_ptr<IRenderingAlgo> algo = RenderAlgorithmFactory(scene, render_params);

    int total_samples_per_pixel = 0;
    int samples_per_pixel_per_iteration = 1;

    while (display->IsOkToKeepGoing() && samples_per_pixel_per_iteration>0)
    {
      buffer.AddSampleCount(samples_per_pixel_per_iteration);
      for (int y = 0; y < render_params.height && display->IsOkToKeepGoing(); ++y)
      {
        for (int x = 0; x < render_params.width; ++x)
        {
          int pixel_index = scene.GetCamera().PixelToUnit({x, y});
          for (int s = 0; s < samples_per_pixel_per_iteration; ++s)
          {
            auto smpl = algo->MakePrettyPixel(pixel_index);
            buffer.Insert(pixel_index, smpl);
          }
        }
        for (const auto &r : algo->GetSensorResponses())
          buffer.Splat(r.unit_index, r.weight);
        algo->GetSensorResponses().clear();
        buffer.ToImage(bm, 0, y+1);
        display->Show(bm);
      }
      algo->NotifyPassesFinished(samples_per_pixel_per_iteration);
      total_samples_per_pixel += samples_per_pixel_per_iteration;
      std::cout << "Iteration finished, spp = " << samples_per_pixel_per_iteration << ", total " << total_samples_per_pixel << std::endl;
      DetermineSamplesPerPixelForNextPass(samples_per_pixel_per_iteration, total_samples_per_pixel, render_params);
    }

    bm.write(output_file.string());
  }
  
  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Rendering time: " << std::chrono::duration<double>(end_time - start_time).count() << " sec." << std::endl;

  return 0;
}



std::unique_ptr<MaybeDisplay> MakeDisplay(bool will_open_a_window)
{
  if (will_open_a_window)
    return std::make_unique<CIMGDisplay>();
  else
    return std::make_unique<NoDisplay>();
}


void HandleCommandLineArguments(int argc, char* argv[], fs::path &input_file, fs::path &output_file, RenderingParameters &render_params, std::unique_ptr<MaybeDisplay> &display)
{
  namespace po = boost::program_options;
  try
  {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help", "Help screen")
      ("nt", po::value<int>(), "Number of Threads")
      ("px", po::value<int>(), "X Pixel")
      ("py", po::value<int>(), "Y Pixel")
      ("w,w", po::value<int>(), "Width")
      ("h,h", po::value<int>(), "Height")
      ("rd", po::value<int>(), "Max ray depth")
      ("max-spp", po::value<int>(), "Max samples per pixel")
      ("algo", po::value<std::string>()->default_value("pt"), "Rendering algorithm: pt or bdpt")
      ("pt-sample-mode", po::value<std::string>(), "Light sampling: 'bsdf' - bsdf importance sampling, 'lights' - sample lights aka. next event estimation, 'both' - both combined by MIS.")
      ("no-display", po::bool_switch()->default_value(false), "Don't open a display window")
      ("include,I", po::value<std::vector<std::string>>(), "Include paths")
      ("output-file,o", po::value<fs::path>(), "Output file")
      ("input-file", po::value<fs::path>(), "Input file");
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
      if (rd <= 0)
        throw po::error("Max ray depth must be greater zero.");
    }
    render_params.max_ray_depth = rd;
    
    if (single_pixel_render)
    {
      if (px<0 || px>w)
        throw po::error("Pixel X is out of image bounds.");
      if (py<0 || py>h)
        throw po::error("Pixel Y is out of image bounds.");
      py = h -1 - py; // Because for some reason my representation of images is flipped.
    }
    render_params.pixel_x = px;
    render_params.pixel_y = py;
    
    if (single_pixel_render)
      render_params.num_threads = 0;
    
    render_params.algo_name = vm["algo"].as<std::string>();
    if (render_params.algo_name != "pt" &&
        render_params.algo_name != "bdpt")
      throw po::error("Algorithm must be pt or bdpt");
    
    int max_spp = -1;
    if (vm.count("max-spp"))
    {
      max_spp = vm["max-spp"].as<int>();
      if (max_spp <= 0)
        throw po::error("Max samples per pixel must be greater zero");
    }
    render_params.max_samples_per_pixel = max_spp;
    
    if (vm.count("input-file"))
      input_file = vm["input-file"].as<fs::path>();
    else
      throw po::error("Input file is required.");
    
    if (vm.count("output-file"))
      output_file = vm["output-file"].as<fs::path>();
    else
    {
      if (input_file.string() == "-")
      {
        throw po::error("Output file must be set when input is from stdin.");
      }
      else
      {
        output_file = input_file.parent_path() / input_file.stem();
        output_file += ".jpg";
      }
    }
    
    if (vm.count("pt-sample-mode"))
    {
      std::string mode = vm["pt-sample-mode"].as<std::string>();
      std::vector<std::string> admissible = { "both", "bsdf", "lights" };
      if (std::find(admissible.begin(), admissible.end(), mode) == admissible.end())
        throw po::error("Bad argument for pt-sample-mode");
      render_params.pt_sample_mode = mode;
    }
    
    bool open_display = !vm["no-display"].as<bool>();
    if (!open_display && render_params.max_samples_per_pixel < 0)
      std::cout << "WARNING: Not opening display and no sample count given. Will run until killed." << std::endl;
    display = MakeDisplay(open_display);
    
    if (vm.count("include"))
    {
      auto list_of_includes = vm["include"].as<std::vector<std::string>>();
      for (auto p : list_of_includes)
        render_params.search_paths.emplace_back(p);
    }
  }
  catch(po::error &ex)
  {
    std::cerr << ex.what() << std::endl;
    exit(0);
  }
}