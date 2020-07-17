#include "image.hxx"
#include "scene.hxx"
#include "renderbuffer.hxx"
#include "renderingalgorithms_interface.hxx"
#include "pathlogger.hxx"

#include <chrono>
#include <thread>
#include <memory>
#include <tuple>
#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>

#include <tbb/tbb_thread.h>
#include <tbb/concurrent_queue.h>
#include <tbb/task_scheduler_init.h>

namespace fs = boost::filesystem;

// Strategy pattern for display algorithm. (No display, or CIMG based display)
class MaybeDisplay
{
public:
  virtual ~MaybeDisplay() = default;
  virtual void Show(const Image &im) = 0;
  virtual bool IsOkToKeepGoing() const = 0;
  virtual bool WantsPeriodicDisplay() const = 0;
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
  
  bool WantsPeriodicDisplay() const override
  {
      return true;
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
  
  bool WantsPeriodicDisplay() const override
  {
      return false;
  }
};


template<class T>
inline T pop(tbb::concurrent_bounded_queue<T> &q)
{
  T tmp;
  q.pop(tmp);
  return tmp;
}


std::unique_ptr<MaybeDisplay> MakeDisplay(bool will_open_a_window);


void HandleCommandLineArguments(int argc, char* argv[], fs::path &input_file, fs::path &output_file, RenderingParameters &render_params, std::unique_ptr<MaybeDisplay> &display);


int main(int argc, char *argv[])
{
  RenderingParameters render_params;
  fs::path input_file;
  fs::path output_file;
  std::unique_ptr<MaybeDisplay> display;
  
  HandleCommandLineArguments(argc, argv, input_file, output_file, render_params, display);
  
  tbb::task_scheduler_init init(std::max(1, render_params.num_threads));
  
  Scene scene;
  
  std::cout << "Parsing input file " << input_file << std::endl;
  std::cout << "Output to " << output_file << std::endl;

  try 
  {
    if (input_file.string() != "-")
    {
      scene.ParseSceneFile(input_file.c_str(), &render_params);
    }
    else
    {
      scene.ParseNFF(std::cin, &render_params);
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error parsing the scene file: " << e.what() << "\n";
    std::cerr << "Exiting ...\n";
    std::exit(-1);
  }
  
  if (!scene.HasCamera())
  {
    std::cout << "There is no camera. Aborting." << std::endl;
    return -1;
  }
  
  if (!scene.HasLights())
  {
    std::cout << "There are no lights. Aborting." << std::endl;
    return -1;
  }

  std::cout << "building acceleration structure " << std::endl;
  scene.BuildAccelStructure();
  scene.PrintInfo();
  
  {
    Image bm(render_params.width, render_params.height);
    bm.SetColor(0, 0, 128);
    bm.DrawRect(0, 0, render_params.width-1, render_params.height-1);
    display->Show(bm);
  }

  // Use a queue and an extra worker thread to perform image display and io in a non-blocking way.
  using ImageWorkItem = std::tuple<Image*, bool>; // Would like to use unique_ptr, but queue implementation requires items to be copyable!
  tbb::concurrent_bounded_queue<ImageWorkItem> image_queue;
  image_queue.set_capacity(2);
  
  auto PopAndProcessQueueItem = [&]{
      auto [im, is_complete_pass] = pop(image_queue); // Blocks
      if (!im)
        return;
      if (display->IsOkToKeepGoing())
        display->Show(*im);
      if (is_complete_pass) // Partially rendered images are probably useless
        im->write(output_file.string());
      delete(im);
  };
  
  tbb::atomic<bool> stop_flag = false;
  tbb::tbb_thread image_handling_thread([&] {
    while (stop_flag.load() == false)
    {
      PopAndProcessQueueItem();
    }
  });
  
#ifdef HAVE_JSON
  Pathlogger::Init("/tmp/paths.json");
  IncompletePaths::Init();
  scene.WriteObj("/tmp/scene.obj");
#endif
  

  auto algo = RenderAlgorithmFactory(scene, render_params);
  algo->InitializeScene(scene);
  
  auto time_of_last_image_request = std::chrono::steady_clock::now();
  
  algo->SetInterruptCallback([&](bool is_complete_pass){
    if ((std::chrono::steady_clock::now() - time_of_last_image_request > std::chrono::milliseconds(1000)) || is_complete_pass)
    {
      auto im = algo->GenerateImage();
      image_queue.push(ImageWorkItem{im.release(), is_complete_pass}); // Would use emplace but my TBB believes that I have no variadic template argument support.
      time_of_last_image_request = std::chrono::steady_clock::now();
    }
  });

  tbb::tbb_thread watchdog_and_image_updater([&] {
    while (display->IsOkToKeepGoing() && stop_flag.load() == false)
    {    
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (display->WantsPeriodicDisplay())
      {
          algo->RequestInterrupt();
      }
    }
    algo->RequestFullStop();
  });
  
  //////////////////////////////////
  // Rendering is done here

  auto start_time = std::chrono::steady_clock::now();

  std::cout << std::endl;
  std::cout << "Rendering ..." << std::endl;
  
  algo->Run();
  
  stop_flag.store(true);
  image_queue.push({ nullptr, true});

  watchdog_and_image_updater.join();
  image_handling_thread.join();
  
  // Ensure that the last image is saved
  while (image_queue.size())
    PopAndProcessQueueItem();

  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Rendering time: " << std::chrono::duration<double>(end_time - start_time).count() << " sec." << std::endl;
  //////////////////////////////////
  
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
      ("spp", po::value<int>(), "Max samples per pixel")
      ("guide-em-every", po::value<int>(), "Guiding: Expectancy maximization every x samples.")
      ("guide-prior-strength", po::value<double>(), "Guiding: Roughly the number of samples were prior becomes insignificant.")
      ("guide-subdiv-factor", po::value<int>(), "Guiding: Less makes the tree more refined. Value ranges around 100 to 10000.")
      ("guide-max-spp", po::value<int>(), "Guiding: The maximal number of samples per pixel. Sample count will double until this value is reached.")
      ("algo", po::value<std::string>()->default_value("pt"), "Rendering algorithm: pt or bdpt")
      ("phm-radius", po::value<double>(), "Initial photon radius for photon mapping")
      ("pt-sample-mode", po::value<std::string>(), "Light sampling: 'bsdf' - bsdf importance sampling, 'lights' - sample lights aka. next event estimation, 'both' - both combined by MIS.")
      ("no-display", po::bool_switch()->default_value(false), "Don't open a display window")
      ("include,I", po::value<std::vector<std::string>>(), "Include paths")
      ("output-file,o", po::value<fs::path>(), "Output file")
      ("linear-out", po::bool_switch()->default_value(false), "Output image in linear color space. Like sRGB but without doing the gamma correction.")
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
    
    if (vm.count("guide-em-every"))
    {
      render_params.guiding_em_every = vm["guide-em-every"].as<int>();
      if (render_params.guiding_em_every <= 0)
        throw po::error("guide-em-every must be positive");
    }
    if (vm.count("guide-prior-strength"))
    {
      render_params.guiding_prior_strength = vm["guide-prior-strength"].as<double>();
      if (render_params.guiding_prior_strength <= 0)
        throw po::error("guide-prior-strength must be positive");
    }
    if (vm.count("guide-subdiv-factor"))
    {
      render_params.guiding_tree_subdivision_factor = vm["guide-subdiv-factor"].as<int>();
      if (render_params.guiding_tree_subdivision_factor <= 0)
        throw po::error("guide-subdiv-factor must be positive");
    }
    if (vm.count("guide-max-spp"))
    {
      render_params.guiding_max_spp = vm["guide-max-spp"].as<int>();
      if (render_params.guiding_max_spp <= 0)
        throw po::error("guide-max-spp must be positive");
    }

    if (vm["linear-out"].as<bool>())
    {
      render_params.linear_output = true;
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
        render_params.algo_name != "pt2" &&
        render_params.algo_name != "bdpt" &&
        render_params.algo_name != "normalvis" &&
        render_params.algo_name != "ptg" &&
        render_params.algo_name != "photonmap")
      throw po::error("Algorithm must be pt or bdpt or normalvis");
    
    int max_spp = -1;
    if (vm.count("spp"))
    {
      max_spp = vm["spp"].as<int>();
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
        output_file += ".png";
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
    
    if (vm.count("phm-radius"))
    {
      auto r = vm["phm-radius"].as<double>();
      if (r <= 0.)
        throw po::error("Bad argument for phm-radius. Radius must be positive");
      render_params.initial_photon_radius = r;
    }
  }
  catch(po::error &ex)
  {
    std::cerr << ex.what() << std::endl;
    exit(0);
  }
}
