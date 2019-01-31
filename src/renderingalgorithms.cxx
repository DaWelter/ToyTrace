#include "renderingalgorithms_simplebase.hxx"
#include "rendering_randomwalk_impl.hxx"
#include "renderingalgorithms_pathtracing.hxx"
#include "image.hxx"

#include <boost/filesystem.hpp>


namespace RandomWalk
{


std::unique_ptr<SimplePixelByPixelRenderingDetails::Worker> BdptAlgo::AllocateWorker(int i)
{
  return std::make_unique<BdptWorker>(this->scene, this->render_params, this);
}


void BdptAlgo::PassCompleted()
{
  namespace fs = boost::filesystem;
  
  for (auto it = debug_buffers.begin(); it != debug_buffers.end(); ++it)
  {
    int eye_idx = it->first.first;
    int light_idx = it->first.second;
    Spectral3ImageBuffer &buffer = it->second;
    buffer.AddSampleCount(GetSamplesPerPixel());
    Image bm(buffer.xres, buffer.yres);
    buffer.ToImage(bm, 0, buffer.yres);
    std::string filename = strformat("bdpt-e%-l%", eye_idx+1, light_idx+1)+".jpg";
    auto filepath = fs::temp_directory_path() / fs::unique_path(filename);
    bm.write(filepath.string());
  }
  
  for (auto it = debug_buffers_mis.begin(); it != debug_buffers_mis.end(); ++it)
  {
    int eye_idx = it->first.first;
    int light_idx = it->first.second;
    Spectral3ImageBuffer &buffer = it->second;
    buffer.AddSampleCount(GetSamplesPerPixel());
    Image bm(buffer.xres, buffer.yres);
    buffer.ToImage(bm, 0, buffer.yres);
    std::string filename = strformat("bdpt-e%-l%_mis", eye_idx+1, light_idx+1)+".jpg";
    auto filepath = fs::temp_directory_path() / fs::unique_path(filename);
    bm.write(filepath.string());
  }
}

}



class NormalVisualizerWorker : public SimplePixelByPixelRenderingDetails::Worker
{
  ToyVector<SensorResponse> rgb_responses;
  const Scene &scene;
  Sampler sampler;
public:
  NormalVisualizerWorker(const Scene &_scene, const AlgorithmParameters &)
    : scene{_scene}
  {
  }
  
  RGB RenderPixel(int pixel_index) override
  {
    auto context = PathContext{Color::LambdaIdxClosestToRGBPrimaries()};
    const auto &camera = scene.GetCamera();
    auto smpl_pos = camera.TakePositionSample(pixel_index, sampler, context);
    auto smpl_dir = camera.TakeDirectionSampleFrom(pixel_index, smpl_pos.coordinates, sampler, context);
    smpl_dir.pdf_or_pmf *= smpl_pos.pdf_or_pmf;
    smpl_dir.value *= smpl_pos.value;
    
    RaySegment seg{{smpl_pos.coordinates, smpl_dir.coordinates}, LargeNumber};
    SurfaceInteraction intersection;
    bool is_hit = scene.FirstIntersectionEmbree(seg.ray, 0., seg.length, intersection);
    if (is_hit)
    {
      RGB col = (intersection.normal.array() * 0.5 + 0.5).cast<Color::RGBScalar>();
      return col;
    }
    else
    {
      return RGB::Zero();
    }
  };
  
  ToyVector<SensorResponse>& GetSensorResponses() override
  {
    return rgb_responses;
  }
};



template<typename WorkerType>
class SimpleAlgo : public SimplePixelByPixelRenderingDetails::SimplePixelByPixelRenderingAlgo
{
public:
  SimpleAlgo(const Scene &scene_,const RenderingParameters &render_params_)
    : SimplePixelByPixelRenderingAlgo{render_params_,scene_}
  {}
protected:
  std::unique_ptr<SimplePixelByPixelRenderingDetails::Worker> AllocateWorker(int i) override
  {
    return std::make_unique<WorkerType>(scene,render_params);
  }
};


// Forward declaration.
std::unique_ptr<RenderingAlgo> AllocatePhotonmappingRenderingAlgo(const Scene &scene, const RenderingParameters &params);

std::unique_ptr<RenderingAlgo> RenderAlgorithmFactory(const Scene &_scene, const RenderingParameters &_params)
{
  using namespace RandomWalk;
  if (_params.algo_name == "bdpt")
    return std::make_unique<BdptAlgo>(_scene, _params);
  else if (_params.algo_name == "normalvis")
    return std::make_unique<SimpleAlgo<NormalVisualizerWorker>>(_scene, _params);
  else if (_params.algo_name == "photonmap")
    return AllocatePhotonmappingRenderingAlgo(_scene, _params);
  else
    return std::make_unique<SimpleAlgo<PathTracingWorker>>(_scene, _params);
}
