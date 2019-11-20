#include "renderingalgorithms_pathtracing.hxx"
#include "rendering_randomwalk_impl.hxx"
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
    std::string filename = strformat("bdpt-e%s-l%s", eye_idx + 1, light_idx + 1) + ".jpg";
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
    std::string filename = strformat("bdpt-e%s-l%s_mis", eye_idx + 1, light_idx + 1) + ".jpg";
    auto filepath = fs::temp_directory_path() / fs::unique_path(filename);
    bm.write(filepath.string());
  }
}

}


std::unique_ptr<RenderingAlgo> AllocatBptPathTracer(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<RandomWalk::BdptAlgo>(scene, params);
}





class ForwardPathTracer : public SimplePixelByPixelRenderingDetails::SimplePixelByPixelRenderingAlgo
{
public:
  ForwardPathTracer(const Scene &scene_, const RenderingParameters &render_params_)
    : SimplePixelByPixelRenderingDetails::SimplePixelByPixelRenderingAlgo{ render_params_,scene_ }
  {}
protected:
  std::unique_ptr<SimplePixelByPixelRenderingDetails::Worker> AllocateWorker(int i) override
  {
    return std::make_unique<RandomWalk::PathTracingWorker>(scene, render_params);
  }
};


std::unique_ptr<RenderingAlgo> AllocatForwardPathTracer(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<ForwardPathTracer>(scene, params);
}