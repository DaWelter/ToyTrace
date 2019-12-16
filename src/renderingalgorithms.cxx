#include "renderingalgorithms_simplebase.hxx"


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
std::unique_ptr<RenderingAlgo> AllocateNormalVisualizer(const Scene &scene, const RenderingParameters &params);
std::unique_ptr<RenderingAlgo> AllocatBptPathTracer(const Scene &scene, const RenderingParameters &params);
std::unique_ptr<RenderingAlgo> AllocatForwardPathTracer(const Scene &scene, const RenderingParameters &params);
std::unique_ptr<RenderingAlgo> AllocatePathtracing2RenderingAlgo(const Scene &scene, const RenderingParameters &params);

std::unique_ptr<RenderingAlgo> RenderAlgorithmFactory(const Scene &scene, const RenderingParameters &params)
{

  if (params.algo_name == "bdpt")
    return AllocatBptPathTracer(scene, params);
  else if (params.algo_name == "normalvis")
    return AllocateNormalVisualizer(scene, params);
  else if (params.algo_name == "photonmap")
    return AllocatePhotonmappingRenderingAlgo(scene, params);
  else if (params.algo_name == "pt2")
    return AllocatePathtracing2RenderingAlgo(scene, params);
  else
    return AllocatForwardPathTracer(scene, params);
}
