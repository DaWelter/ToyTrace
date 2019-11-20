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

std::unique_ptr<RenderingAlgo> RenderAlgorithmFactory(const Scene &_scene, const RenderingParameters &_params)
{

  if (_params.algo_name == "bdpt")
    return AllocatBptPathTracer(_scene, _params);
  else if (_params.algo_name == "normalvis")
    return AllocateNormalVisualizer(_scene, _params);
  else if (_params.algo_name == "photonmap")
    return AllocatePhotonmappingRenderingAlgo(_scene, _params);
  else
    return AllocatForwardPathTracer(_scene, _params);
}
