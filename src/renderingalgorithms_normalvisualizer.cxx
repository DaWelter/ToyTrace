#include "renderingalgorithms_simplebase.hxx"

class NormalVisualizerWorker : public SimplePixelByPixelRenderingDetails::Worker
{
  ToyVector<SensorResponse> rgb_responses;
  const Scene &scene;
  Sampler sampler;
public:
  NormalVisualizerWorker(const Scene &_scene, const RenderingParameters &)
    : scene{ _scene }
  {
  }

  RGB RenderPixel(int pixel_index) override
  {
    auto context = PathContext{ SelectRgbPrimaryWavelengths() };
    const auto &camera = scene.GetCamera();
    auto smpl_pos = camera.TakePositionSample(pixel_index, sampler, context);
    auto smpl_dir = camera.TakeDirectionSampleFrom(pixel_index, smpl_pos.coordinates, sampler, context);
    smpl_dir.pdf_or_pmf *= smpl_pos.pdf_or_pmf;
    smpl_dir.value *= smpl_pos.value;

    RaySegment seg{ {smpl_pos.coordinates, smpl_dir.coordinates}, LargeNumber };
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


class NormalVisualizer : public SimplePixelByPixelRenderingDetails::SimplePixelByPixelRenderingAlgo
{
public:
  NormalVisualizer(const Scene &scene_, const RenderingParameters &render_params_)
    : SimplePixelByPixelRenderingDetails::SimplePixelByPixelRenderingAlgo{ render_params_,scene_ }
  {}
protected:
  std::unique_ptr<SimplePixelByPixelRenderingDetails::Worker> AllocateWorker(int i) override
  {
    return std::make_unique<NormalVisualizerWorker>(scene, render_params);
  }
};


std::unique_ptr<RenderingAlgo> AllocateNormalVisualizer(const Scene &scene, const RenderingParameters &params)
{
  return std::make_unique<NormalVisualizer>(scene, params);
}