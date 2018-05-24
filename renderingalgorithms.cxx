#include "renderingalgorithms.hxx"
#include "rendering_randomwalk_impl.hxx"
#include "rendering_pathtracing_impl.hxx"
#include "image.hxx"

#include <boost/filesystem.hpp>


double UpperBoundToBoundingBoxDiameter(const Scene &scene)
{
  Box bb = scene.GetBoundingBox();
  double diameter = Length(bb.max - bb.min);
  return diameter;
}


bool IterateIntersectionsBetween::Next(RaySegment &seg, RaySurfaceIntersection &intersection)
{
  tfar = this->seg.length;
  using Real = float;
  auto factor = Real(1)+std::numeric_limits<Real>::epsilon(); // TODO: correct this mess eventually if using only float (?)
  auto offset_tnear = factor*tnear;
  bool hit = scene.FirstIntersectionEmbree(this->seg.ray, offset_tnear, tfar, intersection);
  seg.ray = this->seg.ray;
  seg.ray.org += tnear*this->seg.ray.dir;
  seg.length = tfar - tnear;
  tnear = tfar;
  return hit;
}


MediumTracker::MediumTracker(const Scene& _scene)
  : current{nullptr},
    media{}, // Note: Ctor zero-initializes the media array.
    scene{_scene}
{
}


void MediumTracker::initializePosition(const Double3& pos)
{
  std::fill(media.begin(), media.end(), nullptr);
  current = &scene.GetEmptySpaceMedium();
  {
    const Box bb = scene.GetBoundingBox();
    // The InBox check is important. Otherwise I would not know how long to make the ray.
    if (bb.InBox(pos))
    {
      double distance_to_go = 2. * (bb.max-bb.min).maxCoeff(); // Times to due to the diagonal.
      Double3 start = pos;
      start[0] += distance_to_go;
      IterateIntersectionsBetween iter{
        {{start, {-1., 0., 0.}}, distance_to_go}, scene};
      RaySegment seg;
      RaySurfaceIntersection intersection;
      while (iter.Next(seg, intersection))
      {
        goingThroughSurface(seg.ray.dir, intersection);
      }
    }
  }
}



namespace RandomWalk
{

  
void Bdpt::NotifyPassesFinished(int pass_count)
{
  namespace fs = boost::filesystem;
  
  for (auto it = debug_buffers.begin(); it != debug_buffers.end(); ++it)
  {
    int eye_idx = it->first.first;
    int light_idx = it->first.second;
    Spectral3ImageBuffer &buffer = it->second;
    buffer.AddSampleCount(pass_count);
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
    buffer.AddSampleCount(pass_count);
    Image bm(buffer.xres, buffer.yres);
    buffer.ToImage(bm, 0, buffer.yres);
    std::string filename = strformat("bdpt-e%-l%_mis", eye_idx+1, light_idx+1)+".jpg";
    auto filepath = fs::temp_directory_path() / fs::unique_path(filename);
    bm.write(filepath.string());
  }
}
  
}



class NormalVisualizer : public IRenderingAlgo
{
  ToyVector<IRenderingAlgo::SensorResponse> rgb_responses;
  const Scene &scene;
  Sampler sampler;
public:
  NormalVisualizer(const Scene &_scene, const AlgorithmParameters &_params)
    : scene{_scene}
  {
  }
  
  RGB MakePrettyPixel(int pixel_index) override
  {
    auto context = PathContext{Color::LambdaIdxClosestToRGBPrimaries()};
    const auto &camera = scene.GetCamera();
    auto smpl_pos = camera.TakePositionSample(pixel_index, sampler, context);
    auto smpl_dir = camera.TakeDirectionSampleFrom(pixel_index, smpl_pos.coordinates, sampler, context);
    smpl_dir.pdf_or_pmf *= smpl_pos.pdf_or_pmf;
    smpl_dir.value *= smpl_pos.value;
    
    RaySegment seg{{smpl_pos.coordinates, smpl_dir.coordinates}, LargeNumber};
    RaySurfaceIntersection intersection;
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
  
  ToyVector<IRenderingAlgo::SensorResponse>& GetSensorResponses() override
  {
    return rgb_responses;
  }
};



std::unique_ptr<IRenderingAlgo> RenderAlgorithmFactory(const Scene &_scene, const RenderingParameters &_params)
{
  if (_params.algo_name == "bdpt")
    return std::make_unique<Bdpt>(_scene, _params);
  else if (_params.algo_name == "normalvis")
    return std::make_unique<NormalVisualizer>(_scene, _params);
  else
    return std::make_unique<PathTracing>(_scene, _params);
}
