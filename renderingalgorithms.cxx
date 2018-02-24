#include "renderingalgorithms.hxx"
#include "renderingalgorithms_pt.hxx"
#include "image.hxx"

#include <boost/filesystem.hpp>


double UpperBoundToBoundingBoxDiameter(const Scene &scene)
{
  Box bb = scene.GetBoundingBox();
  double diameter = Length(bb.max - bb.min);
  return diameter;
}



PathLogger::PathLogger()
  : file{"paths.log"},// {"paths2.log"}},
    max_num_paths{100},
    num_paths_written{0},
    total_path_index{0}
{
  file.precision(std::numeric_limits<double>::digits10);
}


void PathLogger::PreventLogFromGrowingTooMuch()
{
  if (num_paths_written > max_num_paths)
  {
    //files[0].swap(files[1]);
    file.seekp(std::ios_base::beg);
    num_paths_written = 0;
  }
}


void PathLogger::AddSegment(const Double3 &x1, const Double3 &x2, const Spectral3 &beta_at_end_before_scatter, SegmentType type)
{
  const auto &b = beta_at_end_before_scatter;
  file << static_cast<char>(type) << ", "
       << x1[0] << ", " << x1[1] << ", " << x1[2] << ", "
       << x2[0] << ", " << x2[1] << ", " << x2[2] << ", "
       <<  b[0] << ", " <<  b[1] << ", " <<  b[2] << "\n";
  file.flush();
}

void PathLogger::AddScatterEvent(const Double3 &pos, const Double3 &out_dir, const Spectral3 &beta_after, ScatterType type)
{
  const auto &b = beta_after;
  file << static_cast<char>(type) << ", "
       <<     pos[0] << ", " <<     pos[1] << ", " <<     pos[2] << ", "
       << out_dir[0] << ", " << out_dir[1] << ", " << out_dir[2] << ", "
       <<       b[0] << ", " <<       b[1] << ", " <<       b[2] << "\n";
  file.flush();
}


void PathLogger::NewTrace(const Spectral3 &beta_init)
{
  ++total_path_index;
  ++num_paths_written;
  PreventLogFromGrowingTooMuch();
  const auto &b = beta_init;
  file << "n, " << total_path_index << ", " << b[0] << ", " <<  b[1] << ", " <<  b[2] << "\n";
  file.flush();
}




MediumTracker::MediumTracker(const Scene& _scene)
  : current{nullptr},
    media{}, // Note: Ctor zero-initializes the media array.
    scene{_scene}
{
}


void MediumTracker::initializePosition(const Double3& pos, IntersectionCalculator &intersector)
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
      RaySegment seg{{start, {-1., 0., 0.}}, distance_to_go};
      intersector.All(seg.ray, seg.length);
      for (const auto &hit : intersector.Hits())
      {
        RaySurfaceIntersection intersection(hit, seg);
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
}
  
}