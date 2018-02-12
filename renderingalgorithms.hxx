#pragma once

#include <fstream>

#include "scene.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "renderbuffer.hxx"

namespace ROI = RadianceOrImportance;

// WARNING: THIS IS NOT THREAD SAFE. DON'T WRITE TO THE SAME FILE FROM MULTIPLE THREADS!
class PathLogger
{
  std::ofstream file;
  int max_num_paths;
  int num_paths_written;
  int total_path_index;
  void PreventLogFromGrowingTooMuch();
public:
  enum SegmentType : int {
    EYE_SEGMENT = 'e',
    LIGHT_PATH = 'l',
    EYE_LIGHT_CONNECTION = 'c',
  };
  enum ScatterType : int {
    SCATTER_VOLUME = 'v',
    SCATTER_SURFACE = 's'
  };
  PathLogger();
  void AddSegment(const Double3 &x1, const Double3 &x2, const Spectral3 &beta_at_end_before_scatter, SegmentType type);
  void AddScatterEvent(const Double3 &pos, const Double3 &out_dir, const Spectral3 &beta_after, ScatterType type);
  void NewTrace(const Spectral3 &beta_init);
};


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


#ifndef NDEBUG
#  define PATH_LOGGING(x)
#else
#  define PATH_LOGGING(x)
#endif


/* This thing tracks overlapping media volumes. Since it is a bit complicated
 * to physically correctly handle mixtures of media that would occur
 * in overlapping volumes, I take a simpler approach. This hands over the
 * medium with the highest associated priority. In case multiple volumes with the
 * same medium material overlap it will be as if there was the union of all of
 * those volumes.
 */
class MediumTracker
{
  static constexpr int MAX_INTERSECTING_MEDIA = 4;
  const Medium* current;
  std::array<const Medium*, MAX_INTERSECTING_MEDIA> media;
  const Scene &scene;
  void enterVolume(const Medium *medium);
  void leaveVolume(const Medium *medium);
  const Medium* findMediumOfHighestPriority() const;
  bool remove(const Medium *medium);
  bool insert(const Medium *medium);
public:
  explicit MediumTracker(const Scene &_scene);
  MediumTracker(const MediumTracker &_other) = default;
  void initializePosition(const Double3 &pos, IntersectionCalculator &intersector);
  void initializePosition(const Double3 &pos, IntersectionCalculator &&intersector)
  { // Take control of the temporary that was passed in here. Now it has a name and
    // therefore repeated calling of initializePosition will invoke the one with the
    // lvalue in the parameter list.
    initializePosition(pos, intersector);
  }
  void goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection &intersection);
  const Medium& getCurrentMedium() const;
};


MediumTracker::MediumTracker(const Scene& _scene)
  : current{nullptr},
    media{}, // Note: Ctor zero-initializes the media array.
    scene{_scene}
{
}


const Medium& MediumTracker::getCurrentMedium() const
{
  assert(current);
  return *current;
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


void MediumTracker::goingThroughSurface(const Double3 &dir_of_travel, const RaySurfaceIntersection& intersection)
{
  if (Dot(dir_of_travel, intersection.geometry_normal) < 0)
    enterVolume(intersection.primitive().medium);
  else
    leaveVolume(intersection.primitive().medium);
}


const Medium* MediumTracker::findMediumOfHighestPriority() const
{
  const Medium* medium_max_prio = &scene.GetEmptySpaceMedium();
  for (int i = 0; i < media.size(); ++i)
  {
    medium_max_prio = (media[i] &&  medium_max_prio->priority < media[i]->priority) ?
                        media[i] : medium_max_prio;
  }
  return medium_max_prio;
}


bool MediumTracker::remove(const Medium *medium)
{
  bool is_found = false;
  for (int i = 0; (i < media.size()) && !is_found; ++i)
  {
    is_found = media[i] == medium;
    media[i] = is_found ? nullptr : media[i];
  }
  return is_found;
}


bool MediumTracker::insert(const Medium *medium)
{
  bool is_place_empty = false;
  for (int i = 0; (i < media.size()) && !is_place_empty; ++i)
  {
    is_place_empty = media[i]==nullptr;
    media[i] = is_place_empty ? medium : media[i];
  }
  return is_place_empty;
}


void MediumTracker::enterVolume(const Medium* medium)
{
  // Set one of the array entries that is currently nullptr to the new medium pointer.
  // But only if there is room left. Otherwise the new medium is ignored.
  bool was_inserted = insert(medium);
  current = (was_inserted && medium->priority > current->priority) ? medium : current;
}


void MediumTracker::leaveVolume(const Medium* medium)
{
  // If the medium is in the media stack, remove it.
  // And also make the medium of highest prio the current one.
  remove(medium);
  if (medium == current)
  {
    current = findMediumOfHighestPriority();
  }
}


namespace RadianceOrImportance
{

class TotalEnvironmentalRadianceField : public EnvironmentalRadianceField
{
  inline const EnvironmentalRadianceField& get(int i) const { return scene.GetEnvLight(i); }
  inline int size() const { return scene.GetNumEnvLights(); }
  const Scene& scene;
public:
  
  
  TotalEnvironmentalRadianceField(const Scene& _scene) : scene(_scene) {}
  
  
  DirectionalSample TakeDirectionSample(Sampler &sampler, const LightPathContext &context) const override
  {
    assert(size()>0);
    int idx_sample = sampler.UniformInt(0, size()-1);
    const double selection_probability = 1./size();
    ROI::DirectionalSample smpl = get(idx_sample).TakeDirectionSample(sampler, context);
#if 1
    // This simple version seems to work just as well.
    smpl.pdf_or_pmf *= selection_probability; // Russian roulette style removal of all but one sub-component.
    return smpl;
#else
    if (IsFromPmf(smpl))
    {
      smpl.pdf_or_pmf *= selection_probability;
      return smpl;
    }
    else
    {
      double total_pdf = smpl.pdf_or_pmf;
      for (int i=0; i<size(); ++i)
      {
        if (i == idx_sample) continue;
        smpl.value += get(i).Evaluate(smpl.coordinates, context);
        total_pdf += get(i).EvaluatePdf(smpl.coordinates, context);
      }
      smpl.pdf_or_pmf = Pdf{selection_probability*total_pdf}; // TODO: This cannot be right.
    }
    return smpl;
#endif
  }
  

  Spectral3 Evaluate(const Double3 &emission_dir, const LightPathContext &context) const override
  {
    Spectral3 environmental_radiance{0.};
    for (int i=0; i<size(); ++i)
    {
      const auto &light = get(i);
      environmental_radiance += light.Evaluate(
        emission_dir, context);
    }
    return environmental_radiance;
  }
  
  
  double EvaluatePdf(const Double3 &dir_out, const LightPathContext &context) const override
  {
    const double selection_probability = size()>0 ? 1./size() : 1.;
    double pdf_sum = 0.;
    for (int i=0; i<size(); ++i)
    {
      const auto &light = get(i);
      pdf_sum += light.EvaluatePdf(dir_out, context);
    }
    return pdf_sum * selection_probability;
  }
};

}


class RadianceEstimatorBase
{
protected:
  Sampler sampler;
  IntersectionCalculator intersector;
  const Scene &scene;

  static constexpr int IDX_PROB_ENV = 0;
  static constexpr int IDX_PROB_AREA = 1;
  static constexpr int IDX_PROB_POINT = 2;
  std::array<double, 3> emitter_type_selection_probabilities;
  ROI::TotalEnvironmentalRadianceField envlight;

public:
  std::vector<std::pair<int,RGB>> sensor_responses;
  
public:
  RadianceEstimatorBase(const Scene &_scene)
    : intersector{_scene.MakeIntersectionCalculator()}, scene{_scene}, envlight{_scene}
  {
    const double nl = scene.GetNumLights();
    const double ne = scene.GetNumEnvLights();
    const double na = scene.GetNumAreaLights();
    const double prob_pick_env_light = ne/(nl+ne+na);
    const double prob_pick_area_light = na/(nl+ne+na);
//     std::copy(
//       emitter_type_selection_probabilities.begin(),
//       emitter_type_selection_probabilities.end(),
//               {{ prob_pick_env_light, prob_pick_area_light, 1.-prob_pick_area_light-prob_pick_env_light }};
    emitter_type_selection_probabilities = { prob_pick_env_light, prob_pick_area_light, 1.-prob_pick_area_light-prob_pick_env_light };
  }

  virtual ~RadianceEstimatorBase() {}
  
  const ROI::EnvironmentalRadianceField& GetEnvLight() const
  {
    return envlight;
  }
  
  using LightPick = std::tuple<const ROI::PointEmitter*, double>;
  using AreaLightPick = std::tuple<const Primitive*, const ROI::AreaEmitter*, double>;
  
  LightPick PickLight()
  {
    const int nl = scene.GetNumLights();
    assert(nl > 0);
    int idx = sampler.UniformInt(0, nl-1);
    const ROI::PointEmitter *light = &scene.GetLight(idx);
    double pmf_of_light = 1./nl;
    return LightPick{light, pmf_of_light};
  }
  
  double PmfOfLight(const ROI::PointEmitter &light)
  {
    return 1./scene.GetNumLights();
  }
  
  AreaLightPick PickAreaLight()
  {
    const int na = scene.GetNumAreaLights();
    assert (na > 0);
    const Primitive* prim;
    const ROI::AreaEmitter* emitter;
    std::tie(prim, emitter) = scene.GetAreaLight(sampler.UniformInt(0, na-1));
    double pmf_of_light = 1./na;
    return AreaLightPick{prim, emitter, pmf_of_light}; 
  }

  double PmfOfLight(const ROI::AreaEmitter &light)
  {
    return 1./scene.GetNumAreaLights();
  }
  
  Spectral3 TransmittanceEstimate(RaySegment seg, MediumTracker &medium_tracker, const PathContext &context)
  {
    // TODO: Russian roulette.
    Spectral3 result{1.};
    auto SegmentContribution = [&seg, &medium_tracker, &context, this](double t1, double t2) -> Spectral3
    {
      RaySegment subseg{{seg.ray.PointAt(t1), seg.ray.dir}, t2-t1};
      return medium_tracker.getCurrentMedium().EvaluateTransmission(subseg, sampler, context);
    };

    intersector.All(seg.ray, seg.length);
    double t = 0;
    for (const auto &hit : intersector.Hits())
    {
      RaySurfaceIntersection intersection{hit, seg};
      result *= intersection.shader().EvaluateBSDF(-seg.ray.dir, intersection, seg.ray.dir, context, nullptr);
      if (result.isZero())
        return result;
      result *= SegmentContribution(t, hit.t);
      medium_tracker.goingThroughSurface(seg.ray.dir, intersection);
      t = hit.t;
    }
    result *= SegmentContribution(t, seg.length);
    return result;
  }

  
  struct CollisionData
  {
    CollisionData (const Ray &ray) :
      smpl{},
      segment{ray, LargeNumber},
      hit{}
    {}
    Medium::InteractionSample smpl;
    RaySegment segment;
    HitId hit;
  };
  

  void TrackToNextInteraction(
    CollisionData &collision,
    MediumTracker &medium_tracker,
    const PathContext &context)
  {
    Spectral3 total_weight{1.};
    auto &segment = collision.segment;
    while (true)
    {
      const Medium& medium = medium_tracker.getCurrentMedium();
      
      const auto &hit = collision.hit = intersector.First(segment.ray, segment.length);
      const auto &medium_smpl = collision.smpl = medium.SampleInteractionPoint(segment, sampler, context);
      total_weight *= medium_smpl.weight;
      
      if (medium_smpl.t >= segment.length && hit && hit.primitive->shader->IsPassthrough())
      {
        RaySurfaceIntersection intersection{hit, segment};
        medium_tracker.goingThroughSurface(segment.ray.dir, intersection);
        segment.ray.org = intersection.pos;
        segment.ray.org += AntiSelfIntersectionOffset(intersection, RAY_EPSILON, segment.ray.dir);
        segment.length = LargeNumber;
      }
      else
      {
        collision.smpl.weight = total_weight;
        return;
      }
    }
  }
};



class BaseAlgo : public RadianceEstimatorBase
{
public:
  BaseAlgo(const Scene &scene) : RadianceEstimatorBase{scene} {}
  virtual RGB MakePrettyPixel(int pixel_index) = 0;
};



class NormalVisualizer : public BaseAlgo
{
public:
  NormalVisualizer(const Scene &_scene) : BaseAlgo(_scene) 
  {
  }
  
  RGB MakePrettyPixel(int pixel_index) override
  {
    auto light_context = ROI::LightPathContext(Color::LambdaIdxClosestToRGBPrimaries());
    const auto &camera = scene.GetCamera();
    auto smpl_pos = camera.TakePositionSample(pixel_index, sampler, light_context);
    auto smpl_dir = camera.TakeDirectionSampleFrom(pixel_index, smpl_pos.coordinates, sampler, light_context);
    smpl_dir.pdf_or_pmf *= smpl_pos.pdf_or_pmf;
    smpl_dir.value *= smpl_pos.value;
    
    RaySegment seg{{smpl_pos.coordinates, smpl_dir.coordinates}, LargeNumber};
    HitId hit = intersector.First(seg.ray, seg.length);
    if (hit)
    {
      RaySurfaceIntersection intersection{hit, seg};
      RGB col = (intersection.normal.array() * 0.5 + 0.5).cast<Color::RGBScalar>();
      return col;
    }
    else
    {
      return RGB::Zero();
    }
  };
};


using  AlgorithmParameters = RenderingParameters;
