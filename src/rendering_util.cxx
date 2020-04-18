#include "rendering_util.hxx"
#include "light.hxx"
#include "lightpicker_ucb.hxx"

MediumTracker::MediumTracker(const Scene& _scene) noexcept
  : current{nullptr},
    media{}, // Note: Ctor zero-initializes the media array.
    scene{&_scene}
{
}


void MediumTracker::initializePosition(const Double3& pos)
{
  std::fill(media.begin(), media.end(), nullptr);
  current = &scene->GetEmptySpaceMedium();
  {
    const Box bb = scene->GetBoundingBox();
    // The InBox check is important. Otherwise I would not know how long to make the ray.
    if (bb.InBox(pos))
    {
      const double distance_to_go = 2. * (bb.max-bb.min).maxCoeff(); // Times to due to the diagonal.
      Ray r{ pos, {-1., 0., 0.} };
      r.org[0] += distance_to_go;

      // Does the order matter? I don't think so because 
      // goingThroughSurface essentially counts intersections for every encountered 
      // Medium. The final counter is invariant to the order.
      auto intersections = scene->IntersectionsWithVolumes(r, 0., distance_to_go);
      for (auto is : intersections)
        this->goingThroughSurface(r.dir, is);
      intersections = scene->IntersectionsWithSurfaces(r, 0., distance_to_go);
      for (auto is : intersections)
        this->goingThroughSurface(r.dir, is);
    }
  }
}


Spectral3 TransmittanceEstimate(const Scene &scene, RaySegment seg, MediumTracker &medium_tracker, const PathContext &context, Sampler &sampler, VolumePdfCoefficients *volume_pdf_coeff)
{
    Spectral3 result{1.};
    
    seg.length *= 0.9999; // To avoid intersections with the adjacent nodes/positions/surfaces.

    if (scene.IsOccluded(seg.ray, 0., seg.length))
      return Spectral3::Zero();

    auto iter = VolumeSegmentIterator(scene, seg.ray, medium_tracker, 0., seg.length);
    for (; iter; ++iter)
    {
      auto[snear, sfar] = iter.Interval();
      const Medium& medium = *iter;
      const RaySegment subsegment{ seg.ray, snear, sfar };

      if (volume_pdf_coeff)
      {
        VolumePdfCoefficients local_coeff = medium.ComputeVolumePdfCoefficients(subsegment, context);
        // Note: Float comparisons should be in order here because segment start and end values are propagated into
        // the callback pristinely.
        Accumulate(*volume_pdf_coeff, local_coeff, snear==0., sfar==seg.length);
      }
      result *= medium.EvaluateTransmission(subsegment, sampler, context);
    }

    return result;
}


// Returns pdf w.r.t. radians!
std::tuple<Spectral3, Pdf, RaySegment, Lights::LightRef> ComputeDirectLighting(
  const Scene &scene, 
  const SomeInteraction & interaction, 
  const Lightpickers::LightSelectionProbabilityMap &light_probabilities, 
  const MediumTracker &medium_tracker, 
  const PathContext &context, 
  Sampler &sampler)
{
  RaySegment segment_to_light;
  Pdf pdf;
  Spectral3 light_radiance{};
  Spectral3 path_weight = Spectral3::Ones();
  Lights::LightRef light_ref;

  light_probabilities.Sample(sampler, [&](auto &&light, double prob, const Lights::LightRef &light_ref_)
  {
    std::tie(segment_to_light, pdf, light_radiance) = light.SampleConnection(interaction, scene, sampler, context);
    pdf *= prob;
    path_weight /= (double)(pdf);
    if constexpr (!std::remove_reference<decltype(light)>::type::IsAngularDistribution())
    {
      path_weight *= 1./Sqr(segment_to_light.length);
      if (!pdf.IsFromDelta())
      {
        const double pdf_conversion_factor = PdfConversion::AreaToSolidAngle(
            segment_to_light.length,
            segment_to_light.ray.dir,
            light.SurfaceNormal()) * pdf;
        pdf *= pdf_conversion_factor;
      }
    }
    light_ref = light_ref_;
  });

  MediumTracker medium_tracker_copy{ medium_tracker }; // Copy because well don't want to keep modifications.
  
  if (const SurfaceInteraction* si = std::get_if<SurfaceInteraction>(&interaction); si)
  {
    MaybeGoingThroughSurface(medium_tracker_copy, segment_to_light.ray.dir, *si);
  }
  // Surface specific
  
  Spectral3 transmittance = TransmittanceEstimate(scene, segment_to_light, medium_tracker_copy, context, sampler);
  path_weight *= transmittance;

  Spectral3 incident_radiance_estimator = path_weight*light_radiance;
  return std::make_tuple(incident_radiance_estimator, pdf, segment_to_light, light_ref);
}