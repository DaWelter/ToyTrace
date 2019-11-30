#include "rendering_util.hxx"

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
      const double distance_to_go = 2. * (bb.max-bb.min).maxCoeff(); // Times to due to the diagonal.
      Ray r{ pos, {-1., 0., 0.} };
      r.org[0] += distance_to_go;

      // Does the order matter? I don't think so because 
      // goingThroughSurface essentially counts intersections for every encountered 
      // Medium. The final counter is invariant to the order.
      auto intersections = scene.IntersectionsWithVolumes(r, 0., distance_to_go);
      for (auto is : intersections)
        this->goingThroughSurface(r.dir, is);
      intersections = scene.IntersectionsWithSurfaces(r, 0., distance_to_go);
      for (auto is : intersections)
        this->goingThroughSurface(r.dir, is);
    }
  }
}


Spectral3 TransmittanceEstimate(const Scene &scene, RaySegment seg, MediumTracker &medium_tracker, const PathContext &context, Sampler &sampler, VolumePdfCoefficients *volume_pdf_coeff)
{
    Spectral3 result{1.};

    if (scene.IsOccluded(seg.ray, 0., seg.length))
      return Spectral3::Zero();

    auto iter = VolumeSegmentIterator(scene, seg.ray, 0., seg.length);
    for (; iter; iter.Next(seg.ray, medium_tracker))
    {
      auto[snear, sfar] = iter.Interval();
      const Medium& medium = medium_tracker.getCurrentMedium();
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
