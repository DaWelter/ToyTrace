#include "media_integrator.hxx"
#include "ray.hxx"
#include "shader_util.hxx"
#include "rendering_util.hxx"

namespace TrackingDetail
{


// Is passing parameters like that as efficient has having the same number of item as normal individual arguments?
// Ref: Kutz et a. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"
inline void ComputeProbabilitiesHistoryScheme(
  const Spectral3 &weights,
  std::initializer_list<std::reference_wrapper<const Spectral3>> sigmas,
  std::initializer_list<std::reference_wrapper<double>> probs)
{
  double normalization = 0.;
  auto it_sigma = sigmas.begin();
  auto it_probs = probs.begin();
  for (; it_sigma != sigmas.end(); ++it_sigma, ++it_probs)
  {
    const Spectral3 &s = it_sigma->get();
    assert(s.minCoeff() >= 0.);
    double p = (s*weights).mean();
    it_probs->get() = p;
    normalization += p;
  }
  if (normalization > 0.)
  {
    double norm_inv = 1. / normalization;
    for (it_probs = probs.begin(); it_probs != probs.end(); ++it_probs)
      it_probs->get() *= norm_inv;
  }
  else // Zeroed weights?
  {
    const double p = 1.0 / probs.size();
    for (it_probs = probs.begin(); it_probs != probs.end(); ++it_probs)
      it_probs->get() = p;
  }
}


Medium::InteractionSample HomogeneousSpectralTracking(const materials::Medium &medium, const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context)
{
  // Ref: Kutz et a. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"
  // Much simplified with constant coefficients.
  // Also, very importantly, sigma_s is not multiplied to the final weight! Compare with Algorithm 4, Line 10.
  Medium::InteractionSample smpl{
    0.,
    Spectral3::Ones(),
    Spectral3::Zero()
  };

  const auto coeffs = medium.EvaluateCoeffs(Double3::Zero(), context);
  double sigma_t_majorant = coeffs.sigma_t.maxCoeff();
  const Spectral3 sigma_n = sigma_t_majorant - coeffs.sigma_t;
  double inv_sigma_t_majorant = 1. / sigma_t_majorant;
  constexpr int emergency_abort_max_num_iterations = 100;
  int iteration = 0;
  while (++iteration < emergency_abort_max_num_iterations)
  {
    smpl.t -= std::log(sampler.Uniform01()) * inv_sigma_t_majorant;
    if (smpl.t > segment.length)
    {
      return smpl;
    }
    else
    {
      assert(sigma_n.minCoeff() >= -1.e-3); // By definition of the majorante
      double prob_t, prob_n;
      TrackingDetail::ComputeProbabilitiesHistoryScheme(smpl.weight*initial_weights, { coeffs.sigma_s, sigma_n }, { prob_t, prob_n });
      double r = sampler.Uniform01();
      if (r < prob_t) // Scattering/Absorption
      {
        smpl.weight *= inv_sigma_t_majorant / prob_t;
        smpl.sigma_s = coeffs.sigma_s;
        return smpl;
      }
      else // Null collision
      {
        smpl.weight *= inv_sigma_t_majorant / prob_n * sigma_n;
      }
    }
  }
  assert(false);
  return smpl;
}


Medium::InteractionSample SpectralTracking(const materials::Medium &medium, const RaySegment& segment, const Spectral3 &initial_weights, Sampler& sampler, const PathContext &context)
{
  Medium::InteractionSample smpl{
    0.,
    initial_weights,
    Spectral3::Zero()
  };

  const double sigma_t_majorant = medium.ComputeMajorante(segment, context).maxCoeff();
  assert(std::isfinite(sigma_t_majorant));
  if (sigma_t_majorant <= 0.)
  {
    smpl.t = LargeNumber;
    return smpl;
  }

  const double mean_free_path_length = 1. / sigma_t_majorant;
  // Then start tracking.
  // - No russian roulette because the probability to arrive
  // beyond the end of the segment must be identical to the transmissivity.
  // - Need to limit the number of iterations to protect against evil edge
  // cases where a particles escapes outside of the scene boundaries but
  // due to incorrect collisions the path tracer thinks that we are in a medium.
  constexpr int emergency_abort_max_num_iterations = 1000;
  int iteration = 0;
  while (++iteration < emergency_abort_max_num_iterations)
  {
    smpl.t -= std::log(sampler.Uniform01()) * mean_free_path_length;
    if (smpl.t > segment.length)
    {
      return smpl;
    }
    else
    {
      const Double3 pos = segment.ray.PointAt(smpl.t);
      const auto coeffs = medium.EvaluateCoeffs(pos, context);
      Spectral3 sigma_n = sigma_t_majorant - coeffs.sigma_t;
      assert(sigma_n.minCoeff() >= -1.e-3); // By definition of the majorante
      assert(coeffs.sigma_s.minCoeff() >= 0.);
      sigma_n = sigma_n.cwiseMax(0.);
      double prob_t, prob_n;
      TrackingDetail::ComputeProbabilitiesHistoryScheme(smpl.weight, { coeffs.sigma_s, sigma_n }, { prob_t, prob_n });
      double r = sampler.Uniform01();
      if (r < prob_t) // Scattering/Absorption
      {
        smpl.weight *= mean_free_path_length / prob_t;
        smpl.sigma_s = coeffs.sigma_s;
        return smpl;
      }
      else // Null collision
      {
        smpl.weight *= mean_free_path_length / prob_n * sigma_n;
      }
    }
  }
  assert(false);
  return smpl;
}


Spectral3 EstimateTransmission(const Medium &medium, const RaySegment &segment, Sampler &sampler, const PathContext &context)
{
  Spectral3 estimate{ 1. };
  Spectral3 sigma_s, sigma_a;

  const double sigma_t_majorant = medium.ComputeMajorante(segment, context).maxCoeff();
  assert(std::isfinite(sigma_t_majorant));
  if (sigma_t_majorant <= 0.)
  {
    return estimate;
  }

  double inv_sigma_t_majorant = 1. / sigma_t_majorant;
  // Then start tracking.
  double t = 0.;
  int iteration = 0;
  bool gogogo = true;
  do
  {
    t -= std::log(sampler.Uniform01()) * inv_sigma_t_majorant;
    if (t > segment.length)
    {
      gogogo = false;
    }
    else
    {
      const Double3 pos = segment.ray.PointAt(t);
      auto coeffs = medium.EvaluateCoeffs(pos, context);
      Spectral3 sigma_n = sigma_t_majorant - coeffs.sigma_t;
      assert(sigma_n.minCoeff() >= 0.); // By definition of the majorante
      estimate *= sigma_n * inv_sigma_t_majorant;

      gogogo = TrackingDetail::RussianRouletteSurvival(
        estimate.maxCoeff(), iteration++, sampler, [&estimate](double q) { estimate *= q; });
    }
  } while (gogogo);
  return estimate;
}


void HomogeneousConstructShortBeamTransmittance(const Medium& medium, const RaySegment& segment, Sampler& sampler, const PathContext& context, PiecewiseConstantTransmittance& pct)
{
  const auto coeffs = medium.EvaluateCoeffs(Double3::Zero(), context);
  constexpr auto N = static_size<Spectral3>();
  std::pair<double, int> items[N];
  for (int i = 0; i < N; ++i)
  {
    items[i].first = -std::log(1 - sampler.GetRandGen().Uniform01()) / coeffs.sigma_t[i];
    items[i].second = i;
    for (int j = i; j > 0 && items[j - 1].first > items[j].first; --j)
    {
      std::swap(items[j - 1], items[j]);
    }
  }
  // Zero out the spectral channels one after another.
  Spectral3 w = Spectral3::Ones();
  for (int i = 0; i < N; ++i)
  {
    pct.PushBack(items[i].first, w);
    w[items[i].second] = 0;
  }
}

} // TrackingDetail

namespace nullpath
{

Spectral33 FixNan(const Spectral33 &x)
{
  return x.isNaN().select(Spectral33::Constant(Infinity), x);
}

void AccumulateContributions(Spectral33 &w, const Spectral33 &x)
{
  w *= x;
  // Because we could have had 0 times infinity in the line above.
  // In this case "1/f" should "win", and so the result becomes infinity.
  w = FixNan(w);
}

void AccumulateContributions(Spectral33 &w, const Spectral3 &f, const Spectral3 &p)
{
  Spectral33 tmp = f.cwiseInverse().matrix() * p.matrix().transpose();
  // Should only be nan if both parts were equals zero. 
  // In this case "1/f" should "win", and so the result becomes infinity.
  tmp = tmp.isNaN().select(Spectral33::Constant(Infinity), tmp); 
  AccumulateContributions(w, tmp);
}

InteractionSample Tracking(const materials::Medium &medium, const RaySegment& segment, Sampler& sampler, const PathContext &context)
{
  InteractionSample smpl;

  static const int hero_wavelength = 0;
  const Spectral3 sigma_t_majorant = medium.ComputeMajorante(segment, context);
  
  const double mean_free_path_length = 1. / sigma_t_majorant[hero_wavelength];

  constexpr int emergency_abort_max_num_iterations = 10000;
  int iteration = 0;
  while (++iteration < emergency_abort_max_num_iterations)
  {
    double tlocal = std::min(LargeNumber, -std::log(1.0 - sampler.Uniform01()) * mean_free_path_length);
    const Spectral3 tr = (-std::min(tlocal, segment.length - smpl.t) * sigma_t_majorant).exp();
    smpl.t += tlocal;
    const Double3 pos = segment.ray.PointAt(smpl.t);
    smpl.coeffs = medium.EvaluateCoeffs(pos, context);
    if (smpl.t >= segment.length)
    {
      // TODO: Is this needed? I don't think these factors would cancel in the contribution calculation.
      // But they don't appear in the algorithm listing from the paper.
      AccumulateContributions(smpl.weights_track, tr, tr);
      AccumulateContributions(smpl.weights_nulls, tr, tr);
      return smpl;
    }
    else
    {
      Spectral3 sigma_n = sigma_t_majorant - smpl.coeffs.sigma_t;
      const double prob_n = sigma_n[hero_wavelength] * mean_free_path_length;
      const double r = sampler.Uniform01();
      if (r < prob_n)
      {
        AccumulateContributions(smpl.weights_track, tr*sigma_n, tr*sigma_n);
        AccumulateContributions(smpl.weights_nulls, tr*sigma_n, tr*sigma_t_majorant);
      }
      else
      {
        AccumulateContributions(smpl.weights_track, tr*smpl.coeffs.sigma_t, tr*smpl.coeffs.sigma_t);
        AccumulateContributions(smpl.weights_nulls, tr*smpl.coeffs.sigma_t, tr);
        return smpl;
      }
    }
  }
  assert(false);
  return smpl;
}

ThroughputAndPdfs Transmission(const Medium &medium, const RaySegment &segment, Sampler &sampler, const PathContext &context)
{
  ThroughputAndPdfs smpl;

  static const int hero_wavelength = 0;
  const Spectral3 sigma_t_majorant = medium.ComputeMajorante(segment, context);

  const double mean_free_path_length = 1. / sigma_t_majorant[hero_wavelength];

  constexpr int emergency_abort_max_num_iterations = 10000;
  int iteration = 0;
  double t = 0;
  while (++iteration < emergency_abort_max_num_iterations)
  {
    double tlocal = std::min(LargeNumber, -std::log(1.0 - sampler.Uniform01()) * mean_free_path_length);
    const Spectral3 tr = (-std::min(tlocal, segment.length - t) * sigma_t_majorant).exp();
    t += tlocal;
    if (t >= segment.length)
    {
      AccumulateContributions(smpl.weights_track, tr, tr);
      AccumulateContributions(smpl.weights_nulls, tr, tr);
      return smpl;
    }
    else
    {     
      const Double3 pos = segment.ray.PointAt(t);
      auto coeffs = medium.EvaluateCoeffs(pos, context);
      Spectral3 sigma_n = sigma_t_majorant - coeffs.sigma_t;
      AccumulateContributions(smpl.weights_track, tr*sigma_n, tr*sigma_n);
      AccumulateContributions(smpl.weights_nulls, tr*sigma_n, tr*sigma_t_majorant);
    }
    if ((smpl.weights_nulls.matrix().diagonal().array() > LargeNumber).all())
    {
      smpl.weights_nulls.setConstant(Infinity);
      return smpl;
    }
  }
  assert(false);
  return smpl;
}


std::tuple<MaybeSomeInteraction, double, ThroughputAndPdfs>
Tracking(
  const Scene &scene,
  const Ray &ray,
  Sampler &sampler,
  MediumTracker &medium_tracker,
  const PathContext &context)
{
  ThroughputAndPdfs weights;

  double tfar = LargeNumber;
  const auto hit = scene.FirstIntersection(ray, 0., tfar);

  // The factor by which tfar is decreased is meant to prevent intersections which lie close to the end of the query segment.
  auto iter = VolumeSegmentIterator(scene, ray, medium_tracker, 0., tfar * 0.9999);
  for (; iter; ++iter)
  {
    auto[snear, sfar] = iter.Interval();
    const Medium& medium = *iter;
    RaySegment segment{ ray, snear, sfar };

    //const auto medium_smpl = medium.SampleInteractionPoint(segment, weight*initial_weight, sampler, context);
    const InteractionSample smpl = Tracking(medium, segment, sampler, context);

    AccumulateContributions(weights.weights_nulls, smpl.weights_nulls);
    AccumulateContributions(weights.weights_track, smpl.weights_track);

    const bool interacted_w_medium = smpl.t < segment.length;
    segment.length = interacted_w_medium ? smpl.t : segment.length;

    if (interacted_w_medium)
    {
      tfar = snear + smpl.t;
      VolumeInteraction vi{ ray.PointAt(tfar), medium, Spectral3{ 0. }, smpl.coeffs.sigma_s };
      return { vi, tfar, weights };
    }
  }
  // If we get here, there was no scattering in the medium.
  if (hit)
  {
    return { *hit, tfar, weights };
  }
  else
  {
    return { MaybeSomeInteraction{}, LargeNumber,  weights };
  }
}


ThroughputAndPdfs Transmission(
  const Scene &scene,
  RaySegment seg,
  Sampler &sampler,
  MediumTracker &medium_tracker,
  const PathContext &context)
{
  ThroughputAndPdfs result;

  seg.length *= 0.9999; // To avoid intersections with the adjacent nodes/positions/surfaces.

  if (scene.IsOccluded(seg.ray, 0., seg.length))
  {
    result.weights_nulls.setConstant(Infinity);
    result.weights_track.setConstant(Infinity);
    return result;
  }

  auto iter = VolumeSegmentIterator(scene, seg.ray, medium_tracker, 0., seg.length);
  for (; iter; ++iter)
  {
    auto[snear, sfar] = iter.Interval();
    const Medium& medium = *iter;
    const RaySegment subsegment{ seg.ray, snear, sfar };
    const auto tr = Transmission(medium, subsegment, sampler, context);
    AccumulateContributions(result.weights_nulls, tr.weights_nulls);
    AccumulateContributions(result.weights_track, tr.weights_track);
  }

  return result;
}


} // namespace nullpath
