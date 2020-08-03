#include "media_integrator.hxx"
#include "sampler.hxx"
#include "scene.hxx"
#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>

#ifdef HAVE_JSON
#include <rapidjson/document.h>
#endif

#include <boost/filesystem.hpp>

#include "renderingalgorithms_simplebase.hxx"
#include "rendering_randomwalk_impl.hxx"

using namespace RandomWalk;


void RenderingMediaTransmission1Helper(
  const Scene &scene, 
  const Ray &ray, 
  const Span<int> medium_changes,
  const Span<double> intersect_pos, 
  int num_hits)
{
  std::unordered_map<int, const Medium*> media_table;

  MediumTracker mt(scene);
  mt.initializePosition(ray.org);

  auto CheckOrInsert = [&](int medium_id) -> bool
  {
    auto[it, inserted] = media_table.insert({ medium_id, &mt.getCurrentMedium() });
    return inserted || it->second == &mt.getCurrentMedium();
  };

  CheckOrInsert(medium_changes[0]);

  std::printf("Media Trace:\n");

  int i = 0;
  auto intersections = scene.IntersectionsWithVolumes(ray, 0., LargeNumber);
  ASSERT_EQ(intersections.size(), num_hits);

  for (const auto &intersection : intersections)
  {
    assert(intersection.geom >= 0 && intersection.prim >= 0);
    mt.goingThroughSurface(ray.dir, intersection);
    const Double3 pos = ray.PointAt(intersection.t);
    const bool medium_ok = CheckOrInsert(medium_changes[i + 1]);
    std::printf("IS[%i]: pos=%f, med_expect=%p, got=%p\n", i, pos[2], media_table[medium_changes[i+1]], &mt.getCurrentMedium());
    EXPECT_NEAR(pos[2], intersect_pos[i], 1.e-6);
    EXPECT_TRUE(medium_ok);
    ++i;
  }
}


void CheckVolumePdfCoefficientsForMedium(const VolumePdfCoefficients &coeff, double tr, double sigma_start, double sigma_end, double tol)
{
  EXPECT_NEAR(coeff.transmittance, tr, tol);
  EXPECT_NEAR(coeff.pdf_scatter_fwd, tr*sigma_end, tol);
  EXPECT_NEAR(coeff.pdf_scatter_bwd, tr*sigma_start, tol);
}


TEST(Rendering, MediaTransmission1)
{
  const std::string scenestr {R"""(
shader none
medium med1 1 1 1 2 2 2
medium med2 1 1 1 2 2 2

medium med1
s 0 0 -1 1

medium med2
s 0 0 2 1

medium med1
s 0 0 3 1

medium med1
s 0 0 6 1

medium med2
s 0 0 6 0.5
)"""};
  Scene scene;
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();

  static constexpr int NHITS = 10;
  auto intersect_pos = std::array<double, NHITS>{
    -2, 0, 1, 2, 3, 4, 5, 5.5, 6.5, 7
  };
  auto medium_changes = std::array<int, NHITS + 1>{
  0, 1, 0, 2, 2, 1, 0, 1, 2, 1, 0
  };
  /*
    -2 -- 0  // med 1
            1 -- 3 // med 2
               2 --- 4 // med 1
                            5 ---------- 7    // med 1
                              5.5 -- 6.5 // med 2
  */
  Ray ray{{0, 0, -10}, {0, 0, 1}};
  RenderingMediaTransmission1Helper(scene, ray, AsSpan(medium_changes), AsSpan(intersect_pos), NHITS);
}


TEST(Rendering, MediaTransmission2)
{
  const std::string scenestr {R"""(
shader none
medium med1 1 1 1 2 2 2
medium med2 1 1 1 2 2 2

medium med1
transform 0 0 -1 0 0 0 2 2 2
m testing/scenes/unitcube.dae

medium med2
transform 0 0 2 0 0 0 2 2 2
m testing/scenes/unitcube.dae

medium med1
transform 0 0 3 0 0 0 2 2 2
m testing/scenes/unitcube.dae
)"""};
  Scene scene;
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();
  
  static constexpr int NHITS = 5;
  auto intersect_pos = std::array<double, NHITS>{
     0,   1,  2,  3,   4
  };
  auto media_changes = std::array<int, NHITS+1> {
  1, 0,   2,  2,  1,   0
  };
  /*
   -2 .. 0  // med1
            1  .. 3  // med2
               2 .. 4 // med1
  */
  const Ray ray{{0, 0, -1.}, {0, 0, 1}};
  RenderingMediaTransmission1Helper(scene, ray, AsSpan(media_changes), AsSpan(intersect_pos), NHITS);
}



namespace Eigen
{

template<class Derived>
auto ToVector(const ArrayBase<Derived> &v)
{
  using Scalar = typename internal::traits<Derived>::Scalar;
  if (v.rows() > 1 && v.cols() > 1)
    throw std::runtime_error("Need 1D array!");
  const auto temp = v.eval();
  return ToyVector<Scalar>(temp.data(), temp.data() + temp.size());
}

};



namespace test_medium
{

struct ControlPoint
{
  double x;
  Eigen::ArrayXd mu_s;
  Eigen::ArrayXd mu_a;
};


struct Result
{
  Eigen::ArrayXd transmittance;
  Eigen::ArrayXd mu_s, mu_a;
};

/* 
  ControlPoints define a piecewise constant medium.
  Its values are valid from x to the next control point.
*/
Result ComputeTransmittance(double x, const ToyVector<ControlPoint> &medium)
{
  const auto dim = medium.empty() ? 0 : medium.front().mu_s.size();

  Result res{};
  res.transmittance.resize(dim);
  res.transmittance.setOnes();
  res.mu_s.resize(dim);
  res.mu_s.setZero();
  res.mu_a.resize(dim);
  res.mu_a.setZero();

  if (medium.empty() || x < medium.front().x)
    return res;

  for (std::size_t i = 0; i<medium.size(); ++i)
  {
    double next_x = (i + 1 == medium.size()) ? std::numeric_limits<double>::infinity() : medium[i + 1].x;
    const auto &c = medium[i];
    const bool is_x_in_segment = x < next_x;
    if (is_x_in_segment)
    {
      next_x = x;
      res.mu_a = c.mu_a;
      res.mu_s = c.mu_s;
    }
    res.transmittance *= ((c.mu_a + c.mu_s)*(c.x - next_x)).exp();
    if (is_x_in_segment)
      break;
  }
  return res;
}

}


TEST(Rendering, TransmittanceEstimate)
{
  using namespace test_medium;
  Scene scene;
  const char* scenestr = R"""(
shader none
medium med1 1 1 1 2 2 2

m testing/scenes/unitcube.dae

transform 0 0 2 0 0 0
m testing/scenes/unitcube.dae
)""";
  LambdaSelection wavelengths = SelectRgbPrimaryWavelengths();
  const auto sigma_s = Color::RGBToSpectralSelection(RGB{ 1._rgb }, wavelengths.indices);
  const auto sigma_a = Color::RGBToSpectralSelection(RGB{ 2._rgb }, wavelengths.indices);
  const auto zero = decltype(sigma_s)::Zero();
  const auto exact = ComputeTransmittance(LargeNumber, {
    { -0.5, sigma_s, sigma_a},
    {  0.5, zero,   zero},
    {  1.5, sigma_s, sigma_a},
    {  2.5, zero ,   zero},
  });

  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();
  
  RaySegment seg{{{0.,0.,-10.}, {0.,0.,1.}}, LargeNumber};
  MediumTracker medium_tracker(scene);
  medium_tracker.initializePosition(seg.ray.org);
  Sampler sampler;
  VolumePdfCoefficients volume_pdf_coeff{};
  auto res = ::TransmittanceEstimate(scene, seg, medium_tracker, PathContext{wavelengths}, sampler, &volume_pdf_coeff);

  for (int i=0; i<static_size<Spectral3>(); ++i)
    ASSERT_NEAR(res[i], exact.transmittance[i], 1.e-3);
  CheckVolumePdfCoefficientsForMedium(
    volume_pdf_coeff, exact.transmittance.mean(), 0, 0, 1.e-6);
}


TEST(Rendering, NextInteractionEstimation)
{
  using namespace test_medium;
  Scene scene;
  const char* scenestr = R"""(
shader none
medium med1 3 3 3 0 0 0

m testing/scenes/unitcube.dae

transform 0 0 2 0 0 0
m testing/scenes/unitcube.dae
)""";
  LambdaSelection wavelengths = SelectRgbPrimaryWavelengths();  
  Spectral3 sigma = Color::RGBToSpectralSelection(RGB{3._rgb,3._rgb,3._rgb}, wavelengths.indices);
  const auto zero = decltype(sigma)::Zero();
  const auto mediumTestDescr = ToyVector<ControlPoint>{
    { -0.5, sigma, zero},
    {  0.5, zero,   zero},
    {  1.5, sigma, zero},
    {  2.5, zero ,   zero}
  };

  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();

  Sampler sampler;
  Ray ray{{0.,0.,-10}, {0.,0.,1.}};
  MediumTracker medium_tracker(scene);
  medium_tracker.initializePosition(ray.org);
  ASSERT_EQ(&medium_tracker.getCurrentMedium(), &scene.GetEmptySpaceMedium());

  int num_escaped = 0;
  Spectral3 weight_sum_interacted{0.};
  Spectral3 weight_sum_escaped{0.};
  
  constexpr int NUM_SAMPLES = 1000;
  for (int sample_num = 0; sample_num < NUM_SAMPLES; ++sample_num)
  {
    VolumePdfCoefficients volume_pdf_coeff{};
    //RadianceEstimatorBase::CollisionData collision(ray);
    medium_tracker.initializePosition(ray.org);
    TrackToNextInteraction(scene, ray, PathContext{wavelengths}, Spectral3::Ones(), sampler, medium_tracker, &volume_pdf_coeff,
      /*surface_visitor=*/[&](const SurfaceInteraction &intersection, double distance, const Spectral3 &weight)
      {
        FAIL();
      },
      /*volume visitor=*/[&](const VolumeInteraction &interaction, double distance, const Spectral3 &weight)
      {
        double x = distance + ray.org[2];
        const auto exact = ComputeTransmittance(x, mediumTestDescr);

        CheckVolumePdfCoefficientsForMedium(volume_pdf_coeff, exact.transmittance.mean(), 0., (exact.mu_a + exact.mu_s).mean(), 1.e-5);
        weight_sum_interacted += weight*interaction.sigma_s;
      },
      /* escape_visitor=*/[&](const Spectral3 &weight)
      {
        weight_sum_escaped += weight;
        num_escaped++;
      }
    );
  }
  weight_sum_escaped *= 1./(NUM_SAMPLES);
  weight_sum_interacted *= 1./(NUM_SAMPLES);
  
  auto sigma_e = Color::RGBToSpectralSelection(RGB{3._rgb}, wavelengths.indices);
  Spectral3 expected_transmissions = (-2.*sigma_e).exp();
  // No idea what the variance is. So I determine the acceptance threshold experimentally.
  for (int i=0; i<static_size<Spectral3>(); ++i)
  {
    ASSERT_NEAR(weight_sum_escaped[i], expected_transmissions[i], 1.e-2);
    ASSERT_NEAR(weight_sum_interacted[i], 1.-expected_transmissions[i], 1.e-2);
  }
}


TEST(Rendering, TrackBeam)
{
  using namespace test_medium;
  Scene scene;
  const char* scenestr = R"""(
shader none
medium med1 3 0

m testing/scenes/unitcube.dae

transform 0 0 2 0 0 0
m testing/scenes/unitcube.dae
)""";

  LambdaSelection wavelengths = SelectRgbPrimaryWavelengths();
  Spectral3 sigma = Color::RGBToSpectralSelection(RGB{3._rgb,3._rgb,3._rgb}, wavelengths.indices);
  const auto zero = decltype(sigma)::Zero();
  const auto mediumTestDescr = ToyVector<ControlPoint>{
    { -0.5, sigma, zero},
    {  0.5, zero,   zero},
    {  1.5, sigma, zero},
    {  2.5, zero ,   zero}
  };


  constexpr int NUM_PROBES = 5;
  double probe_locations[NUM_PROBES] = {
    -1., 0., 1., 2., 5.
  }; 
  
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();

  Sampler sampler;
  Ray ray{ {0.,0.,-10}, {0.,0.,1.} };
  MediumTracker medium_tracker(scene);
  
  int num_escaped = 0;
  Spectral3 weight_sum_probed[NUM_PROBES];
  for (int i=0; i<NUM_PROBES; ++i) 
    weight_sum_probed[i] = Spectral3::Zero();
  Spectral3 weight_sum_escaped{0.};
  
  constexpr int NUM_SAMPLES = 10000;
  for (int sample_num = 0; sample_num < NUM_SAMPLES; ++sample_num)
  {
    medium_tracker.initializePosition(ray.org);
    TrackBeam(scene, ray, PathContext{wavelengths}, sampler, medium_tracker,
      /*surface_visitor=*/[&](const SurfaceInteraction &intersection, const Spectral3 &weight)
      {
        FAIL();
      },
      /*segment visitor=*/[&](const RaySegment &segment, const Medium &medium, const PiecewiseConstantTransmittance &pct, const Spectral3 &weight)
      {
        for (int i = 0; i<NUM_PROBES; ++i)
        {
          // Convert probe location to distance along local segment.
          double distance = probe_locations[i]-segment.ray.org[2];
          if (distance >= 0 && distance < segment.length)
            weight_sum_probed[i] += weight*pct(distance);
        }
      },
      /* escape_visitor=*/[&](const Spectral3 &weight)
      {
        weight_sum_escaped += weight;
        num_escaped++;
      }
    );
  }
  
  for (int probe_idx=0; probe_idx<NUM_PROBES; ++probe_idx)
  {
    weight_sum_probed[probe_idx] *= 1./(NUM_SAMPLES);
    const auto exact = ComputeTransmittance(probe_locations[probe_idx], mediumTestDescr);
    for (int i=0; i<static_size<Spectral3>(); ++i)
    {
      ASSERT_NEAR(weight_sum_probed[probe_idx][i], exact.transmittance[i], 1.e-2);
    }
  }

  weight_sum_escaped *= 1./(NUM_SAMPLES);
  auto sigma_e = Color::RGBToSpectralSelection(RGB{3._rgb}, wavelengths.indices);
  Spectral3 expected_transmissions = (-2.*sigma_e).exp();
  // No idea what the variance is. So I determine the acceptance threshold experimentally.
  for (int i=0; i<static_size<Spectral3>(); ++i)
  {
    ASSERT_NEAR(weight_sum_escaped[i], expected_transmissions[i], 1.e-2);
  }
}


TEST(Rendering,PiecewiseConstantTransmittance)
{
  PiecewiseConstantTransmittance pct;
  pct.PushBack(0., Spectral3{2.});
  ASSERT_NEAR(pct(-0.1)[0], 2., 1.e-3);
  ASSERT_NEAR(pct(0.1)[0], 0., 1.e-3);
  
  pct.PushBack(1., Spectral3{1.});
  ASSERT_NEAR(pct(-0.1)[0], 2., 1.e-3);
  ASSERT_NEAR(pct(0.1)[0], 1., 1.e-3);
  ASSERT_NEAR(pct(0.9)[0], 1., 1.e-3);
  ASSERT_NEAR(pct(1.1)[0], 0., 1.e-3);
  
  pct.PushBack(2., Spectral3{3.});
  ASSERT_NEAR(pct(-0.1)[0], 2., 1.e-3);
  ASSERT_NEAR(pct(0.1)[0], 1., 1.e-3);
  ASSERT_NEAR(pct(0.9)[0], 1., 1.e-3);
  ASSERT_NEAR(pct(1.1)[0], 3., 1.e-3);
  ASSERT_NEAR(pct(1.9)[0], 3., 1.e-3);
  ASSERT_NEAR(pct(2.1)[0], 0., 1.e-3);
  
  pct.PushBack(3., Spectral3{4.});
  ASSERT_NEAR(pct(-0.1)[0], 2., 1.e-3);
  ASSERT_NEAR(pct(0.1)[0], 1., 1.e-3);
  ASSERT_NEAR(pct(0.9)[0], 1., 1.e-3);
  ASSERT_NEAR(pct(1.1)[0], 3., 1.e-3);
  ASSERT_NEAR(pct(1.9)[0], 3., 1.e-3);
  ASSERT_NEAR(pct(2.1)[0], 4., 1.e-3);
  ASSERT_NEAR(pct(2.9)[0], 4., 1.e-3);
  ASSERT_NEAR(pct(3.1)[0], 0., 1.e-3);
  
  pct.PushBack(InfinityFloat, Spectral3{5.});
  ASSERT_NEAR(pct(InfinityFloat)[0], 5., 1.e-3);
  ASSERT_NEAR(pct(Infinity)[0], 5., 1.e-3);
}


#if 0
TEST(NullPath, Mis)
{
  /* Transmission estimation using
     a) Delta Tracking
     b) Ratio Tracking
     MIS combination of a and b.
  */
  const double density = 0.6931471805599453; // Half of particles make it through.
  const double expected_transmissions = std::exp(-density);
  // NOTE: I think there is some bias in the result because the 
  // scatter coefficients given here are not converted by their exact value 
  // to the spectral representation!
  const auto scenestr = fmt::format(R"""(
shader none
medium med1 {} 0
m testing/scenes/unitcube.dae

medium default
shader black
transform 0 0 2.5 0 0 0
m testing/scenes/unitcube.dae
)""", density);
  
  Scene scene;
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();

  Sampler sampler;
  RaySegment segment{{ {0.,0.,-2}, {0.,0.,1.} }, 4};
  MediumTracker medium_tracker(scene);
  LambdaSelection wavelengths{};
  wavelengths.indices = Index3{ 0,0,0 };
  PathContext context{wavelengths};

  Accumulators::OnlineVariance<double> n_track_through;
  Accumulators::OnlineVariance<double> transmission_estimate;
  Accumulators::OnlineVariance<double> combined_estimate;
  Accumulators::OnlineVariance<double> mis_weights_track;
  Accumulators::OnlineVariance<double> mis_weights_transmit;
  const int N = 1000;
  for (int iter=0; iter<N; ++iter)
  {
    double combined_estimator = 0;
    {
      medium_tracker.initializePosition(segment.ray.org);
      auto [interaction, dist, factors] = nullpath::Tracking(scene, segment.ray, sampler, medium_tracker, context);
      assert (interaction);
      double estimator = (factors.throughput[0] / factors.pdf_track[0]) *
          ((interaction && mpark::get_if<SurfaceInteraction>(&(*interaction))) ? 1. : 0.);
      n_track_through += estimator;
      double mis_weight = factors.pdf_track[0] / (factors.pdf_track[0] + factors.pdf_nulls[0]);
      combined_estimator += mis_weight * estimator;
      mis_weights_track += mis_weight;
    }
    {
      medium_tracker.initializePosition(segment.ray.org);
      auto factors = nullpath::Transmission(scene, segment, sampler, medium_tracker, context);
      double estimator = factors.throughput[0] / factors.pdf_nulls[0];
      transmission_estimate += estimator;
      double mis_weight = factors.pdf_nulls[0] / (factors.pdf_track[0] + factors.pdf_nulls[0]);
      combined_estimator += mis_weight * estimator;
      mis_weights_transmit += mis_weight;
    }
    combined_estimate += combined_estimator;
  }
  fmt::print("transmission avg = {} +/- {}\n", transmission_estimate.Mean(),transmission_estimate.MeanErr());
  fmt::print("tracking avg = {} +/- {}\n", n_track_through.Mean(),n_track_through.MeanErr());
  fmt::print("combined estimate = {} +/- {}\n", combined_estimate.Mean(), combined_estimate.MeanErr());
  fmt::print("mis_weights_track = {}\n", mis_weights_track.Mean());
  fmt::print("mis_weights_transmit = {}\n", mis_weights_transmit.Mean());
  fmt::print("expected = {}\n", expected_transmissions);
  EXPECT_NEAR(transmission_estimate.Mean(), expected_transmissions, 2.*transmission_estimate.MeanErr());
  EXPECT_NEAR(n_track_through.Mean(), expected_transmissions, 2.*n_track_through.MeanErr());
  EXPECT_NEAR(combined_estimate.Mean(), expected_transmissions, 2.*combined_estimate.MeanErr());
  EXPECT_LE(combined_estimate.MeanErr(), transmission_estimate.MeanErr());
  EXPECT_LE(combined_estimate.MeanErr(), n_track_through.MeanErr());
}
#endif