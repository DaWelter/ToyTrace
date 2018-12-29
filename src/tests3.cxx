#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>
#include <rapidjson/document.h>
#include <boost/filesystem.hpp>

#include "renderingalgorithms.hxx"
#include "rendering_randomwalk_impl.hxx"

using namespace RandomWalk;


void RenderingMediaTransmission1Helper(
  const Scene &scene, 
  const Ray &ray, 
  const Medium **media_after_intersect, 
  const double *intersect_pos, 
  int NHITS)
{
  MediumTracker mt(scene);
  mt.initializePosition({0, 0, -10});
  ASSERT_EQ(&mt.getCurrentMedium(), &scene.GetEmptySpaceMedium());
  RaySegment seg{ray, LargeNumber};
  std::printf("Media Trace:\n");
  IterateIntersectionsBetween iter{seg, scene};
  RaySurfaceIntersection intersection;
  int i = 0;
  for (; iter.Next(seg, intersection); ++i)
  {
    EXPECT_LE(i, NHITS-1);
    mt.goingThroughSurface(seg.ray.dir, intersection);
    std::printf("IS[%i]: pos=%f, med_expect=%p, got=%p\n", i, intersection.pos[2], media_after_intersect[i], &mt.getCurrentMedium());
    EXPECT_NEAR(intersection.pos[2], intersect_pos[i], 1.e-6);
    ASSERT_EQ(&mt.getCurrentMedium(), media_after_intersect[i]);
  }
}


void CheckVolumePdfCoefficientsForMedium(const VolumePdfCoefficients &coeff, double tr, double sigma_start, double sigma_end, double tol)
{
  EXPECT_NEAR(coeff.pdf_scatter_fwd, tr*sigma_end, tol);
  EXPECT_NEAR(coeff.pdf_scatter_bwd, tr*sigma_start, tol);
  EXPECT_NEAR(coeff.transmittance, tr, tol);
}


TEST(Rendering, MediaTransmission1)
{
  const std::string scenestr {R"""(
shader invisible
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
  auto GetMedium = [&scene](int i) {
    return scene.GetMaterialOf({&scene.GetGeometry(1), i}).medium;
  };
  
  for (int i=0; i<5; ++i)
  {
    std::printf("Medium of prim %i = %p\n", i, GetMedium(i));
  }
  const Medium *vac = &scene.GetEmptySpaceMedium();
  const Medium *m1 = GetMedium(0);
  const Medium *m2 = GetMedium(1);
  const Medium *media_after_intersect[] = {
    m1, vac, m2, m2, m1, vac, m1, m2, m1, vac
  };
  int NHITS = sizeof(media_after_intersect)/sizeof(void*);
  double intersect_pos[] = {
    -2, 0,   1,  2,  3,   4,  5,  5.5,6.5, 7
  };
  Ray ray{{0, 0, -10}, {0, 0, 1}};
  RenderingMediaTransmission1Helper(scene, ray, media_after_intersect, intersect_pos, NHITS);
}


TEST(Rendering, MediaTransmission2)
{
  const std::string scenestr {R"""(
shader invisible
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
  auto GetMedium = [&scene](int i) {
    return scene.GetMaterialOf({&scene.GetGeometry(0), i}).medium;
  };
  const Medium *vac = &scene.GetEmptySpaceMedium();
  const Medium *m1 = GetMedium(0);
  const Medium *m2 = GetMedium(12);
  const Medium *media_after_intersect[] = {
    m1, vac, m2, m2, m1, vac
  };
  int NHITS = sizeof(media_after_intersect)/sizeof(void*);
  double intersect_pos[] = {
    -2, 0,   1,  2,  3,   4,
  };
  Ray ray{{0, 0, -10}, {0, 0, 1}}; // Offset to prevent hiting two adjacent triangles exactly at the interfacing edge.
  RenderingMediaTransmission1Helper(scene, ray, media_after_intersect, intersect_pos, NHITS);
}


TEST(Rendering, TransmittanceEstimate)
{
  Scene scene;
  const char* scenestr = R"""(
shader invisible
medium med1 1 1 1 2 2 2

m testing/scenes/unitcube.dae

transform 0 0 2 0 0 0
m testing/scenes/unitcube.dae
)""";
  const double total_length = 2; // Because 2 unit cubes.
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();
  Index3 lambda_idx = Color::LambdaIdxClosestToRGBPrimaries();
  RadianceEstimatorBase rt(scene);
  MediumTracker medium_tracker(scene);
  double ray_offset = 0.1; // because not so robust handling of intersection edge cases. No pun intended.
  RaySegment seg{{{ray_offset,0.,-10.}, {0.,0.,1.}}, LargeNumber};
  medium_tracker.initializePosition(seg.ray.org);
  ASSERT_EQ(&medium_tracker.getCurrentMedium(), &scene.GetEmptySpaceMedium());
  VolumePdfCoefficients volume_pdf_coeff{};
  auto res = rt.TransmittanceEstimate(seg, medium_tracker, PathContext{lambda_idx}, &volume_pdf_coeff);
  
  auto sigma_e = Color::RGBToSpectralSelection(RGB{3._rgb}, lambda_idx);
  Spectral3 expected = (-total_length*sigma_e).exp();
  
  for (int i=0; i<static_size<Spectral3>(); ++i)
    ASSERT_NEAR(res[i], expected[i], 1.e-3);
  
  double approximated_tr = (-1.*sigma_e).exp().mean()*(-1*sigma_e).exp().mean(); 
  // Concatentation of the approximations from the two cubes.
  CheckVolumePdfCoefficientsForMedium(
    volume_pdf_coeff, approximated_tr, 0, 0, 1.e-6);
}


TEST(Rendering, NextInteractionEstimation)
{
  Scene scene;
  const char* scenestr = R"""(
shader invisible
medium med1 3 3 3 0 0 0

m testing/scenes/unitcube.dae

transform 0 0 2 0 0 0
m testing/scenes/unitcube.dae
)""";
  scene.ParseNFFString(scenestr);
  scene.BuildAccelStructure();
  Index3 lambda_idx = Color::LambdaIdxClosestToRGBPrimaries();
  
  Spectral3 sigma = Color::RGBToSpectralSelection(RGB{3._rgb,3._rgb,3._rgb}, lambda_idx);
  double cube1_start = -0.5;
  double cube2_start = 1.5;
  double camera_start = -10.;
  auto AnalyticTrApprox = [=](double x) -> double
  {
    double first_cube_tr = (-sigma).exp().mean();
    if (x < cube1_start) 
      return 1.;
    if (x < cube1_start+1.)
      return (-(x-cube1_start)*sigma).exp().mean();
    if (x < cube2_start)
      return first_cube_tr;
    if (x < cube2_start+1.)
      return first_cube_tr*(-(x-cube2_start)*sigma).exp().mean();
    return first_cube_tr*(-sigma).exp().mean();
  };
  auto AnalyticSigmaApprox = [=](double x) -> double
  {
    if (cube1_start < x && x < cube1_start+1) return sigma.mean();
    if (cube2_start < x && x < cube2_start+1) return sigma.mean();
    return 0.;
  };
  
  
  RadianceEstimatorBase rt(scene);
  MediumTracker medium_tracker(scene);
  double ray_offset = 0.1; // because not so robust handling of intersection edge cases. No pun intended.
  Ray ray{{ray_offset,0.,camera_start}, {0.,0.,1.}};
  medium_tracker.initializePosition(ray.org);
  ASSERT_EQ(&medium_tracker.getCurrentMedium(), &scene.GetEmptySpaceMedium());
  
  int num_escaped = 0;
  Spectral3 weight_sum_interacted{0.};
  Spectral3 weight_sum_escaped{0.};
  
  constexpr int NUM_SAMPLES = 1000;
  for (int sample_num = 0; sample_num < NUM_SAMPLES; ++sample_num)
  {
    VolumePdfCoefficients volume_pdf_coeff{};
    RadianceEstimatorBase::CollisionData collision(ray);
    medium_tracker.initializePosition(ray.org);
    rt.TrackToNextInteraction(collision, medium_tracker, PathContext{lambda_idx}, &volume_pdf_coeff);
    EXPECT_EQ(collision.segment.ray.org[2], ray.org[2]);
    
    {
      double x = collision.smpl.t + camera_start;
      double tr = AnalyticTrApprox(x);
      double s = AnalyticSigmaApprox(x);
      CheckVolumePdfCoefficientsForMedium(volume_pdf_coeff, tr, 0., s, 1.e-5);
    }
    
    if (RadianceEstimatorBase::IsNotEscaped(collision))
    {
      weight_sum_interacted += collision.smpl.weight;
    }
    else
    {
      weight_sum_escaped += collision.smpl.weight;
      num_escaped++;
    }
  }
  weight_sum_escaped *= 1./(NUM_SAMPLES);
  weight_sum_interacted *= 1./(NUM_SAMPLES);
  
  auto sigma_e = Color::RGBToSpectralSelection(RGB{3._rgb}, lambda_idx);
  Spectral3 expected_transmissions = (-2.*sigma_e).exp();
  // No idea what the variance is. So I determine the acceptance threshold experimentally.
  for (int i=0; i<static_size<Spectral3>(); ++i)
  {
    ASSERT_NEAR(weight_sum_escaped[i], expected_transmissions[i], 1.e-2);
    ASSERT_NEAR(weight_sum_interacted[i], 1.-expected_transmissions[i], 1.e-2);
  }
}
