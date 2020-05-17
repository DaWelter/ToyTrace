#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "span.hxx"
#include "sampler.hxx"


namespace vmf_fitting
{

static constexpr float K_THRESHOLD = 1.e-4f;
// Picked such that exp(-2k)>0. Otherwise we get -inf when the log is applied!
static constexpr float K_THRESHOLD_MAX = 4.e1f; 

// 320 Bytes
struct VonMisesFischerMixture
{
  static constexpr int NUM_COMPONENTS = 8;
  using WeightArray = Eigen::Array<float, NUM_COMPONENTS, 1>;
  using ConcArray = Eigen::Array<float, NUM_COMPONENTS, 1>;
  using MeansArray = Eigen::Array<float, NUM_COMPONENTS, 3>;

  MeansArray means;
  WeightArray weights; // Of components
  ConcArray concentrations;
};

float Pdf(const Eigen::Vector3f &mu, const float k, const Eigen::Vector3f &x) noexcept;
float Pdf(const VonMisesFischerMixture &mixture, const Eigen::Vector3f &pos) noexcept;
Eigen::Vector3f Sample(const VonMisesFischerMixture &mixture, std::array<double, 3> rs) noexcept;
void InitializeForUnitSphere(VonMisesFischerMixture &mixture) noexcept;

namespace incremental
{

struct Data
{
  static constexpr int NC = VonMisesFischerMixture::NUM_COMPONENTS;
  // Don't have to be initialize because if data_count==0 then 
  // those vars are assigned to anyway. (Otherwise they're added to.)
  Eigen::Array<float, NC, 3> avg_positions;
  Eigen::Array<float, NC, 1> avg_responsibilities;
  Eigen::Array<float, NC, 1> avg_responsibilities_unweighted;
  std::uint64_t data_count = 0;
  std::uint64_t data_count_weights = 0;
  float avg_weights = 0.f;
};


struct Params
{
  // Can be interpreted as number of samples where the prior start to count less for the final result than the samples.
  float prior_nu = 100.0f; 
  float prior_tau = 100.0f; 
  float prior_alpha = 100.0f;
  int maximization_step_every = 10;
  VonMisesFischerMixture* prior_mode = nullptr;
};


void Fit(VonMisesFischerMixture &mixture, Data &fitdata, const Params &params, Span<const Eigen::Vector3f> data, Span<const float> data_weights) noexcept;

// Large enough so that 1/eps is still finite ...
inline static constexpr float eps = 1.e-38f;

} // namespace incremental

};