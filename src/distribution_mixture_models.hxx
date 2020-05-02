#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "span.hxx"
#include "sampler.hxx"


namespace vmf_fitting
{

static constexpr float K_THRESHOLD = 1.e-4f;
// Picked such that exp(-2k)>0. Otherwise we get -inf when the log is applied!
static constexpr float K_THRESHOLD_MAX = 40.f; 


template<int N = 8>
struct VonMisesFischerMixture
{
  static constexpr int NUM_COMPONENTS = N;
  using WeightArray = Eigen::Array<float, NUM_COMPONENTS, 1>;
  using ConcArray = Eigen::Array<float, NUM_COMPONENTS, 1>;
  using MeansArray = Eigen::Array<float, NUM_COMPONENTS, 3>;

  MeansArray means;
  WeightArray weights; // Of components
  ConcArray concentrations;
};

float Pdf(const Eigen::Vector3f &mu, const float k, const Eigen::Vector3f &x) noexcept;
template<int N>
float Pdf(const VonMisesFischerMixture<N> &mixture, const Eigen::Vector3f &pos) noexcept;
template<int N>
Eigen::Vector3f Sample(const VonMisesFischerMixture<N> &mixture, std::array<double, 3> rs) noexcept;
template<int N>
void InitializeForUnitSphere(VonMisesFischerMixture<N> &mixture) noexcept;
// Note: the product is in general not normalized, i.e. not a probability density.
template<int N, int M>
VonMisesFischerMixture<N*M> Product(const VonMisesFischerMixture<N> &m1, const VonMisesFischerMixture<M> &m2) noexcept;
template<int N>
void Normalize(VonMisesFischerMixture<N> &mixture) noexcept;

template<int N_>
void ExpApproximation(Eigen::Array<float, N_, 1> &vals);
float ExpApproximation(float x);

inline float MeanCosineToConc(float r) noexcept
{
  static constexpr float THRESHOLD = 1.f - 1.e-6f;
  if (r >= THRESHOLD)
    r = THRESHOLD;
  // "Volume Path Guiding Based on Zero-Variance Random Walk Theory" Eq. 29 is wrong. There is r missing
  // in the denominator. Correct version is in "Directional Statistics in Machine Learning: a Brief Review", for instance.
  return r * (3.f - Sqr(r)) / (1.f - Sqr(r));
}

inline float ConcToMeanCos(float k) noexcept
{
  // "Volume Path Guiding Based on Zero-Variance Random Walk Theory" Sec A.2, pg 0:19.
  return 1.0f / std::tanh(k) - 1.0f/k;
}

namespace incremental
{

template<int N = 8>
struct Data
{
  static constexpr int NC = VonMisesFischerMixture<N>::NUM_COMPONENTS;
  // Don't have to be initialize because if data_count==0 then 
  // those vars are assigned to anyway. (Otherwise they're added to.)
  // Eigen::Array<float, NC, 3> avg_positions;
  // Eigen::Array<float, NC, 1> avg_responsibilities;
  // Eigen::Array<float, NC, 1> avg_responsibilities_unweighted;
  // std::uint64_t data_count = 0;
  // std::uint64_t data_count_weights = 0;
  // float avg_weights = 0.f;
  using ArrayNc3 = Eigen::Array<float, NC, 3>;
  using ArrayNc1 = Eigen::Array<float, NC, 1>;
  Accumulators::OnlineAverage<ArrayNc3> avg_positions{ArrayNc3{Eigen::zero}, 0l};
  Accumulators::OnlineAverage<ArrayNc1> avg_responsibilities{ArrayNc1{Eigen::zero}, 0l};
  Accumulators::OnlineAverage<ArrayNc1> avg_responsibilities_unweighted{ArrayNc1{Eigen::zero}, 0l};
  Accumulators::OnlineAverage<float> avg_weights{};
};

template<int N = 8>
struct Params
{
  // Can be interpreted as number of samples where the prior start to count less for the final result than the samples.
  float prior_nu = 100.0f; 
  float prior_tau = 100.0f; 
  float prior_alpha = 100.0f;
  int maximization_step_every = 10;
  VonMisesFischerMixture<N>* prior_mode = nullptr;
};

template<int N = 8>
void Fit(VonMisesFischerMixture<N> &mixture, Data<N> &fitdata, const Params<N> &params, Span<const Eigen::Vector3f> data, Span<const float> data_weights) noexcept;

// Large enough so that 1/eps is still finite ...
inline static constexpr float eps = 1.e-38f;

} // namespace incremental

} // namespace vmf_fitting