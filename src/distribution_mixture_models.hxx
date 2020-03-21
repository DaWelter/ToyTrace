#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "span.hxx"
#include "sampler.hxx"


namespace gmm_fitting
{


struct GaussianMixture2d
{
  static constexpr int NUM_COMPONENTS = 8;
  using WeightArray = Eigen::Array<float, NUM_COMPONENTS, 1>;
  using CovsArray = std::array<Eigen::Matrix2f, NUM_COMPONENTS>;
  using MeansArray = std::array<Eigen::Vector2f, NUM_COMPONENTS>;

  WeightArray weights; // Of components
  CovsArray precisions; // Inverse of covariance matrix
  MeansArray means;
};


float Pdf(const GaussianMixture2d &mixture, const Float2 &pos) noexcept;
Float2 Sample(const GaussianMixture2d &mixture, std::array<double, 3> rs) noexcept;


// Refs:
// [1] Gauvain & Lee (1994) "Maximum a Posteriori Estimate for Multivariate Gaussian Mixture Observations of Markov Chains."
// [2] Gebru et al. (2016) "EM Algorithm for Weighted-Data Clustering with Application to Audio-Visual Scene Analysis"
// [3] Vorba et al. (2014) "On-line learning of parametric mixture models for light transport simulation"

struct FitParams
{
  // As defined in [1]
  float prior_nu = 1.0f; // mixture weight exponent
  //float prior_tau = 1.0; // scaling factor of means
  float prior_alpha = 2.0f; // cov scaling
  float prior_u = 1.0f; // cov value
  int max_iters = 100;
  float convergence_tol = 1.e-6f;
};

void Fit(GaussianMixture2d &mixture, Span<const Eigen::Vector2f> data, Span<const float> data_weights, const FitParams &params);

namespace incremental
{

struct Data
{
  static constexpr int NC = GaussianMixture2d::NUM_COMPONENTS;
  std::array<float, NC> avg_responsibilities;
  std::array<Float2, NC> avg_positions;
  std::array<Eigen::Matrix2f, NC> avg_outer_products;  
  std::uint64_t data_count = 0;
  float avg_weights = 0.f;
};


struct Params
{
  // Can be interpreted as number of samples where the prior start to count less for the final result than the samples.
  float prior_nu = 1.0f; // mixture weight exponent.
  float prior_tau = 1.0f; // strength of means prior
  float prior_alpha = 2.0f; // cov determinant exponent.
  int maximization_step_every = 10;
  GaussianMixture2d* prior_mode = nullptr;
};


void Fit(GaussianMixture2d &mixture, Data &fitdata, const Params &params, Span<const Eigen::Vector2f> data, Span<const float> data_weights);

// Large enough so that 1/eps is still finite ...
inline static constexpr float eps = 1.e-38f;

} // namespace incremental


GaussianMixture2d::MeansArray MeansPriorForUnitDisc();
void InitializeForUnitDisc(GaussianMixture2d &mixture);

} // namespace gmm_fitting


namespace vmf_fitting
{

static constexpr float K_THRESHOLD = 1.e-4f;
// Picked such that exp(-2k)>0. Otherwise we get -inf when the log is applied!
static constexpr float K_THRESHOLD_MAX = 4.e1f; 

// 320 Bytes
struct VonMisesFischerMixture
{
  static constexpr int NUM_COMPONENTS = 16;
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
float GetAverageWeight(const Data &fitdata);

// Large enough so that 1/eps is still finite ...
inline static constexpr float eps = 1.e-38f;

} // namespace incremental

};