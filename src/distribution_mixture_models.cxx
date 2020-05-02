#include "distribution_mixture_models.hxx"

#include <boost/range/combine.hpp>
#include <iomanip>

namespace gmm_fitting
{


namespace normaldistribution
{

float Pdf(const Float2 &x, const Float2 &mu, const Eigen::Matrix2f &prec) noexcept
{
  // Prefactor is only valid for 2D!
  static constexpr float prefactor = float(1. / (Pi*2.));
  const float det = prec.determinant();
  assert(det > 0);
  const auto tmp = (x - mu).dot(prec * (x - mu));
  return prefactor * std::sqrt(det)*std::exp(-0.5f*tmp);
}

}; // namespace normaldistribution


float Pdf(const GaussianMixture2d &mixture, const Float2 &pos) noexcept
{
  float result = 0.f;
  for (int k = 0; k < GaussianMixture2d::NUM_COMPONENTS; ++k)
  {
    result += mixture.weights[k] * normaldistribution::Pdf(pos, mixture.means[k], mixture.precisions[k]);
  }
  return result;
}


Float2 Sample(const GaussianMixture2d &mixture, std::array<double, 3> rs) noexcept
{
  // How to? Cholesky decomposition!
  // https://stackoverflow.com/questions/16706226/how-do-i-draw-samples-from-multivariate-gaussian-distribution-parameterized-by-p

  const int idx = TowerSampling<GaussianMixture2d::NUM_COMPONENTS>(mixture.weights.data(), (float)rs[0]);

  Float2 x = SampleTrafo::ToNormal2d({ (float)rs[1],(float)rs[2] });

  // Scale to match the covariance
  const auto L = mixture.precisions[idx].llt().matrixL().transpose();
  L.solveInPlace(x);

  //x = (mixture.precisions[idx].inverse().llt().matrixL() * x).eval();

  x += mixture.means[idx];

  return x;
}




void Fit(GaussianMixture2d &mixture, Span<const Eigen::Vector2f> data, Span<const float> data_weights, const FitParams &params)
{
  // [1] state that nu>0, but this must be a mistake, since from eq (21) seems clear that for nu<1, the mixture weights can become negative.
  assert(params.prior_nu > 1.);
  // Again [1] state that alpha>dim-1. But I found it does not work because it allows the covariance matrices to become singular.
  // I don't understand why. There is only a hint in https://en.wikipedia.org/wiki/Wishart_distribution, stating 
  // "In fact the above definition can be extended to any real n > p − 1. If n ≤ p − 1, then the Wishart no longer has a 
  // density—instead it represents a singular distribution that takes values in a lower-dimension subspace of the space of p × p matrices."
  // I probably want a proper density. So I require alpha>dim.
  assert(params.prior_alpha > 2.);

  constexpr auto NC = GaussianMixture2d::NUM_COMPONENTS;
  Eigen::Array<double, Eigen::Dynamic, NC> responsibilities(data.size(), 8);

  auto FillResponsibilities = [&]()
  {
    for (int i = 0; i < data.size(); ++i)
    {
      double sum = 0.f;
      for (int k = 0; k < NC; ++k)
      {
        // Maybe multiply precision by data_weights as inspired by [2]?
        responsibilities(i, k) = mixture.weights[k] * normaldistribution::Pdf(data[i], mixture.means[k], mixture.precisions[k]);
        sum += responsibilities(i, k);
      }
      responsibilities.row(i) /= sum;
    }
  };

  std::array<double, NC> responsibility_sums;
  std::array<Double2, NC> means;
  std::array<Eigen::Matrix2d, NC> mean_outer_products;

  auto UpdateSufficientStatistics = [&]()
  {
    // Computation of the statistics in this way is inspired by [3].
    // However, [3] does not give formulae for batch updates with weights.
    // The way to do the weighting is therefor from [2], eqs. (8-10).
    std::fill(responsibility_sums.begin(), responsibility_sums.end(), 0.);
    std::fill(means.begin(), means.end(), Double2::Zero());
    std::fill(mean_outer_products.begin(), mean_outer_products.end(), Eigen::Matrix2d::Zero());

    for (int i = 0; i < data.size(); ++i)
    {
      Eigen::Matrix2d x_xt = (data[i] * data[i].transpose()).cast<double>();
      Double2 x = data[i].cast<double>();
      for (int k = 0; k < NC; ++k)
      {
        responsibility_sums[k] += data_weights[i] * responsibilities(i, k);
        means[k] += data_weights[i] * responsibilities(i, k)*x;
        mean_outer_products[k] += data_weights[i] * responsibilities(i, k)*x_xt;
      }
    }

    for (int k = 0; k < NC; ++k)
    {
      means[k] = means[k] / responsibility_sums[k];
      mean_outer_products[k] = mean_outer_products[k] / responsibility_sums[k];
    }
  };

  auto MaximizationStep = [&]()
  {
    for (int k = 0; k < NC; ++k)
    {
      mixture.weights[k] = static_cast<float>(params.prior_nu - 1 + responsibility_sums[k]);
      mixture.means[k] = means[k].cast<float>();
      Eigen::Matrix2d cov = mean_outer_products[k] - means[k] * means[k].transpose();
      // Use this if using a prior which makes mixture.means[k] different from means[k]:
      //Eigen::Matrix2d cov = m*m.transpose() - m*means[k].transpose() - means[k]*m.transpose() + (outer_products[k] / responsibility_sums[k]);
      cov = (params.prior_u*Eigen::Matrix2d::Identity() + responsibility_sums[k] * cov) / (params.prior_alpha - 2/*=dim*/ + responsibility_sums[k]);
      assert(cov.determinant() > 0);
      mixture.precisions[k] = cov.inverse().cast<float>();
    }
    mixture.weights /= mixture.weights.sum();
  };

  for (int iter = 0; iter < params.max_iters; ++iter)
  {
    FillResponsibilities();
    UpdateSufficientStatistics();
    MaximizationStep();
  }
}


namespace incremental
{

struct FitImpl : public Data
{
  void UpdateStatistics(GaussianMixture2d &mixture, const Params &params, const Eigen::Vector2f &x, float weight);
  void MaximizationStep(GaussianMixture2d &mixture, const Params &params);
};

void FixCov(Eigen::Matrix2f &cov);



void Fit(GaussianMixture2d &mixture, Data &fitdata, const Params &params, Span<const Eigen::Vector2f> data, Span<const float> data_weights)
{
  assert(data.size() == data_weights.size());
  assert(params.prior_mode != nullptr);
  
  for (int i=0; i<data.size(); ++i)
  {
    static_cast<FitImpl&>(fitdata).UpdateStatistics(mixture, params, data[i], data_weights[i]);
    if (fitdata.data_count % params.maximization_step_every == 0)
    {
      static_cast<FitImpl&>(fitdata).MaximizationStep(mixture, params);
    }
  }
}



void FitImpl::UpdateStatistics(GaussianMixture2d & mixture, const Params &params, const Eigen::Vector2f & x, float weight)
{
  const Eigen::Matrix2f x_xt = x * x.transpose();

  Eigen::Array<float, NC, 1> responsibilities;
  for (int k = 0; k < NC; ++k)
  {
    // Maybe multiply precision by data_weights as inspired by [2]?
    responsibilities(k) = mixture.weights[k] * normaldistribution::Pdf(x, mixture.means[k], mixture.precisions[k]) + eps;
    assert(std::isfinite(responsibilities(k)) && responsibilities(k) >= 0.f);
  }
  assert(responsibilities.sum() > 0.f);
  responsibilities /= responsibilities.sum();

  if (data_count == 0)
  {
    for (int k = 0; k < NC; ++k)
    {
      avg_responsibilities[k] = weight*responsibilities(k);
      avg_positions[k] = weight * responsibilities(k)*x;
      avg_outer_products[k] = weight * responsibilities(k)*x_xt;
    }
    avg_weights = weight;
  }
  else
  {
    // How much to "trust" new data.
    const float mix_factor = (float)std::pow(double(data_count), -0.75f);
    for (int k = 0; k < NC; ++k)
    {
      avg_responsibilities[k] = Lerp(avg_responsibilities[k], weight*responsibilities(k), mix_factor);
      avg_positions[k] = Lerp(avg_positions[k], Float2(weight*responsibilities(k)*x), mix_factor);
      avg_outer_products[k] = Lerp(avg_outer_products[k], Eigen::Matrix2f(weight*responsibilities(k)*x_xt), mix_factor);
    }
    avg_weights = Lerp(avg_weights, weight, mix_factor);
  }
  ++data_count;
} 


void FitImpl::MaximizationStep(GaussianMixture2d & mixture, const Params &params)
{

  std::uint64_t unique_data_count = data_count;
  for (int k = 0; k < NC; ++k)
  {
    const float scaled_responsibilities = avg_responsibilities[k] / (avg_weights + eps);

    // The eps is there in case both of the former terms evaluate to zero!
    mixture.weights[k] = (params.prior_nu-1)/unique_data_count*params.prior_mode->weights[k] + scaled_responsibilities + eps;

    { // Posterior of mean.
      const float diminished_prior_factor = avg_weights * params.prior_tau / unique_data_count;
      mixture.means[k] = (diminished_prior_factor*params.prior_mode->means[k] + avg_positions[k]) /
        (diminished_prior_factor + avg_responsibilities[k] + eps);
    }

    { // Posterior of precision matrix.

      // This refers to the the posterior mean 'm', and the "average" 'x'.
      // I put average in quotation marks because it is the average of "x*weight*responsibilities".
      // In order to obtain a useful mean, this average needs to be scaled appropriately.
      const Eigen::Matrix2f mmt = (mixture.means[k] * mixture.means[k].transpose());
      const Eigen::Matrix2f mxt = (mixture.means[k] * avg_positions[k].transpose());
      const Eigen::Matrix2f xmt = (avg_positions[k] * mixture.means[k].transpose());

      const Eigen::Matrix2f cov = (1.f / (avg_weights + eps))*
        (avg_outer_products[k] - xmt - mxt + avg_responsibilities[k]*mmt);

      const auto delta = params.prior_mode->means[k] - mixture.means[k];
      const Eigen::Matrix2f cov_mean_prior = (params.prior_tau/unique_data_count) * (delta*delta.transpose());

      const auto cov_prior = (params.prior_alpha / unique_data_count)*params.prior_mode->precisions[k].inverse();

      Eigen::Matrix2f post_cov = (cov_prior + cov_mean_prior + cov) /
        ((params.prior_alpha / unique_data_count) + scaled_responsibilities + eps);
      // Should post_cov become 0 at this point, then the following function must come to rescue.
      FixCov(post_cov);
      assert(post_cov.array().isFinite().all());

      // std::cout << "component " << k << " stats: \n" << avg_positions[k].transpose() << "\n" << avg_responsibilities[k] << std::endl;                
      // std::cout << "mix: " << mixture.weights[k] << "\n" << mixture.means[k].transpose() << std::endl;
      // std::cout << "cov: " << post_cov << " det = " << post_cov.determinant() << std::endl;

      mixture.precisions[k] = post_cov.inverse().cast<float>();
    }
  }
  assert(mixture.weights.sum() > 0.f);
  mixture.weights /= mixture.weights.sum();
}


void FixCov(Eigen::Matrix2f &cov)
{
  // Yeah this is stupid ... should be improved eventually ...
  float det = cov.determinant();
  if (det < 1.e-6)
  {
    //std::cout << "Fix Matrix: old: " << cov << " which has det " << det << std::endl;
    while (det < 1.e-6)
    {
      cov = Lerp(cov, Eigen::Matrix2f::Identity().eval(), 0.001f);
      det = cov.determinant();
    }
    //std::cout << "Fix Matrix: new: " << cov << " which has det " << det << std::endl;
  }
}


} // namespace incremental


GaussianMixture2d::MeansArray MeansPriorForUnitDisc()
{
  return GaussianMixture2d::MeansArray {{
    {-0.02081824f, -0.00203204f},
    { 0.4037143f , -0.5633242f },
    {-0.60468113f,  0.32800356f},
    {-0.12180968f,  0.6798401f },
    { 0.6943762f , -0.03451705f},
    { 0.4580511f ,  0.52144015f},
    {-0.63349193f, -0.26706135f},
    {-0.18766472f, -0.66355205f}
  }};
}


void InitializeForUnitDisc(GaussianMixture2d &mixture)
{
  mixture.means = MeansPriorForUnitDisc();
  mixture.weights = GaussianMixture2d::WeightArray::Constant(1.f/GaussianMixture2d::NUM_COMPONENTS);
  std::fill(mixture.precisions.begin(), mixture.precisions.end(),
    (10.f*Eigen::Matrix2f::Identity()).eval());
}


} // namespace gmm_fitting


namespace vmf_fitting
{


inline Eigen::Array<float, VonMisesFischerMixture::NUM_COMPONENTS, 1> ComponentPdfs(const VonMisesFischerMixture & mixture, const Eigen::Vector3f & pos) noexcept
{
  const auto& k = mixture.concentrations;
  assert((k >= K_THRESHOLD).all() && (k <= K_THRESHOLD_MAX).all());
  const auto prefactors = float(Pi)*2.f*(1.f - (-2.f*k).exp());
  const auto tmp = (k*((mixture.means.matrix() * pos).array() - 1.f)).exp();
  return (k / prefactors * tmp).eval();
}


float Pdf(const VonMisesFischerMixture & mixture, const Eigen::Vector3f & pos) noexcept
{
  const auto component_pdfs = ComponentPdfs(mixture, pos);
  return (component_pdfs * mixture.weights).sum();
}


Eigen::Vector3f Sample(const Eigen::Vector3f& mu, float k, float r1, float r2) noexcept
{
  assert (k >= K_THRESHOLD);
  assert (k <= K_THRESHOLD_MAX);
  const float vx = std::cos((float)(Pi*2.)*r1);
  const float vy = std::sin((float)(Pi*2.)*r1);
  // if (!std::isfinite(vx) || !std::isfinite(vy))
  //   std::cerr << "Oh noz vx or vy are non-finite! " << vx << ", " << vy << std::endl;
  const float w = 1.f + std::log(r2 + (1.f - r2)*std::exp(-2.f*k)) / k;
  // if (!std::isfinite(w))
  //   std::cerr << "Oh noz w is not finite! " << w << std::endl;
  const float tmp = 1.f - w * w;
  // if (tmp < 0.f)
  //   std::cerr << "Oh noz tmp is negative! " << std::setprecision(std::numeric_limits<float>::digits10 + 1) << tmp << std::endl;
  const float rho = std::sqrt(std::max(0.f, tmp));
  // if (!std::isfinite(rho))
  //   std::cerr << "Oh noz rho is not finite! " << rho << ", " << w << std::endl;
  Eigen::Vector3f x{
    rho*vx, rho*vy, w
  };
  Eigen::Matrix3f frame = OrthogonalSystemZAligned(mu);
  // if (!frame.array().isFinite().all())
  //   std::cerr << "Oh noz the frame is not finite! " << frame << ", mu = " << mu << std::endl;
  return frame * x;
}


Eigen::Vector3f Sample(const VonMisesFischerMixture & mixture, std::array<double, 3> rs) noexcept
{
  const int idx = TowerSampling<VonMisesFischerMixture::NUM_COMPONENTS>(mixture.weights.data(), (float)rs[0]);
  return Sample(mixture.means.row(idx).matrix(), mixture.concentrations[idx], (float)rs[1], (float)rs[2]);
}


void InitializeForUnitSphere(VonMisesFischerMixture & mixture)  noexcept
{
  mixture.means <<
    0.49468744,-0.1440351,-0.8570521,
    0.36735174,-0.8573688,-0.36051556,
    -0.40503788,-0.48854542,-0.772831,
    -0.32770208,-0.12784185,0.9360917,
    -0.46322507,-0.8861626,-0.011769729,
    -0.933043,0.16280192,-0.32082137,
    -0.4759586,0.8746706,-0.09173246,
    -0.2931369,0.36415973,-0.8840013,
    -0.9006278,-0.24284472,0.360411,
    -0.32578856,0.650924,0.685682,
    0.5459307,0.11810207,0.8294646,
    0.9575066,0.22513846,-0.18026012,
    0.84576833,-0.47598344,0.24107178,
    0.47158864,0.8217275,0.31995004,
    0.12246728,-0.77039504,0.6256942,
    0.34670526,0.77106386,-0.5340936
  ;
  mixture.concentrations = VonMisesFischerMixture::ConcArray::Constant(5.f);
  mixture.weights = VonMisesFischerMixture::WeightArray(1.f / VonMisesFischerMixture::NUM_COMPONENTS);
}


namespace incremental
{

struct FitImpl : public Data
{
  void UpdateStatistics(VonMisesFischerMixture &mixture, const Params &params, const Eigen::Vector3f &x, float weight) noexcept;
  void MaximizationStep(VonMisesFischerMixture &mixture, const Params &params) noexcept;
  static float MeanCosineToConc(float r) noexcept;
  static float ConcToMeanCos(float r) noexcept;
};


void Fit(VonMisesFischerMixture &mixture, Data &fitdata, const Params &params, Span<const Eigen::Vector3f> data, Span<const float> data_weights) noexcept
{
  assert(data.size() == data_weights.size());
  assert(params.prior_mode != nullptr);
  
  for (int i=0; i<data.size(); ++i)
  {
    static_cast<FitImpl&>(fitdata).UpdateStatistics(mixture, params, data[i], data_weights[i]);
    if (fitdata.data_count % params.maximization_step_every == 0)
    {
      static_cast<FitImpl&>(fitdata).MaximizationStep(mixture, params);
    }
  }
}



void FitImpl::UpdateStatistics(VonMisesFischerMixture & mixture, const Params &params, const Eigen::Vector3f & x, float weight) noexcept
{
  Eigen::Array<float, NC, 1> responsibilities = mixture.weights * ComponentPdfs(mixture, x) + eps;
  assert(responsibilities.isFinite().all() && (responsibilities > 0.f).all());
  responsibilities /= responsibilities.sum();

  if (data_count == 0)
  {
    avg_responsibilities_unweighted = responsibilities;
    avg_responsibilities = weight * responsibilities;
    avg_positions = weight*(responsibilities.matrix()*x.transpose()).array();
    avg_weights = weight;
  }
  else
  {
    // How much to "trust" new data.
    const float mix_factor = (float)std::pow(double(data_count), -0.75f);
    avg_responsibilities_unweighted += mix_factor*(responsibilities        - avg_responsibilities_unweighted);
    avg_responsibilities            += mix_factor*(weight*responsibilities - avg_responsibilities);
    avg_positions                   += mix_factor*(weight*(responsibilities.matrix()*x.transpose()).array() - avg_positions);
    avg_weights = Lerp(avg_weights, weight, mix_factor);
  }
  ++data_count;
} 


void FitImpl::MaximizationStep(VonMisesFischerMixture & mixture, const Params &params) noexcept
{

  std::uint64_t unique_data_count = data_count;
  for (int k = 0; k < NC; ++k)
  {
    const float scaled_responsibilities = avg_responsibilities[k] / (avg_weights + eps);

    // The eps is there in case both of the former terms evaluate to zero!
    mixture.weights[k] = (params.prior_nu-1)/unique_data_count*params.prior_mode->weights[k] + scaled_responsibilities + eps;

    { // Posterior of mean.
      const float diminished_prior_factor = avg_weights * params.prior_tau / unique_data_count;
      mixture.means.row(k) = (diminished_prior_factor*params.prior_mode->means.row(k) + avg_positions.row(k)) /
        (diminished_prior_factor + avg_responsibilities[k] + eps);
      float norm = mixture.means.row(k).matrix().norm();
      if (norm > eps)
        mixture.means.row(k) *= (1.f/norm);
      else
        mixture.means.row(k).matrix() = Eigen::Vector3f{1.f, 0.f, 0.f};
    }

    {
      // Note: Don't use avg_responsibilities_unweighted[k]*avg_weights here. It computes the average position wrong
      // in a way that mean_cosine is way overestimated.
      const float mean_cosine = (1.f/(avg_responsibilities[k] + eps)) * avg_positions.row(k).matrix().norm();
      const float conc_estimate = MeanCosineToConc(mean_cosine);
      const float diminished_alpha = params.prior_alpha/unique_data_count;
      const float post_conc = 
        (params.prior_mode->concentrations[k]*diminished_alpha + conc_estimate) / (diminished_alpha + 1.f);
      mixture.concentrations[k] = post_conc;

      #if 0
      // This does not do percievably better than the simpler code above!!
      // Here the prior is on the mean cosine, forcing me to go back and forth with the cosine-k-conversions.
      const float prior_cos = ConcToMeanCos(params.prior_mode->concentrations[k]);
      const float mean_cosine = (1.f/(avg_responsibilities[k] + eps)) * avg_positions.row(k).matrix().norm();
      const float diminished_alpha = params.prior_alpha/unique_data_count;
      const float mean_cosine_post = (diminished_alpha*prior_cos + mean_cosine) / (diminished_alpha + 1.f);
      mixture.concentrations[k] = MeanCosineToConc(mean_cosine_post);
      #endif
    }
  }

  mixture.concentrations = mixture.concentrations.max(K_THRESHOLD).min(K_THRESHOLD_MAX).eval();

  assert(mixture.weights.sum() > 0.f);
  mixture.weights /= mixture.weights.sum();
}

inline float FitImpl::MeanCosineToConc(float r) noexcept
{
  static constexpr float THRESHOLD = 1.f - 1.e-6f;
  if (r >= THRESHOLD)
    r = THRESHOLD;
  // "Volume Path Guiding Based on Zero-Variance Random Walk Theory" Eq. 29 is wrong. There is r missing
  // in the denominator. Correct version is in "Directional Statistics in Machine Learning: a Brief Review", for instance.
  return r * (3.f - Sqr(r)) / (1.f - Sqr(r));
}

inline float FitImpl::ConcToMeanCos(float k) noexcept
{
  // "Volume Path Guiding Based on Zero-Variance Random Walk Theory" Sec A.2, pg 0:19.
  return 1.0f / std::tanh(k) - 1.0f/k;
}


float GetAverageWeight(const Data &fitdata)
{
  return fitdata.avg_weights;
}


} // namespace incremental


} // namespace vmf_fitting