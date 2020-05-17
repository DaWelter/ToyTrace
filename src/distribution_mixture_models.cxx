#include "distribution_mixture_models.hxx"

#include <boost/range/combine.hpp>
#include <iomanip>


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
  if constexpr(VonMisesFischerMixture::NUM_COMPONENTS == 16)
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
    0.34670526,0.77106386,-0.5340936;
    mixture.concentrations = VonMisesFischerMixture::ConcArray::Constant(5.f);
  }
  else if constexpr(VonMisesFischerMixture::NUM_COMPONENTS == 8)
  {
    mixture.means <<
      0.66778004,0.73539025,0.11519922,
      -0.9836298,-0.008926501,0.1799797,
      -0.14629413,-0.7809748,0.60718733,
      0.49043104,0.04368747,-0.87038434,
      -0.10088497,0.42765132,0.8982965,
      -0.35999632,-0.6945797,-0.62286574,
      -0.43178025,0.7588791,-0.48751223,
      0.860208,-0.47938251,0.17388113;
    mixture.concentrations = VonMisesFischerMixture::ConcArray::Constant(2.f);
  }
  else 
    assert(!"VonMisesFischerMixture can only have 8 or 16 components");
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


  if (data_count_weights == 0)
    avg_weights = weight;
  else
    avg_weights = Lerp(avg_weights, weight, (float)std::pow(double(data_count_weights), -0.75));

  if (data_count == 0)
  {
    avg_responsibilities_unweighted = responsibilities;
    avg_responsibilities = weight * responsibilities;
    avg_positions = weight*(responsibilities.matrix()*x.transpose()).array();
  }
  else
  {
    // How much to "trust" new data.
    const float mix_factor = (float)std::pow(double(data_count), -0.75f);
    avg_responsibilities_unweighted += mix_factor*(responsibilities        - avg_responsibilities_unweighted);
    avg_responsibilities            += mix_factor*(weight*responsibilities - avg_responsibilities);
    avg_positions                   += mix_factor*(weight*(responsibilities.matrix()*x.transpose()).array() - avg_positions);
  }
  ++data_count;
  ++data_count_weights;
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
      #if 1
      // Note: Don't use avg_responsibilities_unweighted[k]*avg_weights here. It computes the average position wrong
      // in a way that mean_cosine is way overestimated.
      const float mean_cosine = (1.f/(avg_responsibilities[k] + eps)) * avg_positions.row(k).matrix().norm();
      const float conc_estimate = MeanCosineToConc(mean_cosine);
      const float diminished_alpha = params.prior_alpha/unique_data_count;
      const float post_conc = 
        (params.prior_mode->concentrations[k]*diminished_alpha + conc_estimate) / (diminished_alpha + 1.f);
      mixture.concentrations[k] = post_conc;
      #endif

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

  this->data_count = 0;

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

} // namespace incremental


} // namespace vmf_fitting