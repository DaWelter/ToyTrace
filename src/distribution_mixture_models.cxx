#include "distribution_mixture_models.hxx"

#include <boost/range/combine.hpp>
#include <iomanip>


namespace vmf_fitting
{

template<int N = 8>
inline Eigen::Array<float, N, 1> ComponentPdfs(const VonMisesFischerMixture<N> & mixture, const Eigen::Vector3f & pos) noexcept
{
  const auto& k = mixture.concentrations;
  assert((k >= K_THRESHOLD).all() && (k <= K_THRESHOLD_MAX).all());
  const auto prefactors = float(Pi)*2.f*(1.f - (-2.f*k).exp());
  const auto tmp = (k*((mixture.means.matrix() * pos).array() - 1.f)).exp();
  const auto result = (k / prefactors * tmp).eval();
  assert(result.isFinite().all() && (result >= 0.f).all());
  return result;
}


template<int N = 8>
float Pdf(const VonMisesFischerMixture<N> & mixture, const Eigen::Vector3f & pos) noexcept
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


template<int N = 8>
Eigen::Vector3f Sample(const VonMisesFischerMixture<N> & mixture, std::array<double, 3> rs) noexcept
{
  const int idx = TowerSampling<N>(mixture.weights.data(), (float)rs[0]);
  return Sample(mixture.means.row(idx).matrix(), mixture.concentrations[idx], (float)rs[1], (float)rs[2]);
}


template<int N>
void InitializeForUnitSphere(VonMisesFischerMixture<N> & mixture)  noexcept
{
  assert ("!Not implemented!");
}


template<>
void InitializeForUnitSphere<2>(VonMisesFischerMixture<2> & mixture)  noexcept
{
  using MoVMF = VonMisesFischerMixture<2>;
  mixture.means <<
    -1., 0., 0.,
     1., 0., 0.;
  mixture.concentrations = MoVMF::ConcArray::Constant(0.1f);
  mixture.weights = typename MoVMF::WeightArray(1.f / MoVMF::NUM_COMPONENTS);
}


template<>
void InitializeForUnitSphere<8>(VonMisesFischerMixture<8> & mixture)  noexcept
{
  using MoVMF = VonMisesFischerMixture<8>;
    mixture.means <<
      0.66778004,0.73539025,0.11519922,
      -0.9836298,-0.008926501,0.1799797,
      -0.14629413,-0.7809748,0.60718733,
      0.49043104,0.04368747,-0.87038434,
      -0.10088497,0.42765132,0.8982965,
      -0.35999632,-0.6945797,-0.62286574,
      -0.43178025,0.7588791,-0.48751223,
      0.860208,-0.47938251,0.17388113;
  mixture.concentrations = MoVMF::ConcArray::Constant(2.f);
  mixture.weights = typename MoVMF::WeightArray(1.f / MoVMF::NUM_COMPONENTS);
}

template<>
void InitializeForUnitSphere<16>(VonMisesFischerMixture<16> & mixture)  noexcept
{
  using MoVMF = VonMisesFischerMixture<16>;
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
    mixture.concentrations = MoVMF::ConcArray::Constant(5.f);  
    mixture.weights = typename MoVMF::WeightArray(1.f / MoVMF::NUM_COMPONENTS);
}


template<int N, int M>
VonMisesFischerMixture<N*M> Product(const VonMisesFischerMixture<N> &m1, const VonMisesFischerMixture<M> &m2) noexcept
{
  static constexpr int NM = N*M;
  VonMisesFischerMixture<NM> result;
  for (int i=0; i<N; ++i)
  {
    result.means.block(i*M, 0, M, 3) = (m1.concentrations[i]*m1.means.row(i)).replicate(M,1) + (m2.means*m2.concentrations.replicate(1,3));
  }
  result.concentrations = result.means.matrix().rowwise().norm();
  result.means.colwise() /= result.concentrations;

  // Weights ...
  const typename VonMisesFischerMixture<N>::WeightArray  exponentials1 = 1.f+incremental::eps - (-2.f*m1.concentrations    ).exp();
  const typename VonMisesFischerMixture<M>::WeightArray  exponentials2 = 1.f+incremental::eps - (-2.f*m2.concentrations    ).exp();
  const typename VonMisesFischerMixture<NM>::WeightArray exponentialsk = 1.f+incremental::eps - (-2.f*result.concentrations).exp();

  for (int i=0; i<N; ++i)
  {
    for (int j=0; j<M; ++j)
    {
      const int k = i*M + j;
      result.weights[k] = m1.concentrations[i]*m2.concentrations[j]*exponentialsk[k] / 
                          (2.f*PiFloat*result.concentrations[k]*exponentials1[i]*exponentials2[j] + incremental::eps);
      result.weights[k] *= std::exp(m1.concentrations[i]*(m1.means.row(i).matrix().dot(result.means.row(k).matrix())-1.f) + 
                                    m2.concentrations[j]*(m2.means.row(j).matrix().dot(result.means.row(k).matrix())-1.f));
      result.weights[k] *= m1.weights[i] * m2.weights[j];
    }
  }

  // Hack to avoid numerical problems ...
  result.concentrations = result.concentrations.max(K_THRESHOLD).min(K_THRESHOLD_MAX).eval();
  return result;
}


template<int N>
void Normalize(VonMisesFischerMixture<N> &mixture) noexcept
{
  const float wsum = mixture.weights.sum();
  if (unlikely(wsum <= 0.f))
  {
    // Case may happen due to underflow in calculations with exponentials
    mixture.weights.setConstant(1.f/N);
  }
  else
  {
    mixture.weights /= wsum;
  }
}


namespace incremental
{

template<int N>
void UpdateStatistics(VonMisesFischerMixture<N> &mixture, Data<N> &dta, const Params<N> &params, const Eigen::Vector3f &x, float weight) noexcept;

template<int N>
void MaximizationStep(VonMisesFischerMixture<N> &mixture, const Data<N> &dta, const Params<N> &params) noexcept;

template<int N>
void Fit(VonMisesFischerMixture<N> &mixture, Data<N> &dta, const Params<N> &params, Span<const Eigen::Vector3f> data, Span<const float> data_weights) noexcept
{
  assert(data.size() == data_weights.size());
  assert(params.prior_mode != nullptr);
  
  for (int i=0; i<data.size(); ++i)
  {
    UpdateStatistics(mixture, dta, params, data[i], data_weights[i]);
    if (dta.data_count % params.maximization_step_every == 0)
    {
      MaximizationStep(mixture, dta, params);

      //dta.data_count = 0;
    }
  }
}


template<int N>
void UpdateStatistics(VonMisesFischerMixture<N> & mixture, Data<N> &fitdata, const Params<N> &params, const Eigen::Vector3f & x, float weight) noexcept
{
  Eigen::Array<float, N, 1> responsibilities = mixture.weights * ComponentPdfs(mixture, x) + eps;
  responsibilities /= responsibilities.sum();

  if (fitdata.data_count_weights == 0)
    fitdata.avg_weights = weight;
  else
    fitdata.avg_weights = Lerp(fitdata.avg_weights, weight, (float)std::pow(double(fitdata.data_count_weights), -0.75));

  if (fitdata.data_count == 0)
  {
    fitdata.avg_responsibilities_unweighted = responsibilities;
    fitdata.avg_responsibilities = weight * responsibilities;
    fitdata.avg_positions = weight*(responsibilities.matrix()*x.transpose()).array();
  }
  else
  {
    // How much to "trust" new data.
    const float mix_factor = (float)std::pow(double(fitdata.data_count), -0.75f);
    fitdata.avg_responsibilities_unweighted += mix_factor*(responsibilities        - fitdata.avg_responsibilities_unweighted);
    fitdata.avg_responsibilities            += mix_factor*(weight*responsibilities - fitdata.avg_responsibilities);
    fitdata.avg_positions                   += mix_factor*(weight*(responsibilities.matrix()*x.transpose()).array() - fitdata.avg_positions);
  }
  ++fitdata.data_count;
  ++fitdata.data_count_weights;
} 


template<int N>
void MaximizationStep(VonMisesFischerMixture<N> & mixture, const Data<N> &dta, const Params<N> &params) noexcept
{

  std::uint64_t unique_data_count = dta.data_count;
  for (int k = 0; k < VonMisesFischerMixture<N>::NUM_COMPONENTS; ++k)
  {
    const float scaled_responsibilities = dta.avg_responsibilities[k] / (dta.avg_weights + eps);

    // The eps is there in case both of the former terms evaluate to zero!
    mixture.weights[k] = (params.prior_nu-1)/unique_data_count*params.prior_mode->weights[k] + scaled_responsibilities + eps;

    { // Posterior of mean.
      const float diminished_prior_factor = dta.avg_weights * params.prior_tau / unique_data_count;
      mixture.means.row(k) = (diminished_prior_factor*params.prior_mode->means.row(k) + dta.avg_positions.row(k)) /
        (diminished_prior_factor + dta.avg_responsibilities[k] + eps);
      float norm = mixture.means.row(k).matrix().norm();
      if (norm > eps)
        mixture.means.row(k) /= norm;
      else
        mixture.means.row(k).matrix() = Eigen::Vector3f{1.f, 0.f, 0.f};
    }

    {
      #if 1
      // Note: Don't use avg_responsibilities_unweighted[k]*avg_weights here. It computes the average position wrong
      // in a way that mean_cosine is way overestimated.
      const float mean_cosine = (1.f/(dta.avg_responsibilities[k] + eps)) * dta.avg_positions.row(k).matrix().norm();
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
      const float mean_cosine = (1.f/(dta.avg_responsibilities[k] + eps)) * dta.avg_positions.row(k).matrix().norm();
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


} // namespace incremental

} // namespace vmf_fitting


#define INSTANTIATE_VonMisesFischerMixture(n) \
  template void vmf_fitting::incremental::Fit<n>(VonMisesFischerMixture<n> &mixture, Data<n> &fitdata, const Params<n> &params, Span<const Eigen::Vector3f> data, Span<const float> data_weights) noexcept; \
  template float  vmf_fitting::Pdf<n>(const VonMisesFischerMixture<n> &mixture, const Eigen::Vector3f &pos) noexcept; \
  template Eigen::Vector3f  vmf_fitting::Sample<n>(const VonMisesFischerMixture<n> &mixture, std::array<double, 3> rs) noexcept; \
  template void  vmf_fitting::InitializeForUnitSphere<n>(VonMisesFischerMixture<n> &mixture) noexcept; \
  template void vmf_fitting::Normalize(VonMisesFischerMixture<n> &mixture) noexcept;

INSTANTIATE_VonMisesFischerMixture(2)
INSTANTIATE_VonMisesFischerMixture(8)
INSTANTIATE_VonMisesFischerMixture(16)

template vmf_fitting::VonMisesFischerMixture<16> vmf_fitting::Product(const vmf_fitting::VonMisesFischerMixture<2> &m1, const vmf_fitting::VonMisesFischerMixture<8> &m2) noexcept;