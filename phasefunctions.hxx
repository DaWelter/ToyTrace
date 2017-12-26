#pragma once
// DO NOT INCLUDE DIRECTLY.

class Sampler;

void WeightsToCombinationProbabilities(Spectral3 &prob_lambda, Spectral3 *prob_constituent_given_lambda, int num_constituents); // forward decl. TODO find place for this.

namespace PhaseFunctions
{
  
class PhaseFunction
{
  public:
    virtual ScatterSample SampleDirection(const Double3 &reverse_incident_dir, Sampler &sampler) const = 0;
    virtual Spectral3 Evaluate(const Double3 &reverse_indcident_dir, const Double3 &out_direction, double *pdf) const = 0;
};


class Uniform : public PhaseFunction
{
  public:
    ScatterSample SampleDirection(const Double3 &reverse_incident_dir, Sampler &sampler) const override;
    Spectral3 Evaluate(const Double3 &reverse_indcident_dir, const Double3 &out_direction, double *pdf) const override;
};


/* see http://glossary.ametsoc.org/wiki/Rayleigh_phase_function */
class Rayleigh : public PhaseFunction
{
  public:
    ScatterSample SampleDirection(const Double3 &reverse_incident_dir, Sampler &sampler) const override;
    Spectral3 Evaluate(const Double3 &reverse_indcident_dir, const Double3 &out_direction, double *pdf) const override;
};


// PBRT pg's. 682 and 899.
class HenleyGreenstein : public PhaseFunction
{
    double g;
  public:
    HenleyGreenstein(double _g)
      : g(_g) 
      {}
    ScatterSample SampleDirection(const Double3 &reverse_incident_dir, Sampler &sampler) const override;
    Spectral3 Evaluate(const Double3 &reverse_indcident_dir, const Double3 &out_direction, double *pdf) const override;
};


class Combined : public PhaseFunction
{
    static constexpr int NUM_CONSTITUENTS = 2;
    const PhaseFunction* pf[2];
    Spectral3 prob_lambda;
    Spectral3 prob_constituent_given_lambda[2];
  public:
    Combined(const Spectral3 &lambda_weight, const Spectral3 &weight1, const PhaseFunction &pf1, const Spectral3 &weight2, const PhaseFunction &pf2);
    
    ScatterSample SampleDirection(const Double3 &reverse_incident_dir, Sampler &sampler) const override;
    Spectral3 Evaluate(const Double3 &reverse_indcident_dir, const Double3 &out_direction, double *pdf) const override;
};


/* Inplace transformation of weighting factors pertaining to some components to selection probabilities of that components.
 * Example phase function sampling:
 * prob_lambda [In]: Path throughput [Out]: Probability to select each lambda.
 * prob_constituent_given_lambda [In]: Weighting factors per wavelength of each component. [Out] Probability to select each component conditioned on lambda.
 */
inline void WeightsToCombinationProbabilities(Spectral3 &prob_lambda, Spectral3 *prob_constituent_given_lambda, int num_constituents)
{
  double prob_lambda_normalization = 0.;
  for (int lambda = 0; lambda<static_size<Spectral3>(); ++lambda)
  {
    // For a given lambda, the normalization sum goes over the constituents.
    double normalization = 0.;
    for (int c=0; c<num_constituents; ++c)
    {
      // The probability weight to select a constituent for sampling is
      // just the coefficient in front of the respective scattering function.
      normalization += prob_constituent_given_lambda[c][lambda];
    }
    assert(normalization > 0.);
    for (int c=0; c<num_constituents; ++c)
    {
      prob_constituent_given_lambda[c][lambda] /= normalization;
    }
    // The weights of the current path should be reflected in the probability
    // to select some lambda. That is to prevent sampling a lambda which already
    // has a very low weight, or zero as in single wavelength sampling mode.
    // Add epsilon to protect against all zero beta.
    prob_lambda[lambda] = normalization * (prob_lambda[lambda] + Epsilon);
    prob_lambda_normalization += prob_lambda[lambda];
  }
  assert(prob_lambda_normalization > 0.);
  for (int lambda = 0; lambda<static_size<Spectral3>(); ++lambda)
  {
    prob_lambda[lambda] /= prob_lambda_normalization;
  }
}


inline Combined::Combined(const Spectral3& lambda_weight, const Spectral3& weight1, const PhaseFunctions::PhaseFunction& pf1, const Spectral3& weight2, const PhaseFunctions::PhaseFunction& pf2)
  : pf{&pf1, &pf2}, prob_lambda(lambda_weight), prob_constituent_given_lambda{weight1, weight2}
{
  WeightsToCombinationProbabilities(prob_lambda, prob_constituent_given_lambda, NUM_CONSTITUENTS);
}


}
