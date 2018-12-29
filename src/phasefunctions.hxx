#pragma once
// DO NOT INCLUDE DIRECTLY.

class Sampler;

void WeightsToCombinationProbabilities(Spectral3 &prob_lambda, Spectral3 *prob_constituent_given_lambda, int num_constituents); // forward decl. TODO find place for this.

namespace PhaseFunctions
{
  
class PhaseFunction
{
  public:
    virtual ~PhaseFunction() = default;
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




/* Inplace transformation of weighting factors to selection probabilities. Say I have multiple scattering particles in the medium.
 * Each component with its own absorption/scattering spectrum. I want to find which phase-function to sample.
 * To do this efficiently, I first find a wavelength to consider. Probabilistically according to some weight. Then given lambda,
 * I select a component probabilistically according to it's conditional weight depending on lambda.
 
 * This function normalizes the weight so that the numbers can be taken as probabilities.
 * prob_lambda [In]: Path throughput [Out]: Probability to select each lambda.
 * prob_constituent_given_lambda [In]: Weighting factors per wavelength of each component. [Out] Probability to select each component conditioned on lambda.
 */
void WeightsToCombinationProbabilities(Spectral3 &prob_lambda, Spectral3 *prob_constituent_given_lambda, int num_constituents);


class Combined : public PhaseFunction
{
    static constexpr int NUM_CONSTITUENTS = 2;
    const PhaseFunction* pf[2];
    Spectral3 prob_lambda;
    Spectral3 prob_constituent_given_lambda[2];
  public:
    Combined(const Spectral3& lambda_weight, const Spectral3& weight1, const PhaseFunction& pf1, const Spectral3& weight2, const PhaseFunction& pf2)
      : pf{&pf1, &pf2}, prob_lambda(lambda_weight), prob_constituent_given_lambda{weight1, weight2}
    {
      WeightsToCombinationProbabilities(prob_lambda, prob_constituent_given_lambda, NUM_CONSTITUENTS);
    }

    ScatterSample SampleDirection(const Double3 &reverse_incident_dir, Sampler &sampler) const override;
    Spectral3 Evaluate(const Double3 &reverse_indcident_dir, const Double3 &out_direction, double *pdf) const override;
};


class SimpleCombined : public PhaseFunction
{
    static constexpr int NUM_CONSTITUENTS = 2;
    const PhaseFunction* pf[NUM_CONSTITUENTS];
    double selection_probability[NUM_CONSTITUENTS];
    Spectral3 weights[NUM_CONSTITUENTS];
  public:
    SimpleCombined(const Spectral3& weight1, const PhaseFunction& pf1, const Spectral3& weight2, const PhaseFunction& pf2);

    ScatterSample SampleDirection(const Double3 &reverse_incident_dir, Sampler &sampler) const override;
    Spectral3 Evaluate(const Double3 &reverse_indcident_dir, const Double3 &out_direction, double *pdf) const override;
};


}
