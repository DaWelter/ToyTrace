#include "tests_stats.hxx"

#include "gtest/gtest.h"
#include "vec3f.hxx"
#include "util.hxx"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/numeric/interval.hpp>



// A review of statistical testing applied to computer graphics!
// http://www0.cs.ucl.ac.uk/staff/K.Subr/Files/Papers/PG2007.pdf

//http://mathworld.wolfram.com/BinomialDistribution.html
std::tuple<double, double> BinomialDistributionMeanStdev(int n, double p)
{
  assert(p >= 0.);
  return std::make_tuple(p*n, std::sqrt(n*p*(1.-p)));
}

std::tuple<double, double> BinomialDistributionMeanStdev(int _n, double _p, double _p_err)
{
  assert(_p >= 0. && _p_err >= 0.);
  using namespace boost::numeric;
  using namespace interval_lib;
  using I = interval<double>;
  double pl = std::max(0., _p-_p_err);
  double ph = std::min(1., _p+_p_err);
  I p{pl, ph};
  I n{_n, _n};
  I mean = p*n;
  I stddev = sqrt(n*p*(1.-p));
  double uncertainty = stddev.upper() + 0.5*(mean.upper()-mean.lower());
  return std::make_tuple(_p*_n, uncertainty);
}


void CheckNumberOfSamplesInBin(const char *name, int num_smpl_in_bin, int total_num_smpl, double p_of_bin, double number_of_sigmas_threshold, double p_of_bin_error)
{
  double mean, sigma;
  std::tie(mean, sigma) = BinomialDistributionMeanStdev(total_num_smpl, p_of_bin, p_of_bin_error);
  std::stringstream additional_output;
  if (name)
    additional_output << "Expected in " << name << ": " << mean << "+/-" << sigma << " Actual: " << num_smpl_in_bin << " of " << total_num_smpl << std::endl;
  EXPECT_NEAR(num_smpl_in_bin, mean, sigma*number_of_sigmas_threshold) << additional_output.str();
}


namespace ChiSqrInternal
{

struct Bin {
  double count;
  double expected;
};


// TODO: I wish I had a super lightweight range checked (in debug mode only ofc.) array view.

void MergeBinsToGetAboveMinNumSamplesCutoff(Bin *bins, int &num_bins, double low_expected_num_samples_cutoff)
{
  //http://en.cppreference.com/w/cpp/algorithm/make_heap
  // ... and so on.
  auto cmp = [](const Bin &a, const Bin &b) -> bool { return a.count > b.count; }; // Using > instead of < makes it a min-heap.
  std::make_heap(bins, bins+num_bins, cmp);
  while (num_bins > 1)
  {
    // Get smallest element to the end.
    std::pop_heap(bins, bins+num_bins, cmp);
    if (bins[num_bins-1].expected > low_expected_num_samples_cutoff)
      break;
    // Get the second smallest element to the second last position.
    std::pop_heap(bins, bins+num_bins-1, cmp);
    // Merge them.
    Bin &merged = bins[num_bins-2];
    Bin &removed = bins[num_bins-1];
    merged.count += removed.count;
    merged.expected += removed.expected;
    removed.count = 0;
    removed.expected = 0;
    --num_bins; 
    // Modifications done. Get last element back into heap.
    std::push_heap(bins, bins+num_bins, cmp);
  }
  assert(num_bins >= 1);
  assert((num_bins == 1) || (bins[num_bins-1].expected > low_expected_num_samples_cutoff));
}

}


double ChiSquaredProbability(const int *counts, const double *weights, int num_bins, double low_expected_num_samples_cutoff)
{
  using namespace ChiSqrInternal;
  
  int num_samples = std::accumulate(counts, counts+num_bins, 0); // It's the sum.
  double probability_normalization = std::accumulate(weights, weights+num_bins, 0.);

  probability_normalization = probability_normalization>0. ? 1./probability_normalization : 0.;
  
  std::unique_ptr<Bin[]> bins{new Bin[num_bins]};
  for (int i=0; i<num_bins; ++i)
    bins[i] = Bin{(double)counts[i], weights[i]*num_samples*probability_normalization};

  int original_num_bins = num_bins;
  MergeBinsToGetAboveMinNumSamplesCutoff(bins.get(), num_bins, low_expected_num_samples_cutoff);
  
  double chi_sqr = 0.;
  for (int i=0; i<num_bins; ++i)
  {
    chi_sqr += Sqr(bins[i].count - bins[i].expected)/bins[i].expected;
  }

  if (original_num_bins > num_bins*3)
  {
    std::cout << "Chi-Sqr Test merged " << (original_num_bins-num_bins) << " bins because their sample count was below " << low_expected_num_samples_cutoff << std::endl;
  }
  
  // https://www.boost.org/doc/libs/1_66_0/libs/math/doc/html/math_toolkit/dist_ref/dists/chi_squared_dist.html
  // https://www.boost.org/doc/libs/1_66_0/libs/math/doc/html/math_toolkit/dist_ref/nmp.html#math_toolkit.dist_ref.nmp.cdf
  boost::math::chi_squared_distribution<double> distribution(num_bins-1);
  double prob_observe_ge_chi_sqr = cdf(complement(distribution, chi_sqr));
  
  //std::cout << "chi-sqr prob = " << prob_observe_ge_chi_sqr << std::endl;
  //EXPECT_GE(prob_observe_ge_chi_sqr, p_threshold);
  return prob_observe_ge_chi_sqr;  
}


// By https://en.wikipedia.org/wiki/Central_limit_theorem
// for large number of samples.
double StddevOfAverage(double sample_stddev, int num_samples)
{
  return sample_stddev / std::sqrt(num_samples);
}


double StudentsTValue(double sample_avg, double avg_stddev, double true_mean)
{
  return (sample_avg - true_mean) / avg_stddev;
}


double StudentsTDistributionOutlierProbability(double t, double num_samples)
{
  boost::math::students_t_distribution<double> distribution(num_samples);
  if (t > 0.)
    return cdf(complement(distribution, t));
  else
    return cdf(distribution, t);
}


void TestSampleAverage(double sample_avg, double sample_stddev, double num_samples, double true_mean, double true_mean_error, double p_value)
{
  double avg_stddev = StddevOfAverage(sample_stddev, num_samples);
  if (avg_stddev > 0.)
  {
    double prob;
    // Normally I would use the probability of the distance greater than |(sample_avg - true_mean)| w.r.t. to a normal distribution.
    // I use normal distribution rather than t-test stuff because I have a lot of samples.
    // However, since I also have integration errors, I "split" the normal distribution in half and move both sides to the edges of
    // the integration error bounds. Clearly, I the MC estimate is very good, I want to check if sample_avg is precisely within the 
    // integration bounds. On the other hand, if the integration error is zero, I want to use the usual normal distribution as 
    // stated above. My scheme here achives both of these edge cases.
    if (sample_avg < true_mean - true_mean_error)
      prob = StudentsTDistributionOutlierProbability(StudentsTValue(sample_avg, avg_stddev, true_mean - true_mean_error), num_samples);
    else if(sample_avg > true_mean + true_mean_error)
      prob = StudentsTDistributionOutlierProbability(StudentsTValue(sample_avg, avg_stddev, true_mean + true_mean_error), num_samples);
    else
      prob = StudentsTDistributionOutlierProbability(0., num_samples);
    EXPECT_GE(prob, p_value) << "Probability to find sample average " << sample_avg << " +/- " << avg_stddev << " w.r.t. true mean " << true_mean << "+/-" << true_mean_error << 
    " is " << prob << " which is lower than p-value " << p_value << std::endl;
    //std::cout << "Test Sample Avg Prob = " << prob << std::endl;
  }
  else
  {
    EXPECT_NEAR(sample_avg, true_mean, true_mean_error);
  }
}


void TestProbabilityOfMeanLowerThanUpperBound(double sample_avg, double sample_stddev, double num_samples, double bound, double p_value)
{
  double avg_stddev = StddevOfAverage(sample_stddev, num_samples);
  if (avg_stddev > 0.)
  {
    boost::math::students_t_distribution<double> distribution(num_samples);
    double prob = cdf(complement(distribution, StudentsTValue(sample_avg, avg_stddev, bound)));
    EXPECT_GE(prob, p_value) << "Probability to find the mean smaller than " << bound << " is " << prob << " which is lower than the p-value " << p_value << ", given the average " << sample_avg << " +/- " << avg_stddev << std::endl;
    //std::cout << "Test Prob Upper Bound = " << prob << std::endl;
  }
  else
  {
    EXPECT_LE(sample_avg, bound);
  }
}




TEST(BinomialDistribution, WithIntervals)
{
  double mean1, stddev1;
  double mean2, stddev2;
  std::tie(mean1, stddev1) = BinomialDistributionMeanStdev(10, 0.6);
  std::tie(mean2, stddev2) = BinomialDistributionMeanStdev(10, 0.6, 0.);
  EXPECT_NEAR(mean1, mean2, 1.e-9);
  EXPECT_NEAR(stddev1, stddev2, 1.e-9);
}
