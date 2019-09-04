#pragma once

#include <tuple>

// A review of statistical testing applied to computer graphics!
// http://www0.cs.ucl.ac.uk/staff/K.Subr/Files/Papers/PG2007.pdf

//http://mathworld.wolfram.com/BinomialDistribution.html
std::tuple<double, double> BinomialDistributionMeanStdev(int n, double p);

std::tuple<double, double> BinomialDistributionMeanStdev(int _n, double _p, double _p_err);



void CheckNumberOfSamplesInBin(const char *name, int num_smpl_in_bin, int total_num_smpl, double p_of_bin, double number_of_sigmas_threshold=3., double p_of_bin_error = 0.);

double ChiSquaredProbability(const int *counts, const double *weights, int num_bins, double low_expected_num_samples_cutoff = 5);

// By https://en.wikipedia.org/wiki/Central_limit_theorem
// for large number of samples.
double StddevOfAverage(double sample_stddev, int num_samples);

// The standard estimator of the sample standard deviation is itself
// a random variable because it is composed of random variables.
// So it has a probability distribution. The distribution is
// a "scaled chi-squared distribution".
// See https://en.wikipedia.org/w/index.php?title=Variance&oldid=735567901#Distribution_of_the_sample_variance
// This function returns the standard deviation of this distribution.
double StddevOfStddev(double sample_stddev, int num_samples);

double StudentsTValue(double sample_avg, double avg_stddev, double true_mean);

double StudentsTDistributionOutlierProbability(double t, double num_samples);

void TestSampleAverage(double sample_avg, double sample_stddev, double num_samples, double true_mean, double true_mean_error, double p_value);

void TestProbabilityOfMeanLowerThanUpperBound(double sample_avg, double sample_stddev, double num_samples, double bound, double p_value);
