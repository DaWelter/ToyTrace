# -*- coding: utf-8 -*-
"""
This is a quick evaluation of the ratio tracking method presented in
Kutz et al. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"

@author: Michael Welter <michael@welter-4d.de>
"""


import random
import math
import numpy as np

import mediumspec as ms

def exp_sample(sigma):
  """
    Generate random number t according to the prob density sigma*exp(-sigma*t)
  """
  return -math.log(random.uniform(0., 1.))/sigma


def ratio_tracking(lambda_):
  """
    pg. 111:16, Eq (41)
  """
  sigma_t_majorante_lambda = ms.sigma_t_majorante[lambda_]
  x = 0.
  weight = 1.
  while True:
    x += exp_sample(sigma_t_majorante_lambda)
    if x > ms.domain_length:
      return (x, weight)
    else:
      sigma_s = ms.get_sigma_s(x)[lambda_]
      sigma_a = ms.get_sigma_a(x)[lambda_]
      sigma_n = sigma_t_majorante_lambda - sigma_s - sigma_a
      weight *= sigma_n / sigma_t_majorante_lambda


Nsamples = 10000
samples_per_lambda = [ [] for _ in  range(ms.sigma_s_arr.shape[1]) ]
for lambda_ in  range(len(samples_per_lambda)):
  for i in range(Nsamples):
    x, w = ratio_tracking(lambda_)
    samples_per_lambda[lambda_].append(w)
samples_per_lambda = np.asarray(samples_per_lambda).T

estimate = np.average(samples_per_lambda, axis=0)
stdev    = np.std(samples_per_lambda, axis=0) / math.sqrt(len(samples_per_lambda))

print ("Ratio Lambda Tracking: ", estimate, " +/- ", stdev)