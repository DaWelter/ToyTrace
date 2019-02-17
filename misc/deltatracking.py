# -*- coding: utf-8 -*-
"""
This is a quick evaluation of the spectral tracking method presented in
Kutz et al. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"

@author: Michael Welter <michael@welter-4d.de>
"""
from __future__ import print_function

import matplotlib.pyplot as pyplot
import random
import math
import numpy as np

import mediumspec as ms
ms.init('single_box', 'const')

def print_weights_stats(name, weights):
  estimate = np.average(weights, axis=0)
  stdev    = np.std(weights, axis=0) / math.sqrt(len(weights))
  minval   = np.amin(weights, axis=0)
  maxval   = np.amax(weights, axis=0)
  per_channel = ['%f +/- %f [%f, %f]' % (a, b,c,d) for (a,b,c,d) in zip(estimate, stdev, minval, maxval)]
  print(name, " estimate: ", ', '.join(per_channel))


def evaluate_spectral_tracking(sample_generator_function):
  print("-"*len(sample_generator_function.__name__))
  print(sample_generator_function.__name__)
  print("-" * len(sample_generator_function.__name__))
  Nsamples = 10000
  samples = []
  for i in range(Nsamples):
    s = sample_generator_function()
    samples.append(s)
  mask_interacting = np.asarray([ q[0]<=ms.domain_length for q in samples ], np.bool)
  print ("Interacting fraction: ", np.sum(mask_interacting.astype(np.float32))/Nsamples)
  print ("Escaping fraction: ", 1. - np.sum(mask_interacting.astype(np.float32))/Nsamples)

  weights = np.asarray([q[1] for q in samples])
  x = np.asarray([q[0] for q in samples])

  escaping_weights = weights.copy()
  escaping_weights[mask_interacting] = 0.

  interacting_weights = weights.copy()
  interacting_weights[~mask_interacting] = 0.

  print_weights_stats("Integral", interacting_weights)
  print_weights_stats("Transmission", escaping_weights)



def exp_sample(sigma):
  """
    Generate random number t according to the prob density sigma*exp(-sigma*t)
  """
  return -math.log(random.uniform(0., 1.))/sigma


def compute_probabilites_max_scheme(weight, *sigmas):
  """
    Probability scheme as per Sec 5.1.1.
  """
  probs = [ np.abs(a).max() for a in sigmas ]
  normalization = np.sum(probs)
  probs = [ p/normalization for p in probs ]
  return probs


def compute_probabilites_history_scheme(weight, *sigmas):
  """
    Probability scheme as per Sec 5.1.2.
  """
  probs = [ np.average(np.abs(a*weight)) for a in sigmas ]
  normalization = np.sum(probs)
  probs = [ p/normalization for p in probs ]
  return probs

the_probability_scheme = compute_probabilites_history_scheme

def spectral_tracking():
  """
    pg.  111:10, Algorithm 4
  """
  weights = np.ones(2, np.float32)
  x = 0.
  while True:
    x += exp_sample(ms.sigma_t_majorante_across_channels)
    if x > ms.domain_length:
      return (x, weights)
    else:
      sigma_s = ms.get_sigma_s(x)
      sigma_a = ms.get_sigma_a(x)
      sigma_n = ms.sigma_t_majorante_across_channels - sigma_s - sigma_a
      ps, pa, pn = the_probability_scheme(weights, sigma_s, sigma_a, sigma_n)
      r = random.uniform(0.,1.)
      if r < pa:
        return (x, np.zeros(2, np.float32))
      elif r < 1. - pn:
        weights *= sigma_s / ms.sigma_t_majorante_across_channels * ms.get_integrand(x) / ps
        return (x, weights)
      else:
        weights *= sigma_n / ms.sigma_t_majorante_across_channels / pn


def spectral_tracking_no_absorption(x = 0, weights = None):
  """
    pg.  111:10, Algorithm 4.
    Modified as per Sec. 5.1.3.
    Removal of volume absorption/emission events.
  """
  weights = weights.copy() if weights is not None else np.ones(2, np.float32)
  while True:
    x += exp_sample(ms.sigma_t_majorante_across_channels)
    if x > ms.domain_length:
      return (x, weights)
    else:
      sigma_s = ms.get_sigma_s(x)
      sigma_a = ms.get_sigma_a(x)
      sigma_n = ms.sigma_t_majorante_across_channels - sigma_s - sigma_a
      pt, pn = the_probability_scheme(weights, sigma_s, sigma_n)
      r = random.uniform(0.,1.)
      if r < pt:
        weights *= sigma_s / ms.sigma_t_majorante_across_channels * ms.get_integrand(x) / pt
        return (x, weights)
      else:
        weights *= sigma_n / ms.sigma_t_majorante_across_channels / pn


def iterated_tracking():
  w0 = np.asarray([ 80., 1.])
  x = [0]
  w = [w0.copy()]  #np.ones(2, np.float32)
  while x[-1] < ms.domain_length:
    x_, w_ = spectral_tracking_no_absorption(x[-1], w[-1])
    x.append(x_)
    w.append(w_)
  #for w_ in w:
  #  w_ /= w0
  return x, w


def analyze_iterated_tracking():
  x, w = [], []
  for _ in range(1000): # Number of rollouts
    xi, wi = iterated_tracking()
    x.extend(xi)
    w.extend(wi)
  w = np.asarray(w)
  pyplot.scatter(x, w.T[0], c='r')
  pyplot.scatter(x, w.T[1], c='b')
  pyplot.show()



#evaluate_spectral_tracking(spectral_tracking)
#evaluate_spectral_tracking(spectral_tracking_no_absorption)
analyze_iterated_tracking()

# pyplot.plot(ms.x_arr, ms.sigma_s_arr[:,0], c = 'r')
# pyplot.plot(ms.x_arr, ms.transm_array[:,0], c = 'r')
# pyplot.scatter(samples_x, samples_w[:,0], c = 'r')
#
# pyplot.plot(ms.x_arr, ms.sigma_s_arr[:,1], c = 'g')
# pyplot.plot(ms.x_arr, ms.transm_array[:,1], c = 'g')
# pyplot.scatter(samples_x, samples_w[:,1], c = 'g')
# pyplot.show()

def delta_tracking(lambda_):
  """
    pg. 111:5, Algorithm 1. (lhs). Although this is a very
    well known algorithm. Also known as woodcock tracking.
  """
  sigma_t_majorante_lambda = ms.sigma_t_majorante[lambda_]
  x = 0.
  while True:
    x += exp_sample(sigma_t_majorante_lambda)
    if x > ms.domain_length:
      return (x, 0.)
    else:
      sigma_s = ms.get_sigma_s(x)[lambda_]
      sigma_a = ms.get_sigma_a(x)[lambda_]
      sigma_n = sigma_t_majorante_lambda - sigma_s - sigma_a
      r = random.uniform(0.,1.)
      if r < (sigma_a / sigma_t_majorante_lambda):
        return (x, 0.)
      elif r < (1. - (sigma_n / sigma_t_majorante_lambda)):
        return (x, ms.get_integrand(x)[lambda_])
      else:
        pass # Null collision


def weighted_delta_tracking_no_absorption(lambda_):
  """
    pg. 111:5, Algorithm 1. (rhs).
    Modified as per Sec. 5.1.3.
    Removal of volume absorption/emission events.
  """
  sigma_t_majorante_lambda = ms.sigma_t_majorante[lambda_]
  x = 0.
  w = 1.
  while True:
    x += exp_sample(sigma_t_majorante_lambda)
    if x > ms.domain_length:
      return (x, 0.)
    else:
      sigma_s = ms.get_sigma_s(x)[lambda_]
      sigma_a = ms.get_sigma_a(x)[lambda_]
      sigma_n = sigma_t_majorante_lambda - sigma_s - sigma_a
      ps = sigma_s / (sigma_s + abs(sigma_n))
      pn = abs(sigma_n) / (sigma_s + abs(sigma_n))
      r = random.uniform(0.,1.)
      if r < ps:
        w *= sigma_s / sigma_t_majorante_lambda / ps * ms.get_integrand(x)[lambda_]
        return (x, w)
      else:
        w *= sigma_n / sigma_t_majorante_lambda / pn



