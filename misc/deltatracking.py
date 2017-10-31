# -*- coding: utf-8 -*-
"""
This is a quick evaluation of the spectral tracking method presented in
Kutz et al. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"

@author: Michael Welter <michael@welter-4d.de>
"""
import matplotlib.pyplot as pyplot
import random
import math
import numpy as np

import mediumspec as ms

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
      return None
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


def spectral_tracking_no_absorption():
  """
    pg.  111:10, Algorithm 4.
    Modified as per Sec. 5.1.3.
    Removal of volume absorption/emission events.
  """
  weights = np.ones(2, np.float32)
  x = 0.
  while True:
    x += exp_sample(ms.sigma_t_majorante_across_channels)
    if x > ms.domain_length:
      return (x, np.zeros_like(weights))
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


Nsamples = 10000
samples = []
for i in range(Nsamples):
  s = spectral_tracking_no_absorption()
  samples.append(s)
interacting_samples = filter(lambda q: q[0]<ms.domain_length, samples)
print ("Interacting fraction: ", float(len(interacting_samples))/Nsamples)
print ("Escaping fraction: ", 1. - float(len(interacting_samples))/Nsamples)
samples_x = map(lambda q: q[0], samples)
samples_w = np.asarray(map(lambda q: q[1], samples))

estimate = np.average(samples_w, axis=0)
stdev    = np.std(samples_w, axis=0) / math.sqrt(Nsamples)

print ("Tracking estimate: ", estimate, " +/- ", stdev)

pyplot.plot(ms.x_arr, ms.sigma_s_arr[:,0], c = 'r')
pyplot.plot(ms.x_arr, ms.transm_array[:,0], c = 'r')
pyplot.scatter(samples_x, samples_w[:,0], c = 'r')

pyplot.plot(ms.x_arr, ms.sigma_s_arr[:,1], c = 'g')
pyplot.plot(ms.x_arr, ms.transm_array[:,1], c = 'g')
pyplot.scatter(samples_x, samples_w[:,1], c = 'g')


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



samples_per_lambda = [ [] for _ in  range(ms.sigma_s_arr.shape[1]) ]
escapes_per_lambda = np.zeros(ms.sigma_s_arr.shape[1], np.float32)
for lambda_ in  range(len(samples_per_lambda)):
  for i in range(Nsamples):
    x, w = weighted_delta_tracking_no_absorption(lambda_)
    samples_per_lambda[lambda_].append(w)
    escapes_per_lambda[lambda_] += 1 if x > ms.domain_length else 0
samples_per_lambda = np.asarray(samples_per_lambda).T

estimate = np.average(samples_per_lambda, axis=0)
stdev    = np.std(samples_per_lambda, axis=0) / math.sqrt(len(samples_per_lambda))

print ("Single Lambda Tracking: ", estimate, " +/- ", stdev)
print ("Fraction of escapes: ", escapes_per_lambda / len(samples_per_lambda))

pyplot.show()
