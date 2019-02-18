# -*- coding: utf-8 -*-
"""
This is a quick evaluation of the ratio tracking method presented in
Kutz et al. (2017) "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"

@author: Michael Welter <michael@welter-4d.de>
"""
from __future__ import print_function

import random
import math
import numpy as np

import mediumspec as ms
ms.init('single_box', 'const')

def exp_sample(sigma):
  """
    Generate random number t according to the prob density sigma*exp(-sigma*t)
  """
  return -math.log(random.uniform(0., 1.))/sigma


def single_lambda_ratio_tracking(lambda_):
  """
    pg. 111:16, Eq (41)
  """
  sigma_t_majorante_lambda = ms.sigma_t_majorante[lambda_]
  x = 0.
  weight = 1.
  while True:
    x += exp_sample(sigma_t_majorante_lambda)
    if x > ms.domain_length:
      return weight
    else:
      sigma_s = ms.get_sigma_s(x)[lambda_]
      sigma_a = ms.get_sigma_a(x)[lambda_]
      sigma_n = sigma_t_majorante_lambda - sigma_s - sigma_a
      weight *= sigma_n / sigma_t_majorante_lambda

def ratio_tracking():
  """
    pg. 111:16, Eq (41)
  """
  sigma_t_majorante = ms.sigma_t_majorante_across_channels
  x = 0.
  weight = np.ones(2, np.float32)
  while True:
    x += exp_sample(sigma_t_majorante)
    if x > ms.domain_length:
      return weight
    else:
      sigma_s = ms.get_sigma_s(x)
      sigma_a = ms.get_sigma_a(x)
      sigma_n = sigma_t_majorante - sigma_s - sigma_a
      weight *= sigma_n / sigma_t_majorante


def weighted_next_flight_estimator():
  """
    pg. 111:16, Eq (39)
    More variance than ratio tracking?!
  """
  sigma_t_majorante = ms.sigma_t_majorante_across_channels
  x = 0.
  weight_product = np.ones(2, np.float32)
  weight = np.zeros(2, np.float32)
  while True:
    weight += np.exp(sigma_t_majorante * (x - ms.domain_length)) * weight_product
    x += exp_sample(sigma_t_majorante)
    if x > ms.domain_length:
      return weight
    else:
      sigma_s = ms.get_sigma_s(x)
      sigma_a = ms.get_sigma_a(x)
      sigma_n = sigma_t_majorante - sigma_s - sigma_a
      weight_product *= sigma_n / sigma_t_majorante


def run_ratio_tracking():
  Nsamples = 10000

  samples_per_lambda = [ [] for _ in  range(2) ]
  for lambda_ in  range(len(samples_per_lambda)):
    for i in range(Nsamples):
      w = single_lambda_ratio_tracking(lambda_)
      samples_per_lambda[lambda_].append(w)
  samples_per_lambda = np.asarray(samples_per_lambda).T

  estimate = np.average(samples_per_lambda, axis=0)
  stdev    = np.std(samples_per_lambda, axis=0) / math.sqrt(len(samples_per_lambda))

  print ("Single Lambda Ratio Tracking: ", estimate, " +/- ", stdev)

  samples = np.asarray([ ratio_tracking() for _ in range(Nsamples)])
  estimate = np.average(samples, axis=0)
  stdev    = np.std(samples, axis=0) / math.sqrt(len(samples))

  print ("Ratio Tracking: ", estimate, " +/- ", stdev)

  samples = np.asarray([ weighted_next_flight_estimator() for _ in range(Nsamples)])
  estimate = np.average(samples, axis=0)
  stdev    = np.std(samples, axis=0) / math.sqrt(len(samples))

  print ("Weighted Next Flight Estimator: ", estimate, " +/- ", stdev)


def c_(*args):
  return np.concatenate(tuple(map(np.atleast_1d, args)), axis=0)


def construct_piecewise_constant_transmittance():
  """
    By ratio tracking.
    We have positions xi, and associated weight wi. If we want to obtain
    an estimate of the transmittance of some point y, T(y) we look up
    the next xi which comes afterwards and use the estimate T^(y) = wi.
    In expectation T^(y) equals the true transmittance T(y).

  """
  sigma_t_majorante = ms.sigma_t_majorante_across_channels
  x = 0.
  weight = np.ones(2, np.float32)
  points_and_weights = [c_(x, weight)]
  while True:
    x += exp_sample(sigma_t_majorante)
    if x > ms.domain_length:
      points_and_weights += [c_(x,weight)]
      break
    else:
      points_and_weights += [c_(x, weight)]
      sigma_s = ms.get_sigma_s(x)
      sigma_a = ms.get_sigma_a(x)
      sigma_n = sigma_t_majorante - sigma_s - sigma_a
      weight *= sigma_n / sigma_t_majorante
      term_criterion = np.amax(weight)
      # Russian roulette termination. Otherwise I would have to traverse
      # through the entire domain which would be very inefficient if the
      # mean free path length is short
      r = random.uniform(0., 1.)
      survival_probability = term_criterion
      if r < survival_probability:  # live on
        weight /= survival_probability
      else:
        points_and_weights += [c_(ms.domain_length*1.1,np.zeros_like(weight))]
        break
  return np.asarray(points_and_weights)


pw = construct_piecewise_constant_transmittance()


def lookup_piecewise(pw, xs):
  idx = np.digitize(xs, pw.T[0])
  weights = pw[idx,1:]
  return weights


def plot_piecewise(pw):
  import matplotlib.pyplot as pyplot

  fig, ax = pyplot.subplots(2,2)
  ax = ax.ravel()

  for i in range(2):
    ax[i].bar(pw.T[0][1:], height=pw.T[i+1][1:], width=pw.T[0][:-1] - pw.T[0][1:], align='edge', color='rb'[i])
    ax[i].plot(ms.x_arr, ms.transm_array[:,i], color='k')

  ax[2].vlines(pw.T[0], 0, 1.)
  ax[2].scatter(pw.T[0], pw.T[1], color='r')
  ax[2].scatter(pw.T[0], pw.T[2], color='b')


  xs = np.linspace(0., ms.domain_length, 100)
  w = lookup_piecewise(pw, xs)

  dx=(xs[1]-xs[0])*0.5
  ax[3].bar(xs, height=w.T[0], width=dx, color='rb'[0])
  ax[3].bar(xs+dx, height=w.T[1], width=dx, color='rb'[1])

  for a in ax:
    a.set(xlim=(0.,1.))

  pyplot.show(block = False)

plot_piecewise(pw)


def generate_and_plot_average():
  '''
    Numerically compute the expected value of the piecewise transmittance estimate
    at different points along the x-axis.
  '''
  import matplotlib.pyplot as pyplot

  print('Generating Average ...')
  xs = np.linspace(0., ms.domain_length, 100)
  avg = []
  weights = []
  for i in range(1000):
    pw = construct_piecewise_constant_transmittance()
    weights.append(pw[:,1:])
    w = lookup_piecewise(pw, xs)
    avg.append(w)
    if i % 100:
      print ('.',)
  w = np.average(avg, axis=0)
  weights = np.concatenate(tuple(weights),axis=0)
  print ("Weights max:", np.amax(weights,axis=0))
  print ("Weights avg:", np.average(weights,axis=0))

  fig ,ax = pyplot.subplots(1,1)

  dx=(xs[1]-xs[0])*0.5
  ax.bar(xs, height=w.T[0], width=dx, color='rb'[0])
  ax.bar(xs+dx, height=w.T[1], width=dx, color='rb'[1])
  ax.plot(ms.x_arr, ms.transm_array[:, 0], color='r')
  ax.plot(ms.x_arr, ms.transm_array[:, 1], color='b')
  pyplot.show()

generate_and_plot_average()