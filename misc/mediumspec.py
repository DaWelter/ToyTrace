# -*- coding: utf-8 -*-
"""
Specification of a participating medium:
Collision coefficients along one axis.
An integrand.
And majorants of the collision coefficients.

@author: Michael Welter <michael@welter-4d.de>
"""
from __future__ import print_function

import numpy as np

domain_length = 1.
x_arr = np.linspace(0., domain_length, 10000)
delta_x = x_arr[1]-x_arr[0]


def init(coefficient_distribution, integrand):
  global get_sigma_s, get_sigma_a, get_integrand
  if coefficient_distribution == 'two_boxes':
    # Two adjacent boxes of differing heights.
    def get_sigma_s_(x):
      if 0.2 < x <= 0.5:
        return np.asarray([2., 0.1])
      elif 0.5 < x < 0.8:
        return np.asarray([0.1, 0.1])
      return np.asarray([0.001, 0.001])
    def get_sigma_a_(x):
      return np.asarray([0.1, 0.1])
  elif coefficient_distribution == 'single_box':
    # A single box. But varying heights per wavelength.
    def get_sigma_s_(x):
      if 0.25 < x <= 0.75:
        return np.asarray([5, 10])
      else:
        return np.asarray([0., 0.])
    def get_sigma_a_(x):
      return np.asarray([0.1, 0.1])

  if integrand == 'const':
    def get_integrand_(x):
      return np.ones(2, np.float32)
  elif integrand == 'sigma_t_over_sigma_s':
    # To estimate the transmission I can use the following as
    # integrand factor. It will cancel sigma_s and replace it with sigma_t.
    # So the integral integrates the probability density which equals 1-T.
    def get_integrand_(x):
      return (get_sigma_a(x)+get_sigma_s(x))/get_sigma_s(x)
  get_sigma_s = get_sigma_s_
  get_sigma_a = get_sigma_a_
  get_integrand = get_integrand_
  discretized_solution_()


def discretized_solution_():
  global sigma_t_majorante, sigma_t_majorante_across_channels
  global transm_array
  # Discretize to arrays. Thus I can plot the functions. And also calculate the RTE integral numerically.
  sigma_s_arr = np.asarray(map(get_sigma_s, x_arr))
  sigma_a_arr = np.asarray(map(get_sigma_a, x_arr))
  sigma_t_arr = sigma_s_arr+sigma_a_arr
  integrand_arr = np.asarray(map(get_integrand, x_arr))
  # First note the majorantes.
  sigma_t_majorante = np.max(sigma_t_arr, axis = 0)
  sigma_t_majorante_across_channels = sigma_t_arr.max()
  # Compute RTE integral.
  transm_array = np.exp(-np.cumsum(sigma_t_arr * delta_x, axis=0))
  the_integral = np.sum(integrand_arr * sigma_s_arr * transm_array * delta_x, axis=0)
  print ("The Integral  = ", the_integral)
  print ("One minus I   = ", 1. - the_integral)
  print ("Transmission to the end = ", transm_array[-1,:])