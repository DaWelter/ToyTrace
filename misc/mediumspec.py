# -*- coding: utf-8 -*-
"""
Specification of a participating medium:
Collision coefficients along one axis.
An integrand.
And majorants of the collision coefficients.

@author: Michael Welter <michael@welter-4d.de>
"""

import numpy as np

domain_length = 1.
x_arr = np.linspace(0., domain_length, 10000)
delta_x = x_arr[1]-x_arr[0]
if 1:
  # Two adjacent boxes of differing heights.
  def get_sigma_s(x):
    if 0.2 < x <= 0.5:
      return np.asarray([2., 0.1])
    elif 0.5 < x < 0.8:
      return np.asarray([0.1, 0.1])
    return np.asarray([0.001, 0.001])
  def get_sigma_a(x):
    return np.asarray([0.1, 0.1])
else:
  # A single box. But varying heights per wavelength.
  def get_sigma_s(x):
    if 0.25 < x <= 0.75:
      return np.asarray([2., 0.1])
    else:
      return np.asarray([0., 0.])
  def get_sigma_a(x):
    return np.asarray([2., 2.])
if 1:
  def get_integrand(x):
    return np.ones(2, np.float32)
if 0:
  # To estimate the transmission I can use the following as
  # integrand factor. It will cancel sigma_s and replace it with sigma_t.
  # So the integral integrates the probability density which equals T.
  def get_integrand(x):
    return (get_sigma_a(x)+get_sigma_s(x))/get_sigma_s(x)

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