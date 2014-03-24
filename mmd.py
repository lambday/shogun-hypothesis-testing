"""
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 Written (W) 2014 Soumyajit De
"""

#!/bin/basn/env python

from scipy.spatial.distance import squareform, pdist
from scipy import exp
from sklearn.preprocessing import scale
from matplotlib import pylab as plot
import random
import numpy as np
from math import sqrt

"""
 generates 1 dimensional gaussian samples
"""
class GenerateSamples:
	@staticmethod
	def generate_gaussian(p, q, mu_p, mu_q, var_p, var_q):
		samples_from_p = [random.gauss(mu_p, sqrt(var_p)) for j in range(p)]
		samples_from_q = [random.gauss(mu_q, sqrt(var_p)) for j in range(q)]
		return np.array(samples_from_p + samples_from_q)

"""
 class QuadraticTimeMMD that implements computation of unbiased statistic
 MMD_u^2 and MMD_b^2 (See [1] for details). It also provides a method for
 sampling null spectral distribution.

 [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
"""
class QuadraticTimeMMD:
	num_samples_p = 0
	num_samples_q = 0
	kernel = np.array([])
	"""
	initializes the kernel computed on appended samples
	"""
	def __init__(self, p, q, samples_from_p_q):
		self.num_samples_p = p
		self.num_samples_q = q
#		print 'samples', self.samples_from_p_q
		self.kernel = self.rbf(samples_from_p_q, 2)
#		print 'kernel computed'
#		print self.kernel

	"""
	computes a radial basis function kernel on one dimensional data
	provided data points are assumed to be appended
	"""
	@staticmethod
	def rbf(x_y, width):
#		print 'computing rbf between samples', x_y
		pairwise_distance = squareform(pdist(x_y[:, None], 'euclidean'))
#		print 'pairwise_distance', pairwise_distance
		return exp(-pairwise_distance ** 2 / width)

	"""
	computes MMD_u^2 estimate. The return value is (m+n)*MMD_u^2 in general
	and m*MMD_u^2 when m and n are equal
	"""
	def compute_unbiased_statistic(self):
		first = 0.0
		m = self.num_samples_p
		n = self.num_samples_q
		for i in range(m):
			for j in range(m):
				if i != j:
					first += self.kernel.item(i, j)
		first /= m * (m - 1)
		second = 0.0
		for i in range(m, m + n):
			for j in range(m, m + n):
				if i != j:
					second += self.kernel.item(i, j)
		second /= n * (n - 1)
		third = 0.0
		for i in range(m):
			for j in range(m, m + n):
				third += self.kernel.item(i, j)
		statistic = (m + n) * (first + second - third)

		if m == n:
			statistic /= 2
		return statistic

	"""
	computes MMD_b^2 estimate. The return value is (m+n)*MMD_u^2 in general
	and m*MMD_b^2 when m and n are equal
	"""
	def compute_biased_statistic(self):
		first = 0.0
		m = self.num_samples_p
		n = self.num_samples_q
		for i in range(m):
			for j in range(m):
				first += self.kernel.item(i, j)
		first /= m ** 2
		second = 0.0
		for i in range(m, m + n):
			for j in range(m, m + n):
				second += self.kernel.item(i, j)
		second /= n ** 2
		third = 0.0
		for i in range(m):
			for j in range(m, m + n):
				third += self.kernel.item(i, j)
		statistic = (m + n) * (first + second - third)

		if m == n:
			statistic /= 2
		return statistic

	"""
	estimates spectral approximation of null samples
	"""
	def sample_null_spectrum(self, num_samples, num_ev = 3, statistic = 'biased'):
		centered = scale(self.kernel)
#		print centered
		ev, evec = np.linalg.eig(centered)
#		print 'computed eigenvalues'
		m = self.num_samples_p
		n = self.num_samples_q
		rho_x = float(m) / (m + n)
		rho_y = 1 - rho_x
		std_dev = 1 / rho_x + 1 / rho_y
		null_samples = [0] * num_samples
		for i in range(num_samples):
			null_sample = 0
			for j in range(num_ev):
				ev_j = 1.0 / (m + n) * ev[j]
				z_j = random.gauss(0, std_dev)
				term = z_j * z_j
				if statistic == 'unbiased':
					term = term - 1.0 / (rho_x * rho_y)
				null_sample += ev_j * term
			null_samples[i] = null_sample
#		print 'sampled null spectrum'

		if m == n:
			null_samples = map(lambda x: x / 2, null_samples)
		return np.array(null_samples)

if __name__ == '__main__':
	num_null_samples = 1000
	estimates = [0] * num_null_samples
	m = 10
	n = 10
	samples_h0 = GenerateSamples.generate_gaussian(m, n, 0, 0, 1, 1)
	samples_h1 = GenerateSamples.generate_gaussian(m, n, 0, 0, 10, 1)

	mmd = QuadraticTimeMMD(m, n, samples_h0)
	for i in range(num_null_samples):
		estimates[i] = mmd.sample_null_spectrum(1000).mean()
	plot.hist(estimates, 50, normed = 1, alpha = 0.5)

	mmd = QuadraticTimeMMD(m, n, samples_h1)
	for i in range(num_null_samples):
		estimates[i] = mmd.sample_null_spectrum(1000).mean()
	plot.hist(estimates, 50, normed = 1, alpha = 0.5)

	plot.legend(["H0", "H1"])
#	plot.xlim([10, 40])
#plot.ylim([0, 1])
	plot.show()
