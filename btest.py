"""
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 Written (W) 2014 Soumyajit De
"""

#!/bin/bash/env python

from scipy.spatial.distance import squareform, pdist
from scipy import exp
from sklearn.preprocessing import scale
from matplotlib import pylab as plot
import random
import numpy as np
from math import sqrt

"""
 generates d dimensional normalized linspaced samples
 m is the number of samples from first distribution
 n is the number of samples from second distribution
"""
class GenerateSamples:
	@staticmethod
	def linspaced(m, n, d):
		data_p = [range(j,j+d) for j in range(0,m*d,d)]
		data_q = [range(j,j+d) for j in range(m*d,(m+n)*d,d)]
		return data_p, data_q

	@staticmethod
	def linspaced_normalized(m, n, d):
		data_p = map(lambda x:map(lambda y:y/(m*d-1.0),x),[range(j,j+d) for j in range(0,m*d,d)])
		data_q = map(lambda x:map(lambda y:y/((m+n)*d-1.0),x),[range(j,j+d) for j in range(m*d,(m+n)*d,d)])
		return data_p, data_q

"""
 computes a radial basis function kernel provided data points are assumed to be appended
"""
class Rbf:
	sigma = 0
	def __init__(self, sigma):
		self.sigma = sigma
	def compute(self, x_y):
		width = 2.0*self.sigma**2
		pairwise_distance = squareform(pdist(x_y, 'euclidean'))
		return exp(-pairwise_distance**2/width)

class StreamingMMD(object):
	m = 0
	n = 0
	B = 0
	Bx = 0
	By = 0
	data_p = None
	data_q = None
	kernel = None
	statistic_type = None
	variance_type = None

	def __init__(self, m, n, blocksize, p, q, kernel = None):
		self.m = m
		self.n = n
		self.data_p = p
		self.data_q = q
		self.B = blocksize
		self.Bx = m*blocksize/(m+n)
		self.By = n*blocksize/(m+n)
		self.kernel = kernel
		print m, n, p, q, self.B, self.Bx, self.By, self.kernel

	def set_statistic_type(self, statistic_type):
		self.statistic_type = statistic_type
		if self.statistic_type == "INCOMPLETE":
			if self.m != self.n:
				raise NotImplementedError("m and n should be equal for incomplete type statistic")

	def set_variance_type(self, variance_type):
		self.variance_type = variance_type

	def set_kernel(self, kernel):
		self.kernel = kernel

	def compute_statistic_variance(self):
		bx=0
		by=0
		statistic = 0.0
		variance = 0.0
		for b in range(0, (self.m + self.n)/self.B):
			print 'processing block', b
			block_p = data_p[bx: bx+self.Bx]
			block_q = data_q[by: by+self.By]
			result = self.compute_blockwise_statistic_variance(np.array(block_p + block_q))
			statistic += result[0]
			variance += result[1]
			bx+=self.Bx
			by+=self.By
		statistic /= (self.m+self.n)/self.B
		variance /= (self.m+self.n)/self.B
		return statistic * self.stat_multiplier(), variance

	def stat_multiplier(self):
		raise NotImplementedError("Please implement this method")
		return 0

	def compute_blockwise_statistic_variance(self, p_and_q):
		kmatrix = self.kernel.compute(p_and_q)

		np.fill_diagonal(kmatrix, 0)
		print kmatrix

		km_pp = kmatrix[np.ix_(range(self.Bx),range(self.Bx))]
		first = np.sum(km_pp)/self.Bx/(self.Bx-1)

		km_qq = kmatrix[np.ix_(range(self.Bx,self.B),range(self.Bx,self.B))]
		second = np.sum(km_qq)/self.By/(self.By-1)

		km_pq = kmatrix[np.ix_(range(self.Bx),range(self.Bx,self.B))]
		third = 0.0
		if self.statistic_type == "UNBIASED":
			third = 2*np.sum(km_pq)/self.Bx/self.By
		else:
			np.fill_diagonal(km_pq, 0)
			third = 2*np.sum(km_pq)/self.Bx/(self.Bx-1)

		print first, '\n', second, '\n', third
		statistic = first + second - third
		print 'statistic', statistic

		# within block direct estimation
		first = np.sum(np.square(kmatrix))
		second = np.sum(kmatrix)**2/(self.B-1)/(self.B-2)
		third = 2*np.sum(np.linalg.matrix_power(kmatrix,2))/(self.B-2)

		print first, '\n', second, '\n', third
		variance = 2*(first + second - third) / self.B / (self.B-3)
		print 'variance', variance

		return statistic, variance

	def perform_test(self):
		statistic, variance = self.compute_statistic_variance()

class LinearTimeMMD(StreamingMMD):
	def stat_multiplier(self):
		multiplier = sqrt((self.m*self.n)/float(self.m+self.n))
		return multiplier

class BTestMMD(StreamingMMD):
	def stat_multiplier(self):
		multiplier = self.m*self.n/float(self.m+self.n)*sqrt(float(self.B)/(self.m+self.n))
		return multiplier

if __name__ == '__main__':
	m = 2
	n = 2
	d = 3
	blocksize = 4
	data_p,data_q=GenerateSamples.linspaced(m, n, d)
#kernel = Rbf(sigma)
	mmd = LinearTimeMMD(m, n, blocksize, data_p, data_q)
	mmd.set_statistic_type("INCOMPLETE")
	mmd.set_variance_type("WITHIN_BLOCK_DIRECT")
	for j in range(5,8):
		sigma = 2
		kernel = Rbf(sigma)
		mmd.set_kernel(kernel)
		statistic, variance = mmd.compute_statistic_variance()
		print "statistic = {0:.15f}".format(statistic)
		print "variance = {0:.15f}".format(variance)
