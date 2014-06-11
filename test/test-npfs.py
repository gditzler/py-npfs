#!/usr/bin/env python 
import numpy as np 
import npfs 

__author__ = "Gregory Ditzler"
__copyright__ = "Copyright 2014, EESI Laboratory (Drexel University)"
__credits__ = ["Gregory Ditzler"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Gregory Ditzler"
__email__ = "gregory.ditzler@gmail.com"
__status__ = "development"

n_select = 5
n_boots = 50
fs_method = "MIM"
alpha = 0.01
beta = 0.0
parallel = 6

def gen_dat(n_features = 100, n_observations = 500, n_relevant = 7):
  """generate some random data with a few relevant features"""
  data_val_max = 10
  xmax = 10
  xmin = 0
  data = 1.0*np.random.randint(xmax + 1, size = (n_features, n_observations))
  delta = n_relevant * (xmax - xmin) / 2.0
  labels = np.zeros((n_observations,))
  for m in range(n_observations):
    zz = 0.0
    for k in range(n_relevant):
      zz += data[k, m]
    if zz > delta:
      labels[m] = 1
    else:
      labels[m] = 2
  data = data.transpose()
  return data, labels


def run_npfs(data, labels):
  """run npfs on the sythetic data"""
  mdl = npfs.npfs(fs_method=fs_method, n_select=n_select, n_bootstraps=n_boots, \
      verbose=False, alpha=alpha, beta=beta, parallel=parallel)
  i = mdl.fit(data, labels)
  print mdl.Bernoulli_matrix.sum(axis=1)
  print i 
  mdl.plot_bernoulli_matrix(show_npfs=True)
  return None 

def main():
  """do everything"""
  data, labels = gen_dat()
  run_npfs(data, labels)
  return None 

if __name__ == "__main__":
  main() 
