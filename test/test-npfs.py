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

data_val_max = 10
n_features = 50
n_observations = 500
n_relevant = 10
n_select = 5
n_boots = 100
fs_method = "MIM"
alpha = 0.01
beta = 0.0
parallel = 2

def gen_dat():
  """generate some random data with a few relevant features"""
  data = np.random.randint(data_val_max, size=(n_observations, n_features)) 
  labels = np.zeros((n_observations,))
  for n, sample in enumerate(data):
    if np.sum(sample[:n_relevant]) >= data_val_max*1.0/2*n_relevant:
      labels[n] = 1
    else:
      labels[n] = 2
  return data, labels

def run_npfs(data, labels):
  """run npfs on the sythetic data"""
  mdl = npfs.npfs(fs_method=fs_method, n_select=n_select, n_bootstraps=n_boots, \
      verbose=False, alpha=alpha, beta=beta, parallel=parallel)
  i = mdl.fit(data, labels)
  print mdl.Bernoulli_matrix
  print i 
  return None 

def main():
  """do everything"""
  data, labels = gen_dat()
  run_npfs(data, labels)
  return None 

if __name__ == "__main__":
  main() 
