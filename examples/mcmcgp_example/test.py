import sys
sys.path.insert(0,'/Users/lighink/software/gapp_dev/')
sys.path.insert(0,'/Users/lighink/software/gapp_dev/gapp/')
sys.path.insert(0,'/Users/lighink/software/gapp_dev/gapp/covfunctions/')


import gp, covariance
import numpy as np
from numpy import loadtxt,array, concatenate, ones, random, reshape, shape, zeros





def mcmc_log_likelihood(X, Y, Sigma, th):
    g = gp.GaussianProcess(X, Y, Sigma, theta=th)
    logp = g.log_likelihood()
    return logp

th=[2,2]
(X, Y, Sigma) = loadtxt("./inputdata.txt", unpack=True)

print(mcmc_log_likelihood(X, Y, Sigma, th))


