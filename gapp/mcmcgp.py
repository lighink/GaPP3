
import gp, covariance
import numpy as np
from numpy import array, concatenate, ones, random, reshape, shape, zeros
import multiprocessing
import emcee

import os

os.environ["OMP_NUM_THREADS"] = "1"





def mcmc_log_likelihood(th, sc0,  X, Y_mu, Sigma, covfunction, prior, 
                        priorargs):
    try:
        if (np.min(th) < 0.0):
            return np.NINF
        if (sc0 == False):
            theta = th
            scale = 1
        else:
            theta = th[:-1]
            scale = th[-1]
        g = gp.GaussianProcess(X, Y_mu, Sigma, covfunction, theta, prior=prior, 
                               priorargs=priorargs, scale=scale)
        logp = g.log_likelihood()
        return logp
    except KeyboardInterrupt:
        return


def recthread(i, th, sc0, X, Y, Sigma, covfunction, Xstar, mu, muargs):
    try:
        if (sc0 == False):
            theta = th
            scale = 1
        else:
            theta = th[:-1]
            scale = th[-1]
        g = gp.GaussianProcess(X, Y, Sigma, covfunction, theta, Xstar, mu=mu, 
                               muargs=muargs, thetatrain=False, scale=scale, 
                               scaletrain=False)
        nstar = len(Xstar)
        (fmean, fstd) = g.gp(unpack=True)[1:3]
        pred = concatenate((reshape(fmean, (nstar, 1)), 
                            reshape(fstd, (nstar, 1))), axis=1)
        return (i, pred)
    except KeyboardInterrupt:
        return

def recarray(j, recj, k, nsample):
    try:
        rarr = array([])
        for i in range(len(recj)):
            if (recj[i, 1] > 0):
                rarr = concatenate((rarr, random.normal(recj[i, 0], 
                                                        recj[i, 1], 
                                                        k[i] * nsample)))
            else:
                rarr = concatenate((rarr, recj[i, 0] * ones(k[i] * nsample)))
        return (j, rarr)
    except KeyboardInterrupt:
        return


class MCMCGaussianProcess(gp.GaussianProcess):
    def __init__(self, X, Y, Sigma, theta0, Niter=100,
                 covfunction=covariance.SquaredExponential,
                 Xstar=None, cXstar=None, mu=None, muargs=(), prior=None, 
                 priorargs=(), scale0=None, threads=1,
                 nsample=50, sampling=True):


        if (scale0 is not None):
            assert (len(theta0) == len(scale0)) ,\
                "Lengths of theta0 and scale0 must be identical."
            self.pos = concatenate((theta0, reshape(scale0, (len(scale0), 1))), 
                                   axis=1)
            self.sc0 = True
            scale = scale0[0]
        else:
            self.pos = theta0
            self.sc0 = False
            scale = None

        gp.GaussianProcess.__init__(self, X, Y, Sigma, covfunction, 
                                    theta0[0,:], Xstar, cXstar, mu, muargs,
                                    prior, gradprior=None, priorargs=priorargs,
                                    thetatrain=False, scale=scale, 
                                    scaletrain=False)
        self.theta0 = theta0
        self.scale0 = scale0
        self.covfunction = covfunction
        self.Niter = Niter
        self.threads = threads
        self.nsample = nsample
        self.sampling = sampling
        (self.nwalkers, self.ndim) = shape(self.pos)

        if (sampling == True and self.threads == 1):
            
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                                 mcmc_log_likelihood, 
                                                 args=(self.sc0, self.X, self.Y_mu, 
                                                       self.Sigma, covfunction,
                                                       prior, priorargs))
        elif (sampling ==True and self.threads != 1):
            self.pool = multiprocessing.Pool(processes = self.threads)
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                                 mcmc_log_likelihood, pool = self.pool,
                                                 args=(self.sc0, self.X, self.Y_mu, 
                                                       self.Sigma, covfunction,
                                                       prior, priorargs))



    def mcmc_sampling(self):
        print("start burn-in")
        (pos, prob, state) = self.sampler.run_mcmc(self.pos, 100, progress=True)
        c = False
        try:
            tau = self.sampler.get_autocorr_time()
        except (emcee.autocorr.AutocorrError):
            c = True
        while (c):
            (pos, prob, state) = self.sampler.run_mcmc(pos, 100, rstate0=state, progress=True)
            print(prob)
            try:
                tau = self.sampler.get_autocorr_time()
            except (emcee.autocorr.AutocorrError):
                pass
            else:
                c = False
        if (self.sc0 == False):
            self.theta0 = pos
        else:
            self.scale0 = pos[:, -1]
            self.theta0 = pos[:, :-1]
        print("burn-in finished")
        print("number of burn-in steps: " + str(shape(self.sampler.chain)[1]))
        print("autocorrelation time: " + str(tau))
        print("acceptance fraction: " + 
              str(np.mean(self.sampler.acceptance_fraction)))
        self.sampler.reset()
        (pos, prob, state) = self.sampler.run_mcmc(pos, self.Niter, 
                                                   rstate0=state, progress=True)

        self.possample = self.sampler.get_chain(flat=True, thin=10)
        if (self.threads != 1):
            self.pool.close()
            self.pool.join()
        if (self.sc0 == False):
            self.thetasample = self.possample
            self.scalesample = None
        else:
            self.thetasample = self.possample[:, :-1]
            self.scalesample = self.possample[:, -1]


    def mcmcgp(self):
        if (self.sampling == True):
            self.mcmc_sampling()
        else:
            self.possample = self.pos
            self.thetasample = self.theta0
            self.scalesample = self.scale0
        redpossample = []
        k = []
        for i in range(len(self.possample)):
            if (i > 0 and all(self.possample[i, :] == self.possample[i-1, :])):
                k[-1] += 1
            else:
                redpossample.append(self.possample[i, :])
                k.append(1)
        redpossample = array(redpossample)
        if (self.threads == 1):
            self.serialrec(redpossample, k)
        else:
            self.parallelrec(redpossample, k)
        return (self.Xstar, self.reconstruction, self.possample)



    def serialrec(self, redpossample, k):
        rec = zeros((len(redpossample), self.nstar, 2))
        for i in range(len(redpossample)):
            if (self.sc0 == False):
                self.set_theta(redpossample[i, :])
            else:
                self.set_theta(redpossample[i, :-1])
                self.set_scale(redpossample[i, -1])
            (fmean, fstd) = self.gp(unpack=True)[1:3]
            rec[i, :, 0] = fmean[:]
            rec[i, :, 1] = fstd[:]
        reconstruction = zeros((self.nstar, len(self.possample) * self.nsample))
        for j in range(self.nstar):
            rarr = array([])
            for i in range(len(rec)):
                if (rec[i, j, 1] > 0):
                    rarr = concatenate((rarr, random.normal(rec[i, j, 0], 
                                                            rec[i, j, 1], 
                                                            k[i] * self.nsample)))
                else:
                    rarr = concatenate((rarr, rec[i, j, 0] * 
                                        ones(k[i] * self.nsample)))
            reconstruction[j, :] = rarr[:]
        self.reconstruction = reconstruction



    def parallelrec(self, redpossample, k):
        pool = multiprocessing.Pool(processes=self.threads)
        recres = [pool.apply_async(recthread, (i, redpossample[i, :], self.sc0, 
                                               self.X, self.Y, self.Sigma, 
                                               self.covfunction, self.Xstar, 
                                               self.mu, self.muargs)) 
                  for i in range(len(redpossample))]
        rec = zeros((len(redpossample), self.nstar, 2))
        for r in recres:
            a = r.get()
            rec[a[0], :, :] = a[1]
        reconstruction = zeros((self.nstar, len(self.possample) * self.nsample))
        recon = [pool.apply_async(recarray, (j, rec[:, j, :], k, self.nsample)) 
                 for j in range(self.nstar)]
        for r in recon:
            a = r.get()
            reconstruction[a[0], :] = a[1]
        pool.close()
        pool.join()
        self.reconstruction = reconstruction

