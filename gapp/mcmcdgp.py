
import dgp, covariance
import numpy as np
from numpy import array, concatenate, ones, random, reshape, shape, zeros
import multiprocessing
import warnings
import emcee

import os

os.environ["OMP_NUM_THREADS"] = "1"





def mcmc_log_likelihood(th, sc0, scl, X, Y_mu, Sigma, dX, dY_dmu, dSigma, 
                        covfunction, prior, priorargs):
    try:
        if (np.min(th) < 0.0):
            return np.NINF
        if (sc0 == False):
            theta = th
            scale = 1
        else:
            theta = th[: -scl]
            scale = th[-scl :]
        g = dgp.DGaussianProcess(X, Y_mu, Sigma, covfunction, theta, 
                                 dX, dY_dmu, dSigma, 
                                 prior=prior, priorargs=priorargs, 
                                 scale=scale)
        logp = g.log_likelihood()
        return logp
    except KeyboardInterrupt:
        return


def recthread(i, th, sc0, scl, X, Y, Sigma, dX, dY, dSigma, covfunction, Xstar,
              mu, dmu, d2mu, d3mu, muargs, reclist):
    try:
        if (sc0 == False):
            theta = th
            scale = 1
        else:
            theta = th[:-scl]
            scale = th[-scl:]
        g = dgp.DGaussianProcess(X, Y, Sigma, covfunction, theta, dX, dY, 
                                  dSigma, Xstar, mu=mu, dmu=dmu, d2mu=d2mu, 
                                  d3mu=d3mu, muargs=muargs, thetatrain=False, 
                                  scale=scale, scaletrain=False)
        nr = 0
        nstar = len(Xstar)
        pred = zeros((nstar, 2, len(reclist)))
        if (0 in reclist):
            (fmean, fstd) = g.gp(unpack=True)[1:3]
            pred[:, 0, nr] = fmean[:]
            pred[:, 1, nr] = fstd[:]
            nr += 1
        if (1 in reclist):
            (fmean, fstd) = g.dgp(unpack=True)[1:3]
            pred[:, 0, nr] = fmean[:]
            pred[:, 1, nr] = fstd[:]
            nr += 1
        if (2 in reclist):
            (fmean, fstd) = g.d2gp(unpack=True)[1:3]
            pred[:, 0, nr] = fmean[:]
            pred[:, 1, nr] = fstd[:]
            nr += 1
        if (3 in reclist):
            (fmean, fstd) = g.d3gp(unpack=True)[1:3]
            pred[:, 0, nr] = fmean[:]
            pred[:, 1, nr] = fstd[:]
            nr += 1
        if (len(reclist) > 1):
            fc = g.f_covariances(reclist)
        else:
            fc = None
        return (i, pred, fc)
    except KeyboardInterrupt:
        return


def recarray(j, recj, fcovj, k, nsample):
    try:
        if (fcovj is not None):
            rarr = random.multivariate_normal(recj[0, 0, :], 
                                              fcovj[0, :, :], 
                                              k[0] * nsample)
            for i in range(1, len(recj)):
                rs = random.multivariate_normal(recj[i, 0, :], 
                                                fcovj[i, :, :], 
                                                k[i] * nsample)
                rarr = concatenate((rarr, rs))
        else:
            rarr = array([])
            for i in range(len(recj)):
                if (recj[i, 1, 0] > 0):
                    rarr = concatenate((rarr, random.normal(recj[i, 0, 0], 
                                                            recj[i, 1, 0], 
                                                            k[i] * nsample)))
                else:
                    rarr = concatenate((rarr, recj[i, 0, 0] * 
                                        ones(k[i] * nsample)))

        return (j, rarr)
    except KeyboardInterrupt:
        return





class MCMCDGaussianProcess(dgp.DGaussianProcess):
    def __init__(self, X, Y, Sigma, theta0, Niter=100, reclist=[0],
                 covfunction=covariance.SquaredExponential,
                 dX=None, dY=None, dSigma=None, Xstar=None, cXstar=None, 
                 mu=None, dmu=None, d2mu=None, d3mu=None, muargs=(), prior=None, 
                 priorargs=(), scale0=None, threads=1,
                 nsample=50, sampling=True):

        if (scale0 is not None):
            assert (len(theta0) == len(scale0)) ,\
                "Lengths of theta0 and scale0 must be identical."
            self.sc0 = True
            scale = scale0[0]
            self.scl = len(scale)
            if (self.scl == 2 and dX is None):
                scale0 = scale0[:, 0]
                scale = scale0[0]
                self.scl = 1
                warnings.warn("scale0 is two-dimensional, but dX=None. " +
                              "Second dimension of scale0 will be ignored.")
            self.pos = concatenate((theta0, reshape(scale0, (len(scale0), self.scl))), 
                                   axis=1)
        else:
            self.pos = theta0
            self.sc0 = False
            scale = None
            self.scl = None

        dgp.DGaussianProcess.__init__(self, X, Y, Sigma, covfunction, 
                                      theta0[0,:], dX, dY, dSigma, Xstar, cXstar, 
                                      mu, dmu, d2mu, d3mu, muargs,
                                      prior, gradprior=None, priorargs=priorargs,
                                      thetatrain=False, scale=scale, 
                                      scaletrain=False)

        self.theta0 = theta0
        self.scale0 = scale0
        self.covfunction = covfunction
        self.reclist = reclist
        self.Niter = Niter
        self.threads = threads
        self.nsample = nsample
        self.sampling = sampling
        (self.nwalkers, self.ndim) = shape(self.pos)

        if (sampling == True and self.threads == 1):

            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                                 mcmc_log_likelihood, 
                                                 args=(self.sc0, self.scl, self.X, 
                                                       self.Y_mu, self.Sigma, 
                                                       self.dX, self.dY_dmu,
                                                       self.dSigma, covfunction,
                                                       prior, priorargs))

        elif (sampling ==True and self.threads != 1):
            self.pool = multiprocessing.Pool(processes = self.threads)
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                                 mcmc_log_likelihood, pool=self.pool,
                                                 args=(self.sc0, self.scl, self.X, 
                                                       self.Y_mu, self.Sigma, 
                                                       self.dX, self.dY_dmu,
                                                       self.dSigma, covfunction,
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
            self.scale0 = pos[:, -self.scl:]
            self.theta0 = pos[:, :-self.scl]
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
            self.thetasample = self.possample[:, :-self.scl]
            self.scalesample = self.possample[:, -self.scl:]


    def mcmcdgp(self):
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
            reconstrarr = self.serialrec(redpossample, k)
        else:
            reconstrarr = self.parallelrec(redpossample, k)
        nr = 0
        retvalue = [self.Xstar]
        if (0 in self.reclist):
            self.reconstruction = reconstrarr[:, :, nr]
            retvalue.append(self.reconstruction)
            nr += 1
        if (1 in self.reclist):
            self.dreconstruction = reconstrarr[:, :, nr]
            retvalue.append(self.dreconstruction)
            nr += 1
        if (2 in self.reclist):
            self.d2reconstruction = reconstrarr[:, :, nr]
            retvalue.append(self.d2reconstruction)
            nr += 1
        if (3 in self.reclist):
            self.d3reconstruction = reconstrarr[:, :, nr]
            retvalue.append(self.d3reconstruction)
            nr += 1
        return (retvalue, self.possample)



    def serialrec(self, redpossample, k):
        nrl = len(self.reclist)
        rec = zeros((len(redpossample), self.nstar, 2, nrl))
        fcov = zeros((len(redpossample), self.nstar, nrl, nrl))
        for i in range(len(redpossample)):
            nr = 0
            if (0 in self.reclist):
                if (self.sc0 == False):
                    self.set_theta(redpossample[i, :])
                else:
                    self.set_theta(redpossample[i, :-self.scl])
                    self.set_scale(redpossample[i, -self.scl:])
                (fmean, fstd) = self.gp(unpack=True)[1:3]
                rec[i, :, 0, nr] = fmean[:]
                rec[i, :, 1, nr] = fstd[:]
                nr += 1
            if (1 in self.reclist):
                if (self.sc0 == False):
                    self.set_theta(redpossample[i, :])
                else:
                    self.set_theta(redpossample[i, :-self.scl])
                    self.set_scale(redpossample[i, -self.scl:])
                (fmean, fstd) = self.dgp(unpack=True)[1:3]
                rec[i, :, 0, nr] = fmean[:]
                rec[i, :, 1, nr] = fstd[:]
                nr += 1
            if (2 in self.reclist):
                if (self.sc0 == False):
                    self.set_theta(redpossample[i, :])
                else:
                    self.set_theta(redpossample[i, :-self.scl])
                    self.set_scale(redpossample[i, -self.scl:])
                (fmean, fstd) = self.d2gp(unpack=True)[1:3]
                rec[i, :, 0, nr] = fmean[:]
                rec[i, :, 1, nr] = fstd[:]
                nr += 1
            if (3 in self.reclist):
                if (self.sc0 == False):
                    self.set_theta(redpossample[i, :])
                else:
                    self.set_theta(redpossample[i, :-self.scl])
                    self.set_scale(redpossample[i, -self.scl:])
                (fmean, fstd) = self.d3gp(unpack=True)[1:3]
                rec[i, :, 0, nr] = fmean[:]
                rec[i, :, 1, nr] = fstd[:]
                nr += 1
            if (nrl > 1):
                fcov[i, :, :, :] = self.f_covariances(self.reclist)
        reconstrarr = zeros((self.nstar, len(self.possample) * self.nsample, nrl))
        for j in range(self.nstar):
            if (nrl > 1):
                rarr = random.multivariate_normal(rec[0, j, 0, :], 
                                                  fcov[0, j, :, :], 
                                                  k[0] * self.nsample)
                for i in range(1, len(rec)):
                    rs = random.multivariate_normal(rec[i, j, 0, :], 
                                                    fcov[i, j, :, :], 
                                                    k[i] * self.nsample)
                    rarr = concatenate((rarr, rs))
                reconstrarr[j, :, :] = rarr[:, :]
            else:
                rarr = array([])
                for i in range(len(rec)):
                    if (rec[i, j, 1, 0] > 0):
                        rarr = concatenate((rarr, random.normal(rec[i, j, 0, 0], 
                                                                rec[i, j, 1, 0], 
                                                                k[i] * 
                                                                self.nsample)))
                    else:
                        rarr = concatenate((rarr, rec[i, j, 0, 0] * 
                                            ones(k[i] * self.nsample)))
                reconstrarr[j, :, 0] = rarr[:]
        return reconstrarr




    def parallelrec(self, redpossample, k):
        pool = multiprocessing.Pool(processes=self.threads)
        recres = [pool.apply_async(recthread, (i, redpossample[i, :], self.sc0, 
                                               self.scl, self.X, self.Y, 
                                               self.Sigma, self.dX, self.dY, 
                                               self.dSigma, self.covfunction, 
                                               self.Xstar, self.mu,  self.dmu, 
                                               self.d2mu, self.d3mu, 
                                               self.muargs, self.reclist)) 
                  for i in range(len(redpossample))]
        nrl = len(self.reclist)
        rec = zeros((len(redpossample), self.nstar, 2, nrl))
        if (nrl > 1):
            fcov = zeros((len(redpossample), self.nstar, nrl, nrl))
            for r in recres:
                a = r.get()
                rec[a[0], :, :, :] = a[1]
                fcov[a[0], :, :, :] = a[2]
        else:
            for r in recres:
                a = r.get()
                rec[a[0], :, :, :] = a[1]
        reconstrarr = zeros((self.nstar, len(self.possample) * self.nsample, 
                            nrl))
        if (nrl > 1):
            recon = [pool.apply_async(recarray, (j, rec[:, j, :, :], 
                                                 fcov[:, j, :, :], 
                                                 k, self.nsample)) 
                     for j in range(self.nstar)]
        else:
            recon = [pool.apply_async(recarray, (j, rec[:, j, :, :], 
                                                 None, 
                                                 k, self.nsample)) 
                     for j in range(self.nstar)]
        if (nrl > 1):
            for r in recon:
                a = r.get()
                reconstrarr[a[0], :, :] = a[1]
        else:
            for r in recon:
                a = r.get()
                reconstrarr[a[0], :, 0] = a[1]
        pool.close()
        pool.join()
        return reconstrarr


