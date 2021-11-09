#set the python path including gapp and the subpackage covfunctions in GaPP3
import sys
sys.path.insert(0,'/data/tyang/software/GaPP3/') # please set your location where you put GaPP3 directory
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/')
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/covfunctions/')

from gapp import mcmcgp
import numpy as np
from numpy import loadtxt, random, savetxt, zeros
import emcee

if __name__=="__main__":
    (X, Y, Sigma) = loadtxt("./inputdata.txt", unpack=True)

    xmin = 0.0
    xmax = 10.0
    nstar = 50

    nwalker = 32
    theta0 = random.normal(2.0, 0.2, (nwalker, 2))

    g = mcmcgp.MCMCGaussianProcess(X, Y, Sigma, theta0, Niter=1000,
                                     cXstar=(xmin, xmax, nstar),
                                     threads=10)



    (Xstar, rec, postheta) = g.mcmcgp()


    savetxt("rec.txt", rec)
    savetxt("Xstar.txt", Xstar)
    savetxt("postheta.txt", postheta)

    pred = zeros((nstar,3))
    pred[:, 0] = Xstar[:, 0]
    pred[:, 1] = np.mean(rec, axis=1)
    pred[:, 2] = np.std(rec, axis=1)

    savetxt("f.txt", pred)


#    dpred = zeros((nstar,3))
#    dpred[:, 0] = Xstar[:, 0]
#    dpred[:, 1] = np.mean(drec, axis=1)
#    dpred[:, 2] = np.std(drec, axis=1)

#    savetxt("df.txt", dpred)


    # test if matplotlib is installed
#    try:
#        import matplotlib.pyplot
#    except:
#        print("matplotlib not installed. no plots will be produced.")
#        exit
#    # create plot
#    import plot
#    plot.plot(X, Y, Sigma, DX, DY, DSigma, pred, dpred)


