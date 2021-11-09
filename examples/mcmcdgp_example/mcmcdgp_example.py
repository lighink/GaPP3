#set the python path including gapp and the subpackage covfunctions in GaPP3

import sys
sys.path.insert(0,'/data/tyang/software/GaPP3/') # please set your location where you put GaPP3 directory
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/')
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/covfunctions/')



from gapp import mcmcdgp
import numpy as np
from numpy import loadtxt, random, savetxt, zeros


if __name__=="__main__":
    (X, Y, Sigma) = loadtxt("./inputdata.txt", unpack=True)
    (DX, DY, DSigma) = loadtxt("./dinputdata.txt", unpack=True)

    xmin = 0.0
    xmax = 10.0
    nstar = 50

    nwalker = 32
    theta0 = random.normal(2.0, 0.2, (nwalker, 2))

    g = mcmcdgp.MCMCDGaussianProcess(X, Y, Sigma, theta0, Niter=1000,
                                     dX=DX, dY=DY, dSigma=DSigma,
                                     cXstar=(xmin, xmax, nstar),
                                     threads=50, reclist=[0, 1, 2 ,3])


    result = g.mcmcdgp()
    (Xstar, rec, drec, d2rec, d3rec) = result[0]
    postheta=result[1]


    savetxt("Xstar.txt", Xstar)
    savetxt("rec.txt", rec)
    savetxt("drec.txt", drec)
    savetxt("d2rec.txt", d2rec)
    savetxt("d3rec.txt", d3rec)
    
    pred = zeros((nstar,3))
    pred[:, 0] = Xstar[:, 0]
    pred[:, 1] = np.mean(rec, axis=1)
    pred[:, 2] = np.std(rec, axis=1)

    savetxt("f.txt", pred)


    dpred = zeros((nstar,3))
    dpred[:, 0] = Xstar[:, 0]
    dpred[:, 1] = np.mean(drec, axis=1)
    dpred[:, 2] = np.std(drec, axis=1)

    savetxt("df.txt", dpred)

    d2pred = zeros((nstar,3))
    d2pred[:, 0] = Xstar[:, 0]
    d2pred[:, 1] = np.mean(d2rec, axis=1)
    d2pred[:, 2] = np.std(d2rec, axis=1)

    savetxt("d2f.txt", d2pred)

    d3pred = zeros((nstar,3))
    d3pred[:, 0] = Xstar[:, 0]
    d3pred[:, 1] = np.mean(d3rec, axis=1)
    d3pred[:, 2] = np.std(d3rec, axis=1)

    savetxt("d3f.txt", d3pred)


#a    # test if matplotlib is installed
#    try:
#        import matplotlib.pyplot
#    except:
#        print("matplotlib not installed. no plots will be produced.")
#        exit
#    # create plot
#    import plot
#    plot.plot(X, Y, Sigma, DX, DY, DSigma, pred, dpred)


