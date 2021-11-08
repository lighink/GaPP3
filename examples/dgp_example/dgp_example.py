"""
This is an example how to use the dgp module of GaPP.
You can run it with 'python dgp_example.py'.
"""

#set the python path including gapp and the subpackage covfunctions in GaPP3

import sys
sys.path.insert(0,'/data/tyang/software/GaPP3/') # please set your location where you put GaPP3 directory
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/')
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/covfunctions/')

from gapp import dgp
from numpy import loadtxt, savetxt



if __name__=="__main__":
    # load the data from inputdata.txt
    (X, Y, Sigma) = loadtxt("./inputdata.txt", unpack=True)
    (DX, DY, DSigma) = loadtxt('./dinputdata.txt', unpack=True)
    # nstar points of the function will be reconstructed 
    # between xmin and xmax
    xmin = 0.0
    xmax = 10.0
    nstar = 200

    # initial values of the hyperparameters
    initheta = [2.0, 2.0]

    # initialization of the Gaussian Process
    g = dgp.DGaussianProcess(X, Y, Sigma, dX=DX, dY=DY, dSigma=DSigma, cXstar=(xmin, xmax, nstar))

    # training of the hyperparameters and reconstruction of the function
    (rec, theta) = g.gp(theta=initheta)

    # reconstruction of the first, second and third derivatives.
    # theta is fixed to the previously determined value.
    (drec, theta) = g.dgp(thetatrain=False)
    (d2rec, theta) = g.d2gp()
    (d3rec, theta) = g.d3gp()

    # save the output
    savetxt("f.txt", rec)
    savetxt("df.txt", drec)
    savetxt("d2f.txt", d2rec)
    savetxt("d3f.txt", d3rec)


