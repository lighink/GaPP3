"""
This is an example how to use the gp module of GaPP.
You can run it with 'python gp_example.py'.
"""

#set the python path including gapp and the subpackage covfunctions in GaPP3 

import sys
sys.path.insert(0,'/data/tyang/software/GaPP3/') # please set your location where you put GaPP3 directory
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/')
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/covfunctions/')
#print(sys.path)

from gapp import gp
from numpy import loadtxt, savetxt




if __name__=="__main__":
    # load the data from inputdata.txt
    (X,Y,Sigma) = loadtxt("./inputdata.txt", usecols=(0,1,2),unpack=True)

    
    # nstar*nstar points of the function will be reconstructed 
    # on a grid between (xmin, xmin) and (xmax, xmax)
    xmin = 0.0
    xmax = 10.0
    nstar = 40

    # initial values of the hyperparameters
    initheta = [2, 2]

    # initialization of the Gaussian Process
    g = gp.GaussianProcess(X, Y, Sigma, cXstar=(xmin, xmax, nstar))

    # training of the hyperparameters and reconstruction of the function
    (rec, theta) = g.gp(theta=initheta)

    # save the output
    savetxt("f.txt", rec)



#    # test if matplotlib is installed
#    try:
#        import matplotlib.pyplot
#    except:
#        print("matplotlib not installed. no plots will be produced.")
#        exit
#    # create plot
#    import plot
#    plot.plot(X, Y, Sigma, rec)



