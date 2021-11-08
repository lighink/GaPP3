"""
This is an example how to use the gp module of GaPP.
You can run it with 'python gp_example.py'.
"""

#set the python path including gapp and the subpackage covfunctions in GaPP3

import sys
sys.path.insert(0,'/data/tyang/software/GaPP3/') # please set your location where you put GaPP3 directory
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/')
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/covfunctions/')


from gapp import gp
from numpy import loadtxt, savetxt



if __name__=="__main__":
    # load the data from inputdata.txt
    X = loadtxt("./2d-inputdata.txt", usecols=(0,1))
    (Y, Sigma) = loadtxt("./2d-inputdata.txt", usecols=(2,3), unpack=True)

    
    # nstar*nstar points of the function will be reconstructed 
    # on a grid between (xmin, xmin) and (xmax, xmax)
    xmin = 0.0
    xmax = 10.0
    nstar = 40

    # initial values of the hyperparameters
    initheta = [2.0, 2.0]

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
#    plot.plot(X[:,0], X[:,1], Y, Sigma, rec[:,0], rec[:,1], rec[:,2])

