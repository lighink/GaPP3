# GaPP3
The updated version of GaPP (Gaussian Process in Python) with Python3

For the original version of GaPP please refere to Marina Seikel, Chris Clarkson, Mathew Smith, Reconstruction of dark energy and expansion dynamics using Gaussian processes, arXiv:1204.2832.


To use GaPP3,  first use miniconda to build an envirenment with python=3.X, numpy, scipy, matplotlib, emcee, corner. Then in the python script we include the GaPP3 directory into PYTHONPATH.

import sys

sys.path.insert(0,'/data/tyang/software/GaPP3/')
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/')
sys.path.insert(0,'/data/tyang/software/GaPP3/gapp/covfunctions/')

Please replace the location with your own location of GaPP3.

For examples please refere to the examples directory.
