import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import loadtxt, reshape, sqrt, sin, cos


def plot(X1, X2, Y, Sigma, x1s, x2s, ys):
    nstar = sqrt(len(x1s))
    x1s = reshape(x1s, (40, 40))
    x2s = reshape(x2s, (40, 40))
    ys = reshape(ys, (40, 40))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X1, X2, Y, color='red')
    surf = ax.plot_surface(x1s, x2s, ys, rstride=1, cstride=1, 
                           linewidth=0, antialiased=False, alpha=0.3)
    ax.set_zlim(-1.01, 1.01)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.show()
    plt.savefig('plot.pdf')

if __name__=="__main__":
    (X1, X2, Y, Sigma) = loadtxt("./2d-inputdata.txt", unpack='True')

    x1s, x2s, ys = loadtxt("f.txt", usecols=(0,1,2), unpack='True')

    plot(X1, X2, Y, Sigma, x1s, x2s, ys)
