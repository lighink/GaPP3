import matplotlib.pyplot as plt
from numpy import loadtxt

def plot(X, Y, Sigma, rec):

    plt.subplot(211)
    plt.xlim(0, 10)
    plt.fill_between(rec[:, 0], rec[:, 1] + rec[:, 2], rec[:, 1] - rec[:, 2],
                     facecolor='lightblue')
    plt.plot(rec[:, 0], rec[:, 1])
    plt.errorbar(X, Y, Sigma, color='red', fmt='_')
    plt.xlabel('x')
    plt.ylabel('f(x)')

#    plt.subplot(212)
#    plt.xlim(0, 10)
#    plt.fill_between(drec[:, 0], drec[:, 1] + drec[:, 2], 
#                     drec[:, 1] - drec[:, 2], facecolor='lightblue')
#    plt.plot(drec[:, 0], drec[:, 1])
#    plt.errorbar(dX, dY, dSigma, color='red', fmt='_')
#    plt.xlabel('x')
#    plt.ylabel("f'(x)")

    plt.savefig('plot.pdf')


if __name__=="__main__":
    (X, Y, Sigma) = loadtxt("./inputdata.txt", unpack=True)
    rec = loadtxt("f.txt")

    plot(X, Y, Sigma, rec)

