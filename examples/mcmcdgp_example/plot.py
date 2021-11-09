import matplotlib.pyplot as plt
from numpy import loadtxt

def plot(X, Y, Sigma, dX, dY, dSigma, rec, drec, d2rec, d3rec):

    plt.subplot(221)
    plt.xlim(0, 10)
    plt.fill_between(rec[:, 0], rec[:, 1] + rec[:, 2], rec[:, 1] - rec[:, 2],
                     facecolor='lightblue')
    plt.plot(rec[:, 0], rec[:, 1])
    plt.errorbar(X, Y, Sigma, color='red', fmt='_')
    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.subplot(222)
    plt.xlim(0, 10)
    plt.fill_between(drec[:, 0], drec[:, 1] + drec[:, 2], 
                     drec[:, 1] - drec[:, 2], facecolor='lightblue')
    plt.plot(drec[:, 0], drec[:, 1])
    plt.errorbar(dX, dY, dSigma, color='red', fmt='_')
    plt.xlabel('x')
    plt.ylabel("f'(x)")


    plt.subplot(223)
    plt.xlim(0, 10)
    plt.fill_between(d2rec[:, 0], d2rec[:, 1] + d2rec[:, 2], 
                     d2rec[:, 1] - d2rec[:, 2], facecolor='lightblue')
    plt.plot(d2rec[:, 0], d2rec[:, 1])
    plt.xlabel('x')
    plt.ylabel("f''(x)")

    plt.subplot(224)
    plt.xlim(0, 10)
    plt.fill_between(d3rec[:, 0], d3rec[:, 1] + d3rec[:, 2], 
                     d3rec[:, 1] - d3rec[:, 2], facecolor='lightblue')
    plt.plot(d3rec[:, 0], d3rec[:, 1])
    plt.xlabel('x')
    plt.ylabel("f'''(x)")



    plt.savefig('plot.pdf')
if __name__=="__main__":
    (X, Y, Sigma) = loadtxt("./inputdata.txt", unpack=True)
    (dX, dY, dSigma) = loadtxt("./dinputdata.txt", unpack=True)
    rec = loadtxt("f.txt")
    drec = loadtxt("df.txt")
    d2rec = loadtxt("d2f.txt")
    d3rec = loadtxt("d3f.txt")

    plot(X, Y, Sigma, dX, dY, dSigma, rec, drec, d2rec, d3rec)

