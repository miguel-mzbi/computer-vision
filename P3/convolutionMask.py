import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from mpl_toolkits import mplot3d

def h(x, variance):
    exp = math.e**(-x**2/(2*variance))
    base = 1/math.sqrt(2*math.pi*variance)
    total = base * exp
    return (total, (base, exp))

def h2(x, y, variance):
    exp = math.e**(-(x**2+y**2)/(2*variance))
    base = 1/math.sqrt(2*math.pi*variance)
    total = base * exp
    return (total, (base, exp))

if __name__ == "__main__":
    variance = 1

    # gau1D = []
    # for x in range(-3, 4):
    #     total, (base, exp) = h(x, variance)
    #     gau1D.append(exp)
    # for i in range(7):
    #     row = []
    #     for j in range(7):
    #         row.append(gau1D[i]*gau1D[j])
    #     print(row)
    
    # gau = []
    # exp = []
    # for i in range(-4, 5):
    #     rowGau = []
    #     rowExp = []
    #     for j in range(-4, 5):
    #         z, (_, e) = h2(i, j, variance)
    #         rowGau.append(z)
    #         rowExp.append(e)
    #     gau.append(rowGau)
    #     exp.append(rowExp)

    x = np.linspace(-4, 4, 9)
    y = np.linspace(-4, 4, 9)
    X, Y = np.meshgrid(x, y)
    gau, (b, expsGau) = h2(X, Y, variance)
    print(b)
    print(DataFrame(gau))
    print(DataFrame(expsGau))

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z, (_, exps) = h2(X, Y, variance)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Gaussian Filter Convolution Mask');

    plt.figure()
    plt.imshow(Z, cmap='hot', interpolation='nearest', extent=[-4,4,-4,4])

    plt.figure()
    plt.imshow(gau, cmap='hot', interpolation='nearest', extent=[-4,4,-4,4])

    plt.show()

        
    # print(gau2D)
