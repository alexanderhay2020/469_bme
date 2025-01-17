#!/usr/bin/env python2

import numpy as np
import math
import matplotlib.pyplot as plt


np.random.seed(1)

def function(x):
    """
    mu = 0
    sigma = 1
    """
    y = 2*x + np.random.normal(0, 1, x.size) # [1000x1]

    return y

def part_1a():
    """
    MATLAB CODE:
    x = unifrnd(0,10,1000,1);
    y = 2*x + normrnd(0,1,size(x));
    plot(x,y,'.');

    line-by-line translation with same output, modified to fit the assignment

    returns x, y
    """

    x = np.random.uniform(low=-10, high=10, size=1000) # [1000x1]
    y = function(x) # [1000x1]

    fig1 = plt.figure()
    plt.plot(x, y, ".",label="sample points")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part A: Dataset")
    plt.xlabel("Random Uniform Distribution: -10 < n < 10")
    # plt.ylabel("2*x + e: mu = 0, sigma = 1,q size(x)")
    plt.legend()

    return x, y # [1000x1], [1000x1]


def part_1b(x):
    """
    Place their centers at -12 to 12 at every .5 along the x axis.
    Set the standard deviation of each RBF to 1.

    returns radial basis function h(x)
    """
    sigma = 1 # standard deviation
    x_u = np.linspace(-12, 12, 48) # centers at -12 to 12 @ every .5 [48x1]

    h = np.zeros((len(x),len(x_u))) # [1000x48]
    for j in range(len(x_u)): # 1-48
        for i in range(len(x)): # 1-1000
            h[i,j] = math.exp((-abs(x[i]-x_u[j])**2)/sigma**2)

    fig2 = plt.figure()
    plt.plot(x_u, h.T)#,":")
    plt.title("Part B: Radial Basis Functions")
    plt.xlabel("n = 48, -12 < n < 12")
    plt.ylabel("axis=0, mu = 0, sigma = 1, size(x)")
    plt.ylim(bottom = 0)
    plt.ylim(top = 1)
    # plt.legend()

    return h # [1000x48]


def part_1c(y, h):
    """
    calculated weight vector w using linear regression

    returns weight vector
    """
    # w = np.random.random((48)) # starting weights [48x1]
    w = np.dot(np.linalg.inv(np.dot(h.T,h)),np.dot(h.T,y))

    return w


def part_1d(x, y, h, w):
    """
    plots predicted f(x) given x, h(x), and w
    """
    f_x = np.random.random((1000))

    for i in range(len(f_x)):
        f_x[i] = sum(h[i]*w)

    fig3 = plt.figure()
    plt.plot(x, y, ".", label="true y values; n=1,000")
    plt.plot(x, f_x, ".", label="predicted y values; n=1,000")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part D: RBF Predictions")
    plt.legend()


def part_1e(x, y):
    """
    adds pertubation, simulated as 50 points at x=6
    creates new vectors of length 50, at x=6 and y=2x
    appends to existing x, y vectors

    returns x, y
    returns six_x, six_y for future graphing
    """
    six_x = np.arange(50, dtype=int) # creates a vector of length 50
    six_x = np.full_like(six_x, 6)
    six_y = 2*six_x + 10 + np.random.normal(0, 1, six_x.size) # [1050x1]

    x = np.append(x, six_x)
    y = np.append(y, six_y)

    fig4 = plt.figure()
    plt.plot(x, y, ".", label="sample points")
    plt.plot(six_x, six_y, 'g.', label="added points")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part E: Pertubation")
    plt.legend()

    return x, y, six_x, six_y


def part_1f(x, y, six_x, six_y):
    """
    plots predicted f(x) given x, h(x), and w

    returns new f(x)
    """
    sigma = 1 # standard deviation
    x_u = np.linspace(-12, 12, 48) # centers at -12 to 12 @ every .5 [48x1]

    h = np.zeros((len(x),len(x_u))) # [1000x48]
    for j in range(len(x_u)): # 1-48
        for i in range(len(x)): # 1-1000
            h[i,j] = math.exp((-abs(x[i]-x_u[j])**2)/sigma**2)

    w = np.dot(np.linalg.inv(np.dot(h.T,h)),np.dot(h.T,y))

    f_x = np.random.random((1050))

    for i in range(len(f_x)):
        f_x[i] = sum(h[i]*w)

    fig5 = plt.figure()
    plt.plot(x, y, ".", label="sample points; n=1,050")
    plt.plot(x, f_x, ".", label="predicted y values; n=1,050")
    plt.plot(six_x, six_y, '.', label="added points")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part F: RBF Predictions w/ Pertubation")
    plt.legend()

    return f_x


def part_1g(x, f_x):
    """
    finds mean error of all 1,050 samples
    normalizes values as percentile of normal distribution (mu = mean error, sigma = 1)
    """
    sigma = 1
    y = 2*x
    error = (y-f_x)**2
    mean_error = sum(error)/len(f_x)
    mean_error_arr = np.arange(len(x), dtype=int)
    mean_error_arr = np.full_like(mean_error_arr, mean_error)

    percent_error = (error-mean_error)/100

    fig6 = plt.figure()
    plt.plot(percent_error, label="percent error")
    plt.plot(mean_error_arr, 'r.', label="mean error = 0.044")
    plt.title("Part G: Error Analysis")
    plt.xlabel("samples; n=1,050")
    plt.ylabel("error normalized")
    plt.ylim(-0.1,1.0)
    plt.legend()


def main():
    x, y = part_1a()
    h = part_1b(x)
    w = part_1c(y, h)
    part_1d(x, y, h, w)
    x, y, six_x, six_y= part_1e(x, y)
    f_x = part_1f(x, y, six_x, six_y)
    part_1g(x, f_x)
    plt.show()

if __name__ == '__main__':
    main()
