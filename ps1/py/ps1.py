# %% data for problem #1a
# clear
#
# a = 0
# b = 10
# m = 1000
# n = 1
#
# generate mXn matrix of continuous uniform
# random values between a and b
#
# x = unifrnd(a,b,m,n);
# e = normrnd(0,1,size(x));
# y = 2*x
# plot(x,y,'.');

import numpy as np
import math
import logging
import matplotlib.pyplot as plt

from scipy.interpolate import Rbf, InterpolatedUnivariateSpline


np.random.seed(1)

def sigmoid(x):
    """
    args: x - some number
    return: some value between 0 and 1 based on sigmoid function
    """
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid

def sigmoid_derivative(x):
    """
    args: x - some number
    return: derivative of sigmoid given x
    """
    sig_prime = x*(1-x)
    return sig_prime

def part_a():
    """
    MATLAB CODE:
    x = unifrnd(0,10,1000,1);
    y = 2*x + normrnd(0,1,size(x));
    plot(x,y,'.');

    line-by-line translation with same output, modified to fit the assignment
    """

    x = np.random.uniform(low=-10, high=10, size=1000) # [1000x1]
    y = 2*x + np.random.normal(0, 1, x.size) # [1000x1]

    fig1 = plt.figure()
    plt.plot(x,y,".",label="n = 1,000")
    plt.title("Part A: Dataset")
    plt.xlabel("Random Uniform Distribution: -10 < n < 10")
    plt.ylabel("2*x + e: mu = 0, sigma = 1,q size(x)")
    plt.plot(x,y,".")
    plt.legend()

    return x, y # [1000x1], [1000x1]


def part_b(x):
    """
    Place their centers at -12 to 12 at every .5 along the x axis.
    Set the standard deviation of each RBF to 1.
    """
    sigma = 1 # standard deviation
    x_u = np.linspace(-12, 12, 48) # centers at -12 to 12 @ every .5 [48x1]

    h = np.zeros((1000,48)) # [1000x48]
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

def part_c(x, y, h):
    """
    x    [1000x1]
    y    [1000x1]
    h    [1000x48]
    f(x) [1000,1]
    """
    # w = np.random.random((48)) # starting weights [48x1]

    # f_x = np.zeros((1000))
    # f_x = np.dot(w, h.T)
    # error = f_x - y
    # for i in range(1,1000):
    #     sum = sum + np.dot(h, w)
    # print f_x.shape

    # for i in range(48):
        # print h[i,:].shape
        # f_x = f_x + np.dot(h[i,:], w)
        # print f_x.shape
    # for i in range(1000):
    #     f_x = np.dot(w, h.T) #[1000x1] = [1000x48]*[48x1]
    #     f_x = sigmoid(f_x)
    #     error = y - f_x
    #     adjustments = error * sigmoid_derivative(f_x)
    #     w = w + np.dot(h.T, adjustments)

    w = np.dot((1/np.dot(h.T,h)),np.dot(h.T,y))
    print w.shape
    f_x = np.dot(h,w)

    fig3 = plt.figure()
    x2=x
    plt.plot(x, y, ".", label="true y values; n=1,000")
    plt.plot(x2, f_x, ".", label="predicted y values; n=1,000")

    plt.title("Part C: Weight Vectors")
    plt.legend()

    # plt.ylim(bottom = -20)
    # plt.ylim(top = 20)
    # plt.plot(error)
    # print f_x.shape



def main():
    x, y = part_a()
    h = part_b(x)
    part_c(x, y, h)
    plt.show()

if __name__ == '__main__':
    main()
