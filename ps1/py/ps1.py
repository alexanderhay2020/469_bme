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
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

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

    x = np.random.uniform(low=-10, high=10, size=1000)
    y = 2*x + np.random.normal(0, 1, x.size)

    fig1 = plt.figure()
    plt.plot(x,y,".")
    plt.title("Part A: Dataset")
    plt.xlabel("Random Uniform Distribution: n = 1000, -10 < n < 10")
    plt.ylabel("2*x + e: mu = 0, sigma = 1, size(x)")
    plt.plot(x,y,".")

    return x, y


def part_b():
    """
    Place their centers at -12 to 12 at every .5 along the x axis.
    Set the standard deviation of each RBF to 1.
    """
    rbf_x = np.linspace(-12, 12, 48) # centers at -12 to 12 @ every .5

    rbf_y = np.random.normal(0,1,rbf_x.size)
    rbf = Rbf(rbf_x, rbf_y, function="gaussian")

    xi = np.linspace(-12, 12, 1001)
    yi = rbf(xi)

    fig2 = plt.figure()
    plt.title("Part B: Radial Basis Functions")
    plt.plot(rbf_x, rbf_y, '.')
    plt.plot(xi, yi, 'b')
    plt.xlabel("n = 48, -12 < n < 12")
    plt.ylabel("axis=0, mu = 0, sigma = 1, size(x)")

    return rbf_x, rbf_y


def part_c(x, y, rbf_x, rbf_y):
    """
    text
    """
    rbf_x = np.expand_dims(rbf_x, axis=0) # size: 1x1000 (RxC)

    for i in range(1000):
        """
        neuron
        """
        input = rbf_x # size [1x48]
        weights = np.random.random((48,1000)) # size [48x1000]

        # print ("weights shape: " + str(weights.shape))
        # print ("rbf_x shape: " + str(rbf_x.shape))

        in_w = np.dot(rbf_x, weights) # [1x48]*[48*1000]=[1x1000]
        # print ("xw shape: " + str(xw.shape))

        training_output = y
        output = sigmoid(in_w)

        error = training_output - output

        if (i==0):
            # print ("training output: " + str(training_output.shape))
            print ("output: " + str(output))
            print ("error: " + str(error))
            print

        if (i==999):
            # print ("training output: " + str(training_output.shape))
            print ("output: " + str(output))
            print ("error: " + str(error))
            print

        adjustments = error * sigmoid_derivative(output)

        weights = weights + np.dot(rbf_x.T,adjustments)


def main():
    x, y = part_a()
    rbf_x, rbf_y = part_b()
    part_c(x, y, rbf_x, rbf_y)
    # plt.show()

if __name__ == '__main__':
    main()
