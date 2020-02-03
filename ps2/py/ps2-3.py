#!/usr/bin/env python2

"""
Unsupervised Learning - ML Gradient Descent
Autoencoder
"""

import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1)

def sigmoid(x):
    """
    args: x - some number

    return: some value between 0 and 1 based on sigmoid function
    """

    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    """
    args: x - some number

    return: derivative of sigmiod given x
    """
    return sigmoid(x)*(1-sigmoid(x))


def main():
    """
    x1 = [normrnd(0,1,150,1); normrnd(7,2,150,1)];    % the data clusters
    x2 = [normrnd(0,1,150,1); normrnd(7,2,150,1)];
    input = [x1 x2];
    nsamp = length(input);

    w1 = mean(input) + normrnd(-1,1,1,2);            %initialize the means and standard deviations
    w2 = mean(input) + normrnd(-1,1,1,2);
    s1 = sqrt(mean(var(input)));
    s2 = sqrt(mean(var(input)));
    pi1 = .5;                                      % the prior probabilities - we're not fitting those here
    pi2 = .5;
    """

    # x1 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];    % the data clusters
    x1 = np.random.normal(0, 1, 150)                             # (mean, std_dev, size))
    x1 = np.append(x1, np.random.normal(7, 2, 150))
    x1 = np.expand_dims(x1, axis=1)                                 # size (150,) to (150,1)

    # x2 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];
    x2 = np.random.normal(0, 1, 150)                             # (mean, std_dev, size))
    x2 = np.append(x2, np.random.normal(7, 2, 150))
    x2 = np.expand_dims(x2, axis=1)                                 # size (150,) to (150,1)

    # input = [x1 x2];
    input = np.append(x1, x2, axis=1)

    # nsamp = length(input);
    nsamp = len(input)

    # w1 = mean(input) + normrnd(-1,1,1,2);   % intiialize the weights somewhere in the center of the data
    w1 = np.mean(input) + np.random.normal(-1, 1, (1,2))
    w2 = np.mean(input) + np.random.normal(-1, 1, (1,2))
    s1 = math.sqrt(np.mean(np.var(input)))
    s2 = math.sqrt(np.mean(np.var(input)))
    pi1 = 0.5
    pi2 = 0.5


    fig0 = plt.figure()
    plt.plot(w1.T[0], w1.T[1], "rx", label="group 1 initial mean")
    plt.plot(w2.T[0], w2.T[1], "bx", label="group 2 initial mean")
    plt.plot(input.T[0,:150], input.T[1,:150], "r.", label="group 1")
    plt.plot(input.T[0,150:], input.T[1,150:], "b.", label="group 2")
    plt.title("Part 3: ML Gradient Descent Initial Data")
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
