#!/usr/bin/env python2

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
    %% non-linearly separable classification - back propagation

    sd = .85;

    x1 = [normrnd(0,sd,50,1); normrnd(0,sd,50,1);  normrnd(0,sd,50,1)];
    x2 = [normrnd(0,sd,50,1); normrnd(5,sd,50,1);   normrnd(10,sd,50,1)];
    x3 = [ones(150,1)];
    y1 = [ones(50,1) zeros(50,1) zeros(50,1);  zeros(50,1) ones(50,1) zeros(50,1); zeros(50,1) zeros(50,1) ones(50,1) ];
    input = [x1 x2 x3];
    output = y1;

    nsamp = length(x1);
    ninput = 3;
    nhidden = 4;
    noutput = 3;

    w = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
    v = unifrnd(-1,1,nhidden,noutput);

    mu = .05; p = .9;   % a suggested step and momentum size

    lastdW = 0*W;  lastdV = 0*V;   % initialize the previous weight change variables

    % now do back prop
    """

    # sd = .85;
    std_dev = 0.85                                                  # standard deviation

    # x1 = [normrnd(0,sd,50,1); normrnd(0,sd,50,1);  normrnd(0,sd,50,1)];
    x1 = np.random.normal(0, std_dev, (50,1))                       # (mean, std_dev, size))
    x1 = np.append(x1, np.random.normal(0, std_dev, 50))
    x1 = np.append(x1, np.random.normal(0, std_dev, 50))
    x1 = np.expand_dims(x1, axis=1)                                 # size (150,) to (150,1)

    # x2 = [normrnd(0,sd,50,1); normrnd(5,sd,50,1);   normrnd(10,sd,50,1)];
    x2 = np.random.normal(0, std_dev, (50,1))                       # (mean, std_dev, size))
    x2 = np.append(x2, np.random.normal(5, std_dev, 50))
    x2 = np.append(x2, np.random.normal(10, std_dev, 50))
    x2 = np.expand_dims(x2, axis=1)                                 # size (150,) to (150,1)

    # x3 = [ones(150,1)];
    x3 = np.ones((150,1))

    # y1 = [ones(50,1) zeros(50,1) zeros(50,1);  zeros(50,1) ones(50,1) zeros(50,1); zeros(50,1) zeros(50,1) ones(50,1) ];
    y1 = np.array((np.ones(50), np.zeros(50), np.zeros(50))).T
    temp = np.array((np.zeros(50), np.ones(50), np.zeros(50))).T
    y1 = np.append(y1, temp, axis=0)
    temp = np.array((np.zeros(50), np.zeros(50), np.ones(50))).T
    y1 = np.append(y1, temp, axis=0)

    # input = [x1 x2 x3];
    input = np.append(x1, x2, axis=1)
    input = np.append(input, x3, axis=1)

    # output = y1;
    output = y1

    # nsamp = length(x1);
    # ninput = 3;
    # nhidden = 4;
    # noutput = 3;
    nsamp = len(x1)
    ninput = 3
    nhidden = 4
    noutput = 3

    # w = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
    # v = unifrnd(-1,1,nhidden,noutput);
    w = np.random.uniform(-1, 1, size=(ninput, nhidden))             # min, max, size
    v = np.random.uniform(-1, 1, size=(nhidden, noutput))

    # mu = .05; p = .9;   % a suggested step and momentum size
    # lastdW = 0*W;  lastdV = 0*V;   % initialize the previous weight change variables

    mu = .05                                                        # eta       learning rate
    p = .9                                                          # alpha     momentum rate
    lastdw = 0*w
    lastdv = 0*v

    epochs = 200000

    for i in range(len(output)):
        print str(i+1) + str(output[i])
    print

    for j in range(epochs):

        # Feed forward through layers 0, 1, and 2
        l0 = input
        l1 = sigmoid(np.dot(l0,w))
        l2 = sigmoid(np.dot(l1,v))

        # how much did we miss the target value?
        l2_error = output - l2

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*sigmoid_derivative(l2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(v.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * sigmoid_derivative(l1)

        v += l1.T.dot(l2_delta)
        w += l0.T.dot(l1_delta)

    for i in range(len(l2)):
        print str(i+1) + str(l2[i])

    print "Error:" + str(np.mean(np.abs(l2_error)))



if __name__ == '__main__':
    main()
