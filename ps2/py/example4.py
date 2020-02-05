#!/usr/bin/env python2

import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1)


def sigmoid(x):

    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):

    return sigmoid(x)*(1-sigmoid(x))


def forward_prop(model, a0):
    # Load parameters from model
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # Do the first Linear step
    z1 = a0.dot(w1) + b1
    # Put it through the first activation function
    a1 = np.tanh(z1)
    # Second linear step
    z2 = a1.dot(w2) + b2
    # Put through second activation function
    a2 = np.tanh(z2)
    #Third linear step
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2}

    return cache

# This is the backward propagation function
def backward_prop(model, cache, y):
    # Load parameters from model
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # Load forward propagation results
    a0, a1, a2, a3 = cache['a0'], cache['a1'], cache['a2']
    # Get number of samples
    nsamp = y.shape[0]
    # Calculate loss derivative with respect to output
    dz2 = loss_derivative(y = y, y_hat = a2)
    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/nsamp*(a2.T).dot(dz3) #
    dW2 = 1/nsamp*(a1.T).dot(dz2)
    # Calculate loss derivative with respect to second layer bias
    db3 = 1/nsamp*np.sum(dz3, axis=0)

    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T), tanh_derivative(a2))
    dW2 = 1/nsamp*np.dot(a1.T, dz2)
    db2 = 1/nsamp*np.sum(dz2, axis=0)

    dz1 = np.multiply(dz2.dot(W2.T),tanh_derivative(a1))
    dW1 = 1/nsamp*np.dot(a0.T,dz1)
    db1 = 1/nsamp*np.sum(dz1,axis=0)
    # Store gradients
    grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}

    return grads

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


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    DATA INITIALIZATION
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # sd = .85;
    std_dev = 0.85                                                  # standard deviation
    epochs = 2000
    mu = .05                                                        # eta       learning rate
    p = .9


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
    nsamp = len(input)
    ninput = 3
    nhidden = 4
    noutput = 3

    b1 = np.ones((nsamp, 1)) # b1
    b2 = np.ones((nsamp, 1)) # b2

    # w = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
    # v = unifrnd(-1,1,nhidden,noutput);
    w = np.random.uniform(-1, 1, size=(ninput, nhidden))             # min, max, size
    v = np.random.uniform(-1, 1, size=(nhidden, noutput))
