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
    dW3 = 1/m*(a2.T).dot(dz3) #
    dW2 = 1/m*(a1.T).dot(dz2)
    # Calculate loss derivative with respect to second layer bias
    db3 = 1/m*np.sum(dz3, axis=0)
    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T) ,tanh_derivative(a2))
    # Calculate loss derivative with respect to first layer weights
    dW2 = 1/m*np.dot(a1.T, dz2)
    # Calculate loss derivative with respect to first layer bias
    db2 = 1/m*np.sum(dz2, axis=0)
    dz1 = np.multiply(dz2.dot(W2.T),tanh_derivative(a1))
    dW1 = 1/m*np.dot(a0.T,dz1)
    db1 = 1/m*np.sum(dz1,axis=0)
    # Store gradients
    grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}

    return grads
