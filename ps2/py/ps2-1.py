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


def calculate_loss(model):
    w, v = model['w'], model['v']
    # Forward propagation to calculate our predictions
    z1 = np.dot(input, w)
    h1 = sigmoid(z1)
    z2 = h1.dot(v)
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += p/2 * (np.sum(np.square(w)) + np.sum(np.square(v)))
    # return 1./num_examples * data_loss


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
    epochs = 20000
    ninput = 3
    nhidden = 4
    noutput = 3
    mu = .05                                                        # eta       learning rate
    p = .9
    print_loss=True


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

    b1 = np.ones((nsamp, 1)) # b1
    b2 = np.ones((nsamp, 1)) # b2

    # w = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
    # v = unifrnd(-1,1,nhidden,noutput);
    w = np.random.uniform(-1, 1, size=(ninput, nhidden))             # min, max, size
    v = np.random.uniform(-1, 1, size=(nhidden, noutput))

    # mu = .05; p = .9;   % a suggested step and momentum size
    # lastdW = 0*W;  lastdV = 0*V;   % initialize the previous weight change variables
                                                     # alpha     momentum rate
    lastdw = 0*w
    lastdv = 0*v



    for i in range(len(output)):
        print str(i+1) + str(output[i])
    print


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    2-LAYER PERCEPTRON NETWORK
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Build a model with a 4-dimensional hidden layer
    model = {}
    # input = np.append(input, b1, axis=1)
    # Gradient descent. For each batch...
    for i in range(epochs):
        # print i

        # Forward propagation
        z1 = np.dot(input, w) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(h1, v) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        # dz2 = ()

        delta3 = probs
        delta3[range(nsamp), output[i]] -= 1
        lastdv = (h1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(v.T) * sigmoid_derivative(h1)
        lastdw = np.dot(input.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        lastdv += p * v
        lastdw += p * w

        # Gradient descent parameter update
        w += -mu * lastdw
        v += -mu * lastdv

        # Assign new parameters to the model
        model = { 'w': w, 'v': v}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        # if print_loss and i % 1000 == 0:
        #   print("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    h1 = np.dot(input, w)
    prediction = np.dot(h1, v)

    print "w"
    print w.shape
    print w
    print
    print "v"
    print v.shape
    print v
    print
    for i in range(len(prediction)):
        print str(i+1) + str(prediction[i])
    print


if __name__ == '__main__':
    main()
