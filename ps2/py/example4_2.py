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
    w, b1, v, b2 = model['w'], model['b1'], model['v'], model['b2']
    # Do the first Linear step
    z1 = a0.dot(w) + b1
    # Put it through the first activation function
    a1 = sigmoid(z1)
    # Second linear step
    z2 = a1.dot(v) + b2
    # Put through second activation function
    a2 = sigmoid(z2)
    #Third linear step
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2}

    return cache


# This is the backward propagation function
def backward_prop(model, cache, y):
    # Load parameters from model
    w, b1, v, b2 = model['w'], model['b1'], model['v'], model['b2']
    # Load forward propagation results
    a0, a1, a2 = cache['a0'], cache['a1'], cache['a2']
    # Get number of samples
    nsamp = y.shape[0]

    # Calculate loss derivative with respect to output
    dz2 = (a2 - y)


    # Calculate loss derivative with respect to second layer weights

    # dW3 = 1/nsamp*(a2.T).dot(dz3) #
    dW2 = 1/nsamp*(a1.T).dot(dz2)
    # Calculate loss derivative with respect to second layer bias
    # db3 = 1/nsamp*np.sum(dz3, axis=0)


    # Calculate loss derivative with respect to first layer

    # dz2 = np.multiply(dz3.dot(W3.T), sigmoid_derivative(a2))
    # Calculate loss derivative with respect to first layer weights
    # dW2 = 1/nsamp*np.dot(a1.T, dz2)
    # Calculate loss derivative with respect to first layer bias
    db2 = 1/nsamp*np.sum(dz2, axis=0)
    dz1 = np.multiply(dz2.dot(v.T), sigmoid_derivative(a1))
    dW1 = 1/nsamp*np.dot(a0.T,dz1)
    db1 = 1/nsamp*np.sum(dz1,axis=0)

    # Store gradients
    grads = {'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}

    return grads


def update_parameters(model, grads, learning_rate):
    # Load parameters
    w, b1, v, b2 = model['w'], model['b1'], model['v'], model['b2']

    # Update parameters
    w -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    v -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']

    # Store and return parameters
    model = { 'w': w, 'b1': b1, 'v': v, 'b2': b2}

    return model


def initialize_parameters(ninput, nhidden, noutput):
    # First layer weights
    w = np.random.uniform(-1, 1, size=(ninput, nhidden))             # min, max, size

    # First layer bias
    b1 = np.ones((150, nhidden))

    # Second layer weights
    v = np.random.uniform(-1, 1, size=(nhidden, noutput))

    # Second layer bias
    b2 = np.ones((150, noutput))

    # b1 = np.ones((150, 1)) # b1
    # b2 = np.ones((150, 1)) # b2
    # Package and return model
    model = { 'w': w, 'b1': b1, 'v': v, 'b2': b2}
    return model


def train(model,X_,y_,learning_rate, epochs, print_loss=False):
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):

        # Forward propagation
        cache = forward_prop(model,X_)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation

        grads = backward_prop(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model

        model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)

        if i % 1000 == 0:
          print("iteration %i: " %(i))
        # Pring loss & accuracy every 100 iterations
        # if print_loss and i % 100 == 0:
        #     a3 = cache['a3']
        #     print('Loss after iteration',i,':',softmax_loss(y_,a3))
        #     y_hat = predict(model,X_)
        #     y_true = y_.argmax(axis=1)
        #     print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
        #     losses.append(accuracy_score(y_pred=y_hat,y_true=y_true)*100)
    return model


def predict(model, input):
    # Do forward pass
    c = forward_prop(model, input)
    #get y_hat
    y_hat = np.argmax(c['a2'], axis=1)

    # w, v = model['w'], model['v']
    # y_hat = np.dot(input, w)
    # y_hat = np.dot(y_hat, v)

    return y_hat


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

    print "output:"
    for i in range(len(output)):
        print str(i+1) + " " + (str(output[i]))
    print


    # w = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
    # v = unifrnd(-1,1,nhidden,noutput);
    w = np.random.uniform(-1, 1, size=(ninput, nhidden))             # min, max, size
    v = np.random.uniform(-1, 1, size=(nhidden, noutput))

    # This is what we return at the end
    model = initialize_parameters(ninput, nhidden, noutput)
    model = train(model, input, output, mu, epochs, print_loss=True)
    # plt.plot(losses)

    y_hat = predict(model, input)

    print "w"
    print w
    print
    print "v"
    print v
    print

    index = 0

    # Do forward pass
    c = forward_prop(model, input)
    #get y_hat
    y_hat = c['a2']

    # for i in range(len(y_hat)):
    #     max = y_hat[i][0]
    #     for j in range(len(y_hat[i])):
    #         if (y_hat[i][j] >= max):
    #             max = y_hat[i][j]
    #             index = j
    #     for k in range(len(y_hat[i])):
    #         if (k == index):
    #             y_hat[i][k] = 1
    #         else:
    #             y_hat[i][k] = 0


    print "y_hat adjusted"
    for i in range(len(y_hat)):
        if (i<9):
            print str(i+1) + "   " + (str(y_hat[i]))
        elif (i>8 and i<99):
            print str(i+1) + "  " + (str(y_hat[i]))
        elif (i>97):
            print str(i+1) + " " + (str(y_hat[i]))

    # print

    # print "percent error:"
    # error = (sum(abs(y_hat - output))/150)*100
    # print error


if __name__ == '__main__':
    main()
