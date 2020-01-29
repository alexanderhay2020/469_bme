#!/usr/bin/env python2

import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1)

def create_data():
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

    W = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
    V = unifrnd(-1,1,nhidden,noutput);

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
    x1 = np.expand_dims(x1, axis=1)                                  # size (150,) to (150,1)

    # x2 = [normrnd(0,sd,50,1); normrnd(5,sd,50,1);   normrnd(10,sd,50,1)];
    x2 = np.random.normal(0, std_dev, (50,1))                       # (mean, std_dev, size))
    x2 = np.append(x2, np.random.normal(5, std_dev, 50))
    x2 = np.append(x2, np.random.normal(10, std_dev, 50))
    x2 = np.expand_dims(x2, axis=1)                                  # size (150,) to (150,1)

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
