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
    x1 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];    % the data clusters
    x2 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];
    input = [x1 x2];
    nsamp = length(input);

    w1 = mean(input) + normrnd(-1,1,1,2);   % intiialize the weights somewhere in the center of the data
    w2 = mean(input) + normrnd(-1,1,1,2);
    """

    # x1 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];    % the data clusters
    x1 = np.random.normal(0, 1, (50,1))                             # (mean, std_dev, size))
    x1 = np.append(x1, np.random.normal(5, 1, 50))
    x1 = np.expand_dims(x1, axis=1)                                 # size (150,) to (150,1)

    # x2 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];
    x2 = np.random.normal(0, 1, 50)                             # (mean, std_dev, size))
    x2 = np.append(x2, np.random.normal(5, 1, 50))
    x2 = np.expand_dims(x2, axis=1)                                 # size (150,) to (150,1)

    # input = [x1 x2];
    input = np.append(x1, x2, axis=1)

    # nsamp = length(input);
    nsamp = len(input)

    # w1 = mean(input) + normrnd(-1,1,1,2);   % intiialize the weights somewhere in the center of the data
    w1 = np.mean(input) + np.random.normal(-1, 1, (1,2))
    w2 = np.mean(input) + np.random.normal(-1, 1, (1,2))

    tempw1 = np.mean(input) + np.random.normal(-1, 1, (1,2))
    tempw2 = np.mean(input) + np.random.normal(-1, 1, (1,2))

    diff1 = 1
    diff2 = 1

    iter = 0

    cluster = np.empty(nsamp)

    fig0 = plt.figure()
    plt.plot(w1.T[0], w1.T[1], "rx", label="group 1 initial mean")
    plt.plot(w2.T[0], w2.T[1], "bx", label="group 2 initial mean")
    plt.plot(input.T[0,:50], input.T[1,:50], "r.", label="group 1")
    plt.plot(input.T[0,50:], input.T[1,50:], "b.", label="group 2")
    plt.title("Part 2: K-Means Clustering Initial Data")
    plt.legend()

    while (diff1 > 0) & (diff2 > 0):
        tempx1 = 0
        tempx2 = 0
        tempy1 = 0
        tempy2 = 0

        # classifies data
        for i in range(nsamp):
            euclidian1 = np.sqrt((w1.T[0] - input[i][0])**2 + (w1.T[1] - input[i][1])**2)
            euclidian2 = np.sqrt((w2.T[0] - input[i][0])**2 + (w2.T[1] - input[i][1])**2)

            if euclidian1 < euclidian2:
                cluster[i] = 0
            else:
                cluster[i] = 1

        # moves cluster mean
        for i in range(nsamp):
            if cluster[i] == 0:
                tempx1 += input[i][0]
                tempy1 += input[i][1]
            elif cluster[i] ==1:
                tempx2 += input[i][0]
                tempy2 += input[i][1]

        tempw1[0][0] = tempx1/sum(cluster)
        tempw1[0][1] = tempy1/sum(cluster)

        tempw2[0][0] = tempx2/(nsamp-sum(cluster))
        tempw2[0][1] = tempy2/(nsamp-sum(cluster))

        diff1 = sum(sum(abs(tempw1 - w1)))
        diff2 = sum(sum(abs(tempw2 - w2)))

        w1 = tempw1
        w2 = tempw2

        iter += 1

        fig2 = plt.figure()
        plt.plot(w1.T[0], w1.T[1], "rx", label="group 1 initial mean")
        plt.plot(w2.T[0], w2.T[1], "bx", label="group 2 initial mean")
        plt.plot(input.T[0,:50], input.T[1,:50], "r.", label="group 1")
        plt.plot(input.T[0,50:], input.T[1,50:], "b.", label="group 2")
        plt.title("Part 2: K-Means Clustering Iteration " + str(iter))
        plt.legend()

    fig2 = plt.figure()
    plt.plot(tempw1.T[0], tempw1.T[1], "rx", label="group 1 initial mean")
    plt.plot(tempw2.T[0], tempw2.T[1], "bx", label="group 2 initial mean")
    plt.plot(input.T[0,:50], input.T[1,:50], "r.", label="group 1")
    plt.plot(input.T[0,50:], input.T[1,50:], "b.", label="group 2")
    plt.title("Part 2: K-Means Clustering Final Data")
    plt.legend()
    

if __name__ == '__main__':
    main()
    plt.show()
