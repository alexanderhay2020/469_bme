import numpy as np
import math
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

def function(x):

    y = 2*x + np.random.normal(0, 1, x.size) # [1000x1]

    return y

def part_2a():
    """
    MATLAB CODE:
    x = unifrnd(0,10,1000,1);
    y = 2*x + normrnd(0,1,size(x));
    plot(x,y,'.');

    dat1 = [normrnd(6,2,100,1) normrnd(2,1,100,1)];
    dat2 = [normrnd(2,3,100,1) normrnd(8,1,100,1)];
    dat = [dat1; dat2];

    line-by-line translation with same output, modified to fit the assignment
    """
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy))

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,1,size=100)
    dat2 = np.array((tempx,tempy))

    fig1 = plt.figure()
    plt.plot(dat1[0], dat1[1], "r.", label="dat1")
    plt.plot(dat2[0], dat2[1], "b.", label="dat2")
    plt.title("Linear Classification: Part A")
    plt.xlim(-4,12)
    plt.ylim(-2,12)
    plt.legend()
    plt.show()


def main():
    part_2a()

if __name__ == '__main__':
    main()
