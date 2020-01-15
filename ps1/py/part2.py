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


def part_2a():
    """
    MATLAB CODE:
    dat1 = [normrnd(6,2,100,1) normrnd(2,1,100,1)];
    dat2 = [normrnd(2,3,100,1) normrnd(8,1,100,1)];
    dat = [dat1; dat2];
    y = [ones(100,1); 0*ones(100,1)];

    line-by-line translation with same output, modified to fit the assignment
    """
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,1,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0)

    y = np.array((np.ones((100)),-1*np.ones((100)))).T

    fig1 = plt.figure()
    plt.plot(dat1[0], dat1[1], "r.", label="dat1")
    plt.plot(dat2[0], dat2[1], "b.", label="dat2")
    plt.title("Linear Classification: Part A")
    plt.xlim(-4,12)
    plt.ylim(-2,12)
    plt.legend()
    plt.show()

def part_2b():
    """
    MATLAB CODE:
    dat1 = [normrnd(6,2,100,1) normrnd(2,2,100,1)];
    dat2 = [normrnd(2,3,100,1) normrnd(8,2,100,1)];
    dat = [dat1; dat2];
    y = [ones(100,1); 0*ones(100,1)];

    line-by-line translation with same output, modified to fit the assignment
    """
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,2,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,2,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0)

    y = np.append(np.ones((100)),0*np.ones((100)))

    # for i in range(1):
        # """
        # neuron
        # """
    w = np.random.random((2)) # starting weights [48x1]

    y_hat = np.dot(dat,w) # [200x2]*[2x1]=[200x1]

    output = sigmoid(y_hat)

    error = (y-output)**2

    adjustments = error * sigmoid_derivative(output)

    w = w + np.dot(dat.T,adjustments)

    print w.shape
    print dat.shape

    classfied = np.dot(dat,w)

    print classfied.shape

    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "r.", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "b.", label="dat2")
    plt.plot(dat.T[1],classfied)
    plt.title("Linear Classification: Part B")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()


def main():
    # part_2a()
    part_2b()
    plt.show()

if __name__ == '__main__':
    main()
