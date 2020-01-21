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

def make_data():
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,2,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,2,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0) # [200x2]

    tempx = np.random.normal(6,2,size=10)
    tempy = np.random.normal(2,2,size=10)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=10)
    tempy = np.random.normal(8,2,size=10)
    dat2 = np.array((tempx,tempy)).T

    test = np.append(dat1, dat2, axis=0) # [10x2]
    y_test = np.expand_dims(np.append(np.ones((5,1)),0*np.ones((5,1))),axis=1) # [200x1]



def part_2b():
    """
    MATLAB Dataset
    """
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,2,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,2,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0) # [200x2]

    y = np.expand_dims(np.append(np.ones((100,1)),0*np.ones((100,1))),axis=1) # [200x1]

    temp = np.append(dat,y,axis=1)
    np.random.shuffle(temp)
    dat = temp[:,:2]
    tempy = temp[:,-1]
    y = np.expand_dims(tempy,axis=1)

    w = np.random.random((3,2)) # starting weights [3x2]
    v = np.random.random((3,1))

    # bias = 1 # bias
    bias = np.ones((len(dat),1)) # bias

    for i in range(10000):

        input = dat
        input = np.append(input,bias,axis=1)

        hidden = sigmoid(np.dot(input,w))
        hidden = np.append(hidden,bias,axis=1)

        output = sigmoid(np.dot(hidden,v))

        output_err = (output - y)**2
        output_delta = -2*(output_err * sigmoid_derivative(output))

        hidden_err = np.dot(output_delta, v.T)
        hidden_delta = -2*(hidden_err * sigmoid_derivative(hidden))

    print output
    print len(output)

    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "r.", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "b.", label="dat2")
    plt.title("Linear Classification: Part B")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()


def main():
    part_2b()
    plt.show()

if __name__ == '__main__':
    main()


    # for i in range(2000):
    #     """
    #     neuron
    #     """
    #     # xw = np.dot(dat,w) # [201x2]*[2*1]=[201x1]
    #
    #     h = sigmoid(np.dot(dat,w)) # hidden layer
    #     o = sigmoid(np.dot(h,v)) # output layer
    #     print
    #     print("dat.shape: " + str(dat.shape))
    #     print("w.shape: " + str(w.shape))
    #     print("o.shape: " + str(o.shape))
    #     print("h.shape: " + str(h.shape))
    #     print("v.shape: " + str(v.shape))
    #     print
    #
    #     o_error = y-o # output error
    #     o_adjustment = np.multiply(o_error,sigmoid_derivative(o))
    #     h_error = np.dot(o_adjustment,w.T)
    #     h_adjustment = np.multiply(h_error,sigmoid_derivative(h))
    #
    #     print("o_error.shape: " + str(o_error.shape))
    #     print("o_adj.shape: " + str(o_adjustment.shape))
    #     print("h_error.shape: " + str(h_error.shape))
    #     print("h_adj.shape: " + str(h_adjustment.shape))
    #     print
    #     # adjustments = error * sigmoid_derivative(y_hat)#[:-1])
    #
    #     # w = w + np.dot(dat.T,adjustments)
    #     v = v + np.dot(h.T,o_adjustment)
    #     w = w + np.dot(dat.T,h_adjustment)
    #
    #     print ("new w.shape: " + str(w.shape))
    #     print ("new v.shape: " + str(v.shape))
    #     print
    #     print
