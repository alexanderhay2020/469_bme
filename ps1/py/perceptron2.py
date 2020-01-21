"""
Based on video:
https://www.youtube.com/watch?v=kft1AJ9WVDk)

neuron should value first and second columns, and disregard third
"""

import numpy as np

np.random.seed(1) # for troubleshooting, can reproduce

def sigmoid(x):
    """
    args: x - some number
    return: some value between 0 and 1 based on sigmoid function
    """

    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    """
    args: x - some number
    return: derivative of sigmoid given x
    """
    x_prime = x*(1-x)
    return x_prime

tempx = np.random.normal(6,2,size=100)
tempy = np.random.normal(2,1,size=100)
dat1 = np.array((tempx,tempy)).T

tempx = np.random.normal(2,3,size=100)
tempy = np.random.normal(8,1,size=100)
dat2 = np.array((tempx,tempy)).T

dat = np.append(dat1, dat2, axis=0) # [200x2]

y = np.expand_dims(np.append(np.ones((100,1)),0*np.ones((100,1))),axis=1) # [200x1]

# temp = np.append(dat,y,axis=1)
# np.random.shuffle(temp)
# dat = temp[:,:2]
# tempy = temp[:,-1]
# y = np.expand_dims(tempy,axis=1)

weights = np.random.random((2,1)) # starting weight for each column (synapse)
training_output = y

print "Starting Weights: "
print weights
print

for i in range(2000):
    """
    neuron
    """
    input = dat
    xw = np.dot(input,weights) # [4x3]*[3*1]=[4x1]

    output = sigmoid(xw)

    error = training_output - output

    adjustments = error * sigmoid_derivative(output)

    weights = weights + np.dot(input.T,adjustments)

print "Weights after training: "
print weights
print

print "Output: "
# print np.round(output,0)
print output
print
