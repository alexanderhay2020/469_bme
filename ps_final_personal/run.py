"""*****************************************************************************
Alex Hay
BME_469
Final Project

Comparing neural model learning performance to monkey performance

SOURCES:
https://www.sciencedirect.com/science/article/pii/S0021929009001444#fig5


*****************************************************************************"""


import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)


class NeuronNetwork:
    mu = .05      # eta       learning rate
    p = .9

    def __init__(self, nsamp, num_hidden, num_outputs):

        self.nsamp = nsamp
        self.hidden = NeuronLayer(num_hidden)
        self.output = NeuronLayer(num_outputs)

        self.init_weights()

    # w = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
    # v = unifrnd(-1,1,nhidden,noutput);
    def init_weights(self):
        for j in range(len(self.hidden.neurons)):
            for k in range(self.nsamp):
                self.hidden.neurons[j].weights.append(np.random.normal(-1, 1))

        for i in range(len(self.output.neurons)):
            for j in range(len(self.hidden.neurons)):
                self.output.neurons[i].weights.append(np.random.normal(-1, 1))

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden.feed_forward(inputs)
        return self.output.feed_forward(hidden_layer_outputs)


    # Uses online learning, ie updating the weights after each training case
    def train(self, input, output):
        self.feed_forward(input)

        # 1. Output neuron deltas
        # dE/dz
        output_delta = [0] * len(self.output.neurons)
        for i in range(len(self.output.neurons)):
            # print len(self.output.neurons)
            output_delta[i] = self.output.neurons[i].dE_dz(output)

        # 2. Hidden neuron deltas
        # dE/dy = Sigma dE/dz * dz/dy = Sigma dE/dz * w
        # dE/dz = dE/dy * dz/d
        hidden_delta = [0] * len(self.hidden.neurons)
        for j in range(len(self.hidden.neurons)):
            d_error_wrt_hidden_neuron_output = 0
            for i in range(len(self.output.neurons)):
                d_error_wrt_hidden_neuron_output += output_delta[i] * self.output.neurons[i].weights[j]

            hidden_delta[j] = d_error_wrt_hidden_neuron_output * self.hidden.neurons[j].sigmoid_derivative()

        # 3. Update output neuron weights
        # dE/dw = dE/dz * dz/dw
        # delta_w = a * dE/dw
        for i in range(len(self.output.neurons)):
            for w_o in range(len(self.output.neurons[i].weights)):
                pd_error_wrt_weight = (self.p * output_delta[i]) * self.output.neurons[i].dz_dw(w_o)
                self.output.neurons[i].weights[w_o] -= self.mu * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        # dE/dw = dE/dz * dz/dw
        # delta_w = a * dE/dw
        for j in range(len(self.hidden.neurons)):
            for w_h in range(len(self.hidden.neurons[j].weights)):
                pd_error_wrt_weight = hidden_delta[j] * self.hidden.neurons[j].dz_dw(w_h)
                self.hidden.neurons[j].weights[w_h] -= self.mu * pd_error_wrt_weight


    def calculate_total_error(self, training_sets):
        total_error = 0
        # print training_sets
        for t in range(len(training_sets)):
            temp = training_sets[t]
            input = temp[:2]
            output = temp[-1]
            self.feed_forward(input)
            # for o in range(len(output)):
            total_error += self.output.neurons[0].calc_MSE(output)
                # print self.output.neurons

        return total_error


class NeuronLayer:
    def __init__(self, num_neurons):

        # Every neuron in a layer shares the same bias
        self.bias = 1

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))


    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs


    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []


    def calculate_output(self, inputs):
        self.inputs = inputs
        self.y_hat = self.sigmoid(self.xw())
        return self.y_hat


    def xw(self):
        total = 0
        for i in range(2):
            # print len(self.inputs)
            total += self.inputs[i] * self.weights[i]
        return total + self.bias


    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


    def sigmoid_derivative(self):
        return self.y_hat * (1 - self.y_hat)

    # d = dE/dz = dE/dy * dy/dz
    def dE_dz(self, y):
        return -(y - self.y_hat) * (self.y_hat * (1 - self.y_hat));

    # The error for each neuron is calculated by the Mean Square Error method:
    def calc_MSE(self, y):
        return 0.5 * (y - self.y_hat) ** 2

    # = dz/dw = x
    def dz_dw(self, index):
        return self.inputs[index]


def create_data():
    """******************************************
    recreating the data

    GIVEN:
    angular distance between targets:       50deg
    moment of inertia of head about C7:     1418.0 +- 245.5

    MEASURED:
    peak head velocity:                     173deg/s +- 12deg/s

    CALCULATIONS:
    time = angular distance/head velocity

    ASSUMES:
    motor command is encoded as peak torque necessary to move head to target
    ******************************************"""

    I = 1418

    # inputs
    w = np.random.normal(173, 12, 20)
    theta = 50

    # hidden layer
    t = theta/w

    # output
    torque = (I*w)/t

    return t, w, torque

def main():
    """******************************************
    recreating the data

    GIVEN:
    angular distance between targets:       50deg
    moment of inertia of head about C7:     1418.0 +- 245.5

    MEASURED:
    peak head velocity:                     173deg/s +- 12deg/s

    CALCULATIONS:
    time = angular distance/head velocity

    ASSUMES:
    motor command is encoded as peak torque necessary to move head to target
    ******************************************"""

    I = 0.1418*np.ones(20)                       # kgm^2
    theta = 50*(np.pi/180)

    w = np.random.normal(173, 12, 20)
    w = w*(np.pi/180)                            # rad/sec
    t = theta/w                                  # sec

    torque = (I*w)/t                             # kgm^2/s^2

    input = np.array([t, torque, I]).T
    output = w

    epochs = 150
    nsamp = len(input)
    ninput = 3
    nhidden = 3
    noutput = 1

    # Supervised Learning
    training_set = np.array([t, torque, I, w]).T
    total_error = [None]*epochs
    # total_error = np.zeros(epochs)

    nn = NeuronNetwork(ninput, nhidden, noutput)
    for i in range(epochs):
        index = random.randint(0,len(input)-1)
        nn.train(input[index], output[index])
        if i % 10 == 0:
            print "epoch: " + str(i) + "/" + str(epochs)
        total_error[i] = nn.calculate_total_error(training_set)

    norm=max(total_error)
    # for i in range(len(total_error)):
        # total_error[i] = total_error[i]/norm
    # Error Plot
    fig1 = plt.figure()
    plt.title('Total Error')
    plt.plot(total_error/norm)

    plt.show()

if __name__ == '__main__':
    main()
