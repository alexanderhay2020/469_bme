#!/usr/bin/env python2
"""
https://github.com/dennybritz/nn-from-scratch
https://www.freecodecamp.org/news/building-a-3-layer-neural-network-from-scratch-99239c4af5d3/
https://www.kaggle.com/jantinbergen/challenge-1-bletchley-three-layer-nnetwork?scriptVersionId=2460820
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
http://staff.itee.uq.edu.au/janetw/cmc/chapters/BackProp/index2.html
https://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html
"""


import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)

class NeuralNetwork:
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
            output_delta[i] = self.output.neurons[i].dE_dz(output[i])

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
        for t in range(len(training_sets)):
            input, output = training_sets[t]
            self.feed_forward(input)
            for o in range(len(output)):
                total_error += self.output.neurons[o].calc_MSE(output[o])
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
        for i in range(len(self.inputs)):
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
    mu = .05; p = .9;   % a suggested step and momentum size
    lastdW = 0*W;  lastdV = 0*V;   % initialize the previous weight change variables
    % now do back prop
    """

    # sd = .85;
    std_dev = 0.85

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

    epochs = 15000
    nsamp = len(input)
    ninput = 3
    nhidden = 4
    noutput = 3

    # Initial Data Plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    for i in range(nsamp):
        xs = input[i][0]
        ys = input[i][1]
        zs = input[i][2]
        if (i<50):
            ax.scatter(xs, ys, zs, marker='.', color='b', label='dataset 1')
        elif (50<i<100):
            ax.scatter(xs, ys, zs, marker='.', color='r', label='dataset 2')
        elif (i>100):
            ax.scatter(xs, ys, zs, marker='.', color='g', label='dataset 3')
    ax.set_title('Dataset')


    # Supervised Learning
    training_set = [None]*nsamp
    total_error = [None]*epochs
    for i in range(nsamp):
        training_set[i] = [list(input[i]),list(output[i])]

    nn = NeuralNetwork(ninput, nhidden, noutput)
    for i in range(epochs):
        index = random.randint(0,len(input)-1)
        nn.train(list(input[index]), list(output[index]))
        if i % 100 == 0:
            print "epoch: " + str(i) + "/" + str(epochs)
        total_error[i] = nn.calculate_total_error(training_set)


    # Error Plot
    fig3 = plt.figure()
    plt.title('Total Error')
    plt.plot(total_error)

    plt.show()

if __name__ == '__main__':
    main()
