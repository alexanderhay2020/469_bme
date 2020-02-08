#!/usr/bin/env python2

import random
import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs):

        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden)
        self.output_layer = NeuronLayer(num_outputs)

        self.init_weights()


    def init_weights(self):
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                self.hidden_layer.neurons[h].weights.append(random.random())

        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                self.output_layer.neurons[o].weights.append(random.random())


    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        # print hidden_layer_outputs
        return self.output_layer.feed_forward(hidden_layer_outputs)


    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        output_delta = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # dE/dz
            output_delta[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        hidden_delta = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dy = Sigma dE/dz * dz/dy = Sigma dE/dz * w
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += output_delta[o] * self.output_layer.neurons[o].weights[h]

            # dE/dz = dE/dy * dz/d
            hidden_delta[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].sigmoid_derivative()
        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # dE/dw = dE/dz * dz/dw
                pd_error_wrt_weight = output_delta[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # delta_w = a * dE/dw
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # dE/dw = dE/dz * dz/dw
                pd_error_wrt_weight = hidden_delta[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # delta_w = a * dE/dw
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight


    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calc_MSE(training_outputs[o])
                # print self.output_layer.neurons[o]
        return total_error


    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')


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


    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []


    def calculate_output(self, inputs):
        self.inputs = inputs
        self.y_hat = self.sigmoid(self.calculate_total_net_input())
        return self.y_hat


    def calculate_total_net_input(self):
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
    def calculate_pd_error_wrt_total_net_input(self, y):
        return -(y - self.y_hat) * (self.y_hat * (1 - self.y_hat));

    # The error for each neuron is calculated by the Mean Square Error method:
    def calc_MSE(self, y):
        return 0.5 * (y - self.y_hat) ** 2

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = z = net = xw + xw ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = dz/dw = some constant + 1 * xw^(1-0) + some constant ... = x
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


def main():


    std_dev = 0.85                                                  # standard deviation
    epochs = 10000
    ninput = 3
    nhidden = 4
    noutput = 3
    mu = .05                                                        # eta       learning rate
    p = .9

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

    nsamp = len(input)

    training_set = [None]*nsamp
    total_error = [None]*epochs
    for i in range(nsamp):
        training_set[i] = [list(input[i]),list(output[i])]

    nn = NeuralNetwork(len(input[0]), 4, len(output[0]))
    for i in range(epochs):
        index = random.randint(0,len(input)-1)
        nn.train(list(input[index]), list(output[index]))
        if i % 100 == 0:
            print str(i) + "/" + str(epochs)
        total_error[i] = nn.calculate_total_error(training_set)

    fig0 = plt.figure()
    plt.plot(total_error)
    plt.show()
    # nn.inspect()

if __name__ == '__main__':
    main()
