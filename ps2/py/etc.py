#!/usr/bin/env python2

import random
import math
import numpy as np


# class NeuralNetwork:
#     LEARNING_RATE = 0.5
#
#     def __init__(self, num_inputs, num_hidden, num_outputs):
#
#         self.num_inputs = num_inputs
#
#         self.hidden_layer = NeuronLayer(num_hidden)
#         self.output_layer = NeuronLayer(num_outputs)
#
#         self.init_weights_from_inputs_to_hidden_layer_neurons()
#         self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons()
#
#
#     def init_weights_from_inputs_to_hidden_layer_neurons(self):
#         weight_num = 0
#         for h in range(len(self.hidden_layer.neurons)):
#             for i in range(self.num_inputs):
#                 self.hidden_layer.neurons[h].weights.append(random.random())
#                 weight_num += 1
#
#
#     def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self):
#         weight_num = 0
#         for o in range(len(self.output_layer.neurons)):
#             for h in range(len(self.hidden_layer.neurons)):
#                 self.output_layer.neurons[o].weights.append(random.random())
#                 weight_num += 1


class NeuronLayer:
    def __init__(self, num_neurons):

        # Every neuron in a layer shares the same bias
        self.bias = 1

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))


    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)


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
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output


    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias


    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    # def squash(self, total_net_input):
    #     return 1 / (1 + math.exp(-total_net_input))

    def sigmoid(self, x):

        return 1/(1+np.exp(-x))


    def sigmoid_derivative(self, x):

        return sigmoid(x)*(1-sigmoid(x))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (dE/dy) and
    # the derivative of the output with respect to the total net input (dy/dz) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (d) [1]
    # d = dE/dz = dE/dy * dy/dz
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.sigmoid_derivative();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as y and target output as t so:
    # = dE/dy = -(t - y)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # y = psi = 1 / (1 + e^(-z))
    # Note that where j represents the output of the neurons in whatever layer we're looking at and i represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dy/dz = y * (1 - y)
    def sigmoid_derivative(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = z = net = xw + xw ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = dz/dw = some constant + 1 * xw^(1-0) + some constant ... = x
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


LEARNING_RATE = 0.5
def NeuralNetwork(num_inputs, num_hidden, num_outputs):

    num_inputs = num_inputs

    hidden_layer = NeuronLayer(num_hidden)
    output_layer = NeuronLayer(num_outputs)

    init_weights_from_inputs_to_hidden_layer_neurons()
    init_weights_from_hidden_layer_neurons_to_output_layer_neurons()


def init_weights_from_inputs_to_hidden_layer_neurons():
    weight_num = 0
    for h in range(len(hidden_layer.neurons)):
        for i in range(num_inputs):
            hidden_layer.neurons[h].weights.append(random.random())
            weight_num += 1


def init_weights_from_hidden_layer_neurons_to_output_layer_neurons():
    weight_num = 0
    for o in range(len(output_layer.neurons)):
        for h in range(len(hidden_layer.neurons)):
            output_layer.neurons[o].weights.append(random.random())
            weight_num += 1

training_sets = [
    [[1, 0, 0], [1, 0, 0]],
    [[0, 1, 0], [0, 1, 0]],
    [[0, 0, 1], [0, 0, 1]],
    # [[0, 0, 0], [0, 0, 0]],
    # [[1, 1, 1], [0, 0, 0]]
]

nn = NeuralNetwork(len(training_sets[0][0]), 4, len(training_sets[0][1]))
