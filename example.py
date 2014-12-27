from network import Network, Perceptron
from train import train, validate
from train import constantLearningRate, inverseTimeLearningRate, \
    randomInverseTimeLearningRate, exponentialLearningRate
from activationFunctions import arctanActivation

"""Trains and validates networks on different types of data."""

"""Data set 1 (train1.txt and validation1.txt) is made up of the classes (y < 20 - x  /2) and (y >= 20 - x/2). Data set 2 (train2.txt and validation2.txt) is made up of the classes (y <= x) and (y > x). Data set 3 (train3.txt and
validation3.txt) is made up of 4 classes, one for each of the 4 quadrants of the
plane."""

"""An I-H1-H2-...-HN-O network refers to a network with I input nodes, H1 nodes
in its first hidden layer, H2 nodes in its second hidden layer (etc.), and O
nodes in its output layer. For example, a 2-4-3 network has 2 input nodes, one
hidden layer with 4 nodes, and 3 output nodes."""

train1 = 'examples/train1.txt'
train2 = 'examples/train2.txt'
train3 = 'examples/train3.txt'
validate1 = 'examples/validation1.txt'
validate2 = 'examples/validation2.txt'
validate3 = 'examples/validation3.txt'
set3Labels = ['upper_left', 'upper_right', 'lower_left', 'lower_right']

p1 = Perceptron(2)
p2 = Perceptron(2)
print("\nPerceptrons with constant learning rates, datasets 1 and 2:")
train(p1, train1, constantLearningRate(1))
validate(p1, validate1)
train(p2, train2, constantLearningRate(1))
validate(p2, validate2)

p1 = Perceptron(2)
p2 = Perceptron(2)
print("\nPerceptrons with inverse time learning rates, datasets 1 and 2:")
train(p1, train1, inverseTimeLearningRate(1))
validate(p1, validate1)
train(p2, train2, inverseTimeLearningRate(1))
validate(p2, validate2)

p1 = Perceptron(2, bias=True)
p2 = Perceptron(2, bias=True)
print("\nPerceptrons with exponential learning rates and bias, "
    + "datasets 1 and 2:")
train(p1, train1, exponentialLearningRate(0.5))
validate(p1, validate1)
train(p2, train2, exponentialLearningRate(0.5))
validate(p2, validate2)

net1 = Network(2, [], 2)
net2 = Network(2, [], 2)
print("\n2-2 networks with inverse time learning rates, datasets 1 and 2:")
train(net1, train1, inverseTimeLearningRate(1))
validate(net1, validate1)
train(net2, train2, inverseTimeLearningRate(1))
validate(net2, validate2)

net = Network(2, [], 4, validLabels = set3Labels)
print("\n2-4 network with inverse time learning rates, dataset 3:")
train(net, train3, inverseTimeLearningRate(1))
validate(net, validate3)
