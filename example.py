from network import Perceptron
from train import trainPerceptron, validateNetwork
from train import constantLearningRate, inverseTimeLearningRate, \
    randomInverseTimeLearningRate

"""Trains and validates two perceptrons on different types of data."""

"""Data set 1 (train1.txt and validation1.txt) is made up of the classes (y < 20 - x  /2) and (y >= 20 - x/2). Data set 2 (train2.txt and validation2.txt) is made up of the classes (y <= x) and (y > x)."""

p1 = Perceptron(2)
p2 = Perceptron(2)
src1 = open('examples/train1.txt')
val1 = open('examples/validation1.txt')
src2 = open('examples/train2.txt')
val2 = open('examples/validation2.txt')
print("Constant learning rates:")
trainPerceptron(p1, src1, constantLearningRate(1))
validateNetwork(p1, val1)
trainPerceptron(p2, src2, constantLearningRate(1))
validateNetwork(p2, val2)

p1 = Perceptron(2)
p2 = Perceptron(2)
src1 = open('examples/train1.txt')
val1 = open('examples/validation1.txt')
src2 = open('examples/train2.txt')
val2 = open('examples/validation2.txt')
print("Inverse time learning rates:")
trainPerceptron(p1, src1, inverseTimeLearningRate(1))
validateNetwork(p1, val1)
trainPerceptron(p2, src2, inverseTimeLearningRate(1))
validateNetwork(p2, val2)

p1 = Perceptron(2)
p2 = Perceptron(2)
src1 = open('examples/train1.txt')
val1 = open('examples/validation1.txt')
src2 = open('examples/train2.txt')
val2 = open('examples/validation2.txt')
print("Random inverse time learning rates:")
trainPerceptron(p1, src1, randomInverseTimeLearningRate(1))
validateNetwork(p1, val1)
trainPerceptron(p2, src2, randomInverseTimeLearningRate(1))
validateNetwork(p2, val2)
