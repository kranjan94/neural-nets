from network import Network
from train import train, validate, inverseTimeLearningRate
from activationFunctions import identityActivation

"""Build, trains, and validates a network on the MNIST handwritten digit
data. Data must be read via mnistData/readMNISTData.py first."""

numPixels = 28 * 28
net = Network(numPixels, [], 10, inputActivationFunction=identityActivation)
train(net, 'mnistData/mnistTrain.txt', inverseTimeLearningRate(1, 0.7))
validate(net, 'mnistData/mnistValidate.txt')
