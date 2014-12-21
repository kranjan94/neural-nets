from random import random
from network import Perceptron
from activationFunctions import DERIVATIVES
from networkExceptions import TrainingError

""" -- Learning rate functions -- """
"""These functions return functions which take a time argument as a parameter
and return a learning rate."""
def constantLearningRate(rate):
    """Returns the same rate, regardless of time parameter."""
    def function(t):
        return rate
    return function

def inverseTimeLearningRate(rate, k=1):
    """Decays the learning rate as 1/t^k per learning session. The parameter
    sets the initial learning rate."""
    def function(t):
        return float(rate)/t**k
    return function

def randomInverseTimeLearningRate(rate):
    """At time t, selects the learning rate uniformly at random from the range
    [0, rate/t)."""
    def function(t):
        return random() * float(rate)/t
    return function

""" -- Training functions -- """

ERROR_THRESHOLD = 0.05  # Threshold for detecting errors from networks

def train(network, sourceFile, learningRateFunction, summary=True):
    """Main function for training networks."""
    dataSource = open(sourceFile)
    if isinstance(network, Perceptron):
        trainPerceptron(network, dataSource, learningRateFunction, summary)
    else:
        trainNetwork(network, dataSource, learningRateFunction, summary)

def trainPerceptron(perceptron, dataSource, learningRateFunction, summary=True):
    """Trains a perceptron according to the data points in a data source.
    Each line should contain n whitespace-delimited inputs (where n is the
    number of inputs of the perceptron) followed by one label (either 0 or 1 for
    a perceptron). For example,
        0.4 -1.2 5.2 1
    denotes a data point with features (0.4, -1.2, 5.2) in class 1. If summary
    is set to False, the session will not be summarized."""
    t = 0
    lines = dataSource.readlines()
    numUpdates = 0
    for line in lines:
        line = line[:-1] if line[-1] == '\n' else line
        t += 1
        learningRate = learningRateFunction(t)
        data = line.split()
        label = int(data[-1])
        inputs = list(map(int, data[:-1]))
        prediction = perceptron.run(inputs)[0]
        weights = perceptron.getWeights(perceptron.outputLayer[0])
        delta = learningRate * (label - prediction)
        if len(inputs) == len(weights) - 1:
            inputs = [1.0] + inputs  # Adjust for bias node
        newWeights = {w + delta * i for w, i in zip(weights, inputs)}
        if delta != 0:
            numUpdates += 1
        perceptron.setWeights(perceptron.outputLayer[0], newWeights)
    if summary:
        print("Training complete: " + str(t) + " samples, "
            + str(numUpdates) + " updates.")

def computeNodeErrors(net, inputs, actualOutputs):
    """Computes the partial error of the network for each node. For a node n,
    this is the derivative of n's activation function evaluation function
    evaluated on the sum of its weighted inputs times the sum of n's children's
    errors weighted by their weights to n. For output nodes, it is instead
    defined as the predicted output minus the actual output. Also returns the
    squared error of the network on the data point."""
    errors = {}
    squareError = 0.0
    predictedOutputs = net.run(inputs)
    for i, node in enumerate(net.outputLayer):
        errors[node] = predictedOutputs[i] - actualOutputs[i]
        squareError += errors[node] ** 2
    reversedLayers = net.hiddenLayers[:]
    reversedLayers.reverse()
    toBeComputed = []
    for layer in reversedLayers:
        toBeComputed += layer
    while len(toBeComputed) > 0:  # Repeat until all nodes processed
        newToBeComputed = []
        for node in toBeComputed:
            notComputable = False
            totalChildError = 0
            for child in node.children:
                if child not in errors:  # Cannot yet compute
                    notComputable = True
                    break
                childError = errors[child]
                weight = child.weights[node]
                totalChildError += childError * weight
            if notComputable:
                newToBeComputed.append(node)
            else:
                activationDerivative = DERIVATIVES[node.activationFunc]
                if activationDerivative is None:
                    raise TrainingError("Activation function of node "
                        + str(node) + " is non-differentiable.")
                weightedInput = node.getWeightedInputSum()
                nodeDerivative = activationDerivative(weightedInput)
                errors[node] = nodeDerivative * totalChildError
        toBeComputed = newToBeComputed
    return (errors, squareError)

def backpropagation(net, inputs, actualOutputs, learningRate):
    """Performs backpropagation training for a single training sample. Returns
    true if updates were made (i.e. if squareError is below a threshold); false
    otherwise."""
    nodeErrors, squareError = computeNodeErrors(net, inputs, actualOutputs)
    if squareError < ERROR_THRESHOLD:
        return False
    targetNodes = net.outputLayer[:]
    for layer in net.hiddenLayers:
        targetNodes += layer
    for node in targetNodes:
        newWeights = []
        for parent in node.inputs:
            parentInput = parent.process()
            weightChange = -1 * learningRate * parentInput * nodeErrors[node]
            currWeight = node.weights[parent]
            newWeights.append(currWeight + weightChange)
        net.setWeights(node, newWeights)
    return True

def trainNetwork(net, dataSource, learningRateFunction, summary=True):
    """Uses backpropagation to train a network on several training samples."""
    numInputs = len(net.inputLayer)
    t = 0
    lines = dataSource.readlines()
    numUpdates = 0
    for line in lines:
        line = line[:-1] if line[-1] == '\n' else line
        t += 1
        learningRate = learningRateFunction(t)
        data = list(map(int, line.split()))
        inputs = data[:numInputs]
        actualOutputs = data[numInputs:]
        if backpropagation(net, inputs, actualOutputs, learningRate):
            numUpdates += 1
    if summary:
        print("Training complete: " + str(t) + " samples, "
            + str(numUpdates) + " updates.")

def validateNetwork(net, dataSource, summary=True):
    """Runs validation on a network given data from a source file. Each line
    should contain n whitespace-delimited inputs (where n is the number of
    inputs to the network) followed by one label for the point. For example,
        0.4 -1.2 5.2 red
    denotes a data point with features (0.4, -1.2, 5.2) in class "red". If
    summary is set to False, the session will not be summarized."""
    lines = dataSource.readlines()
    total = len(lines)
    correct = 0.0
    for line in lines:
        line = line[:-1] if line[-1] == '\n' else line
        data = line.split()
        label = data[-1]
        inputs = list(map(int, data[:-1]))
        prediction = net.run(inputs)[0]
        if str(prediction) == label:
            correct += 1.0
    if summary:
        print("Validation complete: " + str(total) + " samples, " + str(correct)
            + " correct, " + str(correct*100/total) + "% accuracy.")
