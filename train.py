from random import random
from network import Perceptron, BYPASS
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

def exponentialLearningRate(base):
    """At time t, returns the learning rate base^(t-1)."""
    def function(t):
        return base ** (t-1)
    return function

""" -- Training functions -- """

def train(network, sourceFile, learningRateFunction, summary=True):
    """Main function for training networks."""
    dataSource = open(sourceFile)
    if isinstance(network, Perceptron):
        trainPerceptron(network, dataSource, learningRateFunction, summary)
    else:
        trainNetwork(network, dataSource, learningRateFunction, summary)

def validate(network, sourceFile, summary=True):
    """Main function for validating networks."""
    dataSource = open(sourceFile)
    validateNetwork(network, dataSource, summary)

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
        label = data[-1]
        inputs = list(map(int, data[:-1]))
        prediction = perceptron.run(inputs)
        weights = perceptron.getWeights(perceptron.outputLayer[0])
        error = int(label) - int(prediction)
        delta = learningRate * error
        if len(inputs) == len(weights) - 1:
            inputs = [1.0] + inputs  # Adjust for bias node
        newWeights = [w + delta * i for w, i in zip(weights, inputs)]
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
    label = net.run(inputs)
    predictedOutputs = [1.0 if l == label else 0.0 for l in net.validLabels]
    for i, node in enumerate(net.outputLayer):
        errors[node] = predictedOutputs[i] - actualOutputs[i]
        squareError += errors[node] ** 2
    if squareError == 0.0:  # No need to continue
        return ({node:0.0 for node in net.nodes}, 0.0)
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
                weightedInput = node.getWeightedInputSum(BYPASS)
                nodeDerivative = activationDerivative(weightedInput)
                errors[node] = nodeDerivative * totalChildError
        toBeComputed = newToBeComputed
    return (errors, squareError)

def backpropagation(net, inputs, actualOutputs, learningRate):
    """Performs backpropagation training for a single training sample. Returns
    true if updates were made (i.e. if squareError is 0); false otherwise."""
    nodeErrors, squareError = computeNodeErrors(net, inputs, actualOutputs)
    if squareError == 0.0:
        return False
    targetNodes = net.outputLayer[:]
    for layer in net.hiddenLayers:
        targetNodes += layer
    for node in targetNodes:
        newWeights = []
        for parent in node.inputs:
            parentInput = parent.process(BYPASS)
            weightChange = -1 * learningRate * parentInput * nodeErrors[node]
            currWeight = node.weights[parent]
            newWeights.append(currWeight + weightChange)
        net.setWeights(node, newWeights)
    return True

def trainNetwork(net, dataSource, learningRateFunction, summary=True):
    """Uses backpropagation to train a network on several training samples."""
    t = 0
    lines = dataSource.readlines()
    numUpdates = 0
    labels = net.validLabels
    for line in lines:
        line = line[:-1] if line[-1] == '\n' else line
        t += 1
        learningRate = learningRateFunction(t)
        rawData = line.split()
        inputs = list(map(float, rawData[:-1]))
        actualLabel = rawData[-1]
        outputVector = [float(l == actualLabel) for l in labels]
        if backpropagation(net, inputs, outputVector, learningRate):
            numUpdates += 1
        if t % 100 == 0:
            print(str(t) + ' training rounds performed; ' + str(numUpdates)
                + ' updates so far.')
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
    correct = 0
    t = 0
    for line in lines:
        t += 1
        line = line[:-1] if line[-1] == '\n' else line
        data = line.split()
        label = data[-1]
        inputs = list(map(float, data[:-1]))
        prediction = net.run(inputs)
        if str(prediction) == label:
            correct += 1
        if t % 100 == 0:
            print(str(t) + ' validation rounds performed; ' + str(t - correct)
                + ' errors so far.')
    if summary:
        print("Validation complete: " + str(total) + " samples, " + str(correct)
            + " correct, " + str(correct*100/total) + "% accuracy.")
