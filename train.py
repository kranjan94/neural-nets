from random import random

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

def train(network, sourceFile, learningRateFunction, summary=True):
    """Main function for training networks."""
    dataSource = open(sourceFile)
    if isinstance(network, Perceptron):
        trainPerceptron(network, dataSource, learningRateFunction, summary)
    else:
        pass  # TODO: General networks

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
        weights = perceptron.getWeights(-1, 0)
        delta = learningRate * (label - prediction)
        if len(inputs) == len(weights) - 1:
            inputs = [1.0] + inputs  # Adjust for bias node
        newWeights = [w + delta * i for w, i in zip(weights, inputs)]
        if delta != 0:
            numUpdates += 1
        perceptron.setWeights(-1, 0, newWeights)
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
