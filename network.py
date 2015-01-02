from activationFunctions import identityActivation, zeroOneActivation, \
    sigmoidActivation, arctanActivation
from networkExceptions import InvalidInputNodeException, BadInputException, \
    NetworkFileException
from math import exp
from random import random

class Network(object):
    """Main class for a network. Keeps track of each layer of the network as
    well as its harness. Uses a sigmoid activation function by default,
    though others can be used (see activationFunctions.py)."""
    def __init__(self, numInput=0, numsHidden=[], numOutput=0, bias=False,
            validLabels=None,
            inputActivationFunction=sigmoidActivation,
            hiddenActivationFunction=sigmoidActivation,
            outputActivationFunction=sigmoidActivation):
        """Basic initializer. Initializes a network with a given number of input
        nodes, hidden layers and hidden nodes, and output nodes. This method
        initializes a fully connected network (i.e. every node is connected to
        every node in the previous layer) by default. More complex topologies
        can be built via .network files (see below)."""
        self.harness = NetworkHarness()
        self.nodes = {}  # Maps node indices to Node objects
        self.bias = BiasNode() if bias else False  # Initialize bias node
        self.validLabels = validLabels
        if validLabels == None:
            self.validLabels = [str(i) for i in range(numOutput)]
        # Initialize input layer
        self.inputLayer = []
        for j in range(numInput):
            index = '0.' + str(j + 1)
            node = InputNode(inputActivationFunction, index, self.harness)
            self.harness.registerInputNode(node)
            self.inputLayer.append(node)
            self.nodes[index] = node
        # Initialize hidden layers
        self.hiddenLayers = []
        for i in range(len(numsHidden)):
            layer = []
            prevLayer = self.inputLayer if i == 0 else self.hiddenLayers[i-1]
            for j in range(numsHidden[i]):
                index = str(i + 1) + '.' + str(j + 1)
                node = HiddenNode(hiddenActivationFunction, index, self.bias)
                for parent in prevLayer:
                    node.registerParent(parent)
                layer.append(node)
                self.nodes[index] = node
            self.hiddenLayers.append(layer)
        # Initialize output layer
        self.outputLayer = []
        penultimateLayer = self.inputLayer
        if len(self.hiddenLayers) != 0:
            penultimateLayer = self.hiddenLayers[-1]
        outputLayerNumber = len(self.hiddenLayers) + 1
        for j in range(numOutput):
            index = str(outputLayerNumber) + '.' + str(j + 1)
            node = OutputNode(outputActivationFunction, index, self.bias)
            self.harness.registerOutputNode(node)
            for parent in penultimateLayer:
                node.registerParent(parent)
            self.outputLayer.append(node)
            self.nodes[index] = node

    def __str__(self):
        """Prints each layer of the network, from input to output."""
        out = "Network:\n\tInput layer:\t" + str(map(str, self.inputLayer))
        for i, layer in enumerate(self.hiddenLayers):
            layerString = str(map(str, layer))
            out += "\n\tHidden layer " + str(i + 1) + ":\t" + layerString
        out += "\n\tOutput layer:\t" + str(map(str, self.outputLayer))
        return out

    def nodeDetails(self):
        """Prints the weight vector for every node in the network."""
        print("Input layer:")
        for node in self.inputLayer:
            print("\t" + str(node) + ": (connected to input harness)")
        for num, layer in enumerate(self.hiddenLayers):
            print("Hidden layer " + str(num + 1) + ":")
            for node in layer:
                print("\t" + str(node) + ": " + str(node.weightVectorString()))
        print("Output layer:")
        for node in self.outputLayer:
            print("\t" + str(node) + ": " + str(node.weightVectorString()))

    @staticmethod
    def fromFile(filename, bias=False,
        inputActivationFunction=sigmoidActivation,
        hiddenActivationFunction=sigmoidActivation,
        outputActivationFunction=sigmoidActivation):
        """Initializes a network from a .network file. See networkFileReader.py
        for instructions on creating these files."""
        from networkFileReader import NetworkFileReader
        layers, nodes, harness, labels = NetworkFileReader.read(filename,
            inputActivationFunction, hiddenActivationFunction,
            outputActivationFunction, bias)
        network = Network()
        network.harness, network.nodes, network.bias = harness, nodes, bias
        network.inputLayer, network.outputLayer = layers[0], layers[-1]
        network.hiddenLayers = [] if len(layers) < 3 else layers[1:-1]
        network.validLabels = labels
        try:
            network.run([0.0] * len(network.inputLayer))
        except RuntimeError:
            raise NetworkFileException("Loop detected in " + filename + ".")
        return network

    def getWeights(self, node):
        """Returns the weights of a node, ordered by the order in which its
        inputs were registered."""
        return [node.weights[i] for i in node.inputs]

    def setWeights(self, node, newWeights):
        """Sets the weights of a node according to newWeights. newWeights should
        have the weights ordered as in getWeights (e.g. if this node is node j
        and the input list is [i1, i2, i3], then the newWeights vector should be
        [w1j, w2j, w3j])."""
        node.weights = {n:weight for n, weight in zip(node.inputs, newWeights)}

    def run(self, data):
        """Run data through the network and returns the label with the highest
        softmax score."""
        scores = self.harness.run(data)
        exponentials = [exp(s) for s in scores]
        total = sum(exponentials)
        softmaxScores = [x/total for x in exponentials]
        maxIndex = softmaxScores.index(max(softmaxScores))
        return self.validLabels[maxIndex]


class Perceptron(Network):
    """Perceptron implementation as a special case of the Network class with no
    hidden layers and one output node. Uses a zero-one activation function and
    a bias input by default."""
    def __init__(self, numInputs, bias=True,
        inputActivationFunction=identityActivation,
        outputActivationFunction=zeroOneActivation):
        Network.__init__(self, numInputs, [], 1, bias, validLabels=['0','1'],
        inputActivationFunction=identityActivation,
        outputActivationFunction=outputActivationFunction)

    def run(self, data):
        """Overrides the general Network run() method to return 0 or 1."""
        return str(self.harness.run(data)[0])


BYPASS = -1  # When passed to a node's process method, uses last output.

class NetworkHarness(object):
    """A NetworkHarness provides the framework for a neural network. It feeds
    data to the input nodes and runs data through the network to the output
    layer."""
    def __init__(self):
        self.inputs = []
        self.inputNodesToIndices = {}
        self.outputs = []

    def registerInputNode(self, node):
        """Registers an input node for the network with this harness."""
        self.inputNodesToIndices[node] = len(self.inputs)
        self.inputs.append(0.0)

    def registerOutputNode(self, node):
        """Registers an output node for the network with this harness."""
        self.outputs.append(node)

    def getInputValue(self, node):
        """Called by a registered input node to get the latest input value for
        that node."""
        if node not in self.inputNodesToIndices:
            raise InvalidInputNodeException(node)
        idx = self.inputNodesToIndices[node]
        return self.inputs[idx]

    def run(self, data):
        """Load a new set of data into the inputs and propogate through the
        network to the outputs."""
        if len(data) != len(self.inputs):
            raise BadInputException(len(self.inputs), len(data))
        self.inputs = data
        callingSignature = random()
        return [node.process(callingSignature) for node in self.outputs]


class Node(object):
    """Represents a single node in the network."""
    def __init__(self, activationFunc, index ,bias=False):
        self.activationFunc = activationFunc
        self.index = index  # A node's index is its unique identifier
        # If bias is False, ignore it; else, incorporate the bias node
        self.bias = bias
        self.currentValue = 0.0
        self.lastSignature = None  # Used for memoization
        self.weights = {bias: 0.0} if bias else {}
        self.inputs = [bias] if bias else []
        self.children = set()

    def __str__(self):
        """The str() method for a Node prints its type, its index, and its
        current output value."""
        typeName = str(self.__class__.__name__)
        return typeName + " " + self.index + ": " + str(self.process(BYPASS))

    def registerParent(self, node):
        """Registers an input node and initializes its weight to 0. Also
        registers self as a child of the node."""
        self.inputs.append(node)
        self.weights[node] = 0.0
        node.registerChild(self)

    def registerChild(self, node):
        """Registers a node as a child of this node."""
        self.children.add(node)

    def getWeightedInputSum(self, callingSignature):
        """Returns the weighted sum of the inputs to this node."""
        weightedInput = 0.0
        for node in self.inputs:
            rawInput = node.process(callingSignature)
            weightedInput += rawInput * self.weights[node]
        return weightedInput

    def weightVectorString(self):
        """Returns a simplified string version of the weight vector."""
        vector = [node.index + ": " + str(self.weights[node]) \
            for node in self.weights]
        vector.sort()
        return "(" + ", ".join(vector) + ")"

    def process(self, callingSignature=None):
        """Grabs inputs to this node, computes net input, and returns output
        using the activation function. Implemented by each Node subclass.
        callingSignature is passed in by the harness on each run; this enables
        a 'caching' functionality from later nodes, as they need not recalculate
        values that have already been calculated for the same run."""
        pass

class InputNode(Node):
    """An input node. Only source of input is one input from the network
    harness."""
    def __init__(self, activationFunc, index, harness):
        Node.__init__(self, activationFunc, index, False)
        self.harness = harness

    def process(self, callingSignature=None):
        """Grabs input value from the network harness and passes it through
        the activation function."""
        if callingSignature in [self.lastSignature, BYPASS]:
            return self.currentValue
        else:
            newVal = self.activationFunc(self.harness.getInputValue(self))
            self.currentValue = newVal
            self.lastSignature = callingSignature
            return newVal


class HiddenNode(Node):
    """A node in a hidden layer. Grabs data from either the input layer or the
    previous hidden layer to compute its value."""
    def process(self, callingSignature):
        """Computes weighted sum of input nodes and returns the value after
        passing through the activation function."""
        if callingSignature in [self.lastSignature, BYPASS]:
            return self.currentValue
        else:
            newVal = self.activationFunc(\
                self.getWeightedInputSum(callingSignature))
            self.currentValue = newVal
            self.lastSignature = callingSignature
            return newVal


class OutputNode(Node):
    """An output node. Grabs data from either the input layer or the last
    hidden layer to compute its value."""
    def process(self, callingSignature=None):
        """Computes weighted sum of input nodes and returns the value after
        passing through the activation function."""
        if callingSignature in [self.lastSignature, BYPASS]:
            return self.currentValue
        else:
            newVal = self.activationFunc(\
                self.getWeightedInputSum(callingSignature))
            self.currentValue = newVal
            self.lastSignature = callingSignature
            return newVal


class BiasNode(Node):
    """Node that always returns a fixed value from its process() method. Only
    one instance should exist per Network."""
    def __init__(self, value=1.0):
        self.value = value
        self.index = "bias"

    def process(self, callingSignature=None):
        return self.value
