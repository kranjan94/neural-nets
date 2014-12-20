from activationFunctions import identityActivation, zeroOneActivation, \
    sigmoidActivation, arctanActivation
from networkExceptions import InvalidInputNodeException, BadInputException

class Network(object):
    """Main class for a network. Keeps track of each layer of the network as
    well as its harness. Uses a sigmoid activation function by default,
    though others can be used (see activationFunctions.py)."""
    def __init__(self, numInput, numsHidden, numOutput, bias=False,
            inputActivationFunction=sigmoidActivation,
            hiddenActivationFunction=sigmoidActivation,
            outputActivationFunction=sigmoidActivation):
        self.harness = NetworkHarness()
        self.bias = BiasNode() if bias else False  # Initialize bias node
        # Initialize input layer
        self.inputLayer = []
        for j in range(numInput):
            node = InputNode(inputActivationFunction, self.harness)
            self.harness.registerInputNode(node)
            self.inputLayer.append(node)
        # Initialize hidden layers
        self.hiddenLayers = []
        for i in range(len(numsHidden)):
            layer = []
            prevLayer = self.inputLayer if i == 0 else self.hiddenLayers[i-1]
            for j in range(numsHidden[i]):
                node = HiddenNode(hiddenActivationFunction, self.bias)
                for parent in prevLayer:
                    node.registerParent(parent)
                layer.append(node)
            self.hiddenLayers.append(layer)
        # Initialize output layer
        self.outputLayer = []
        penultimateLayer = self.inputLayer
        if len(self.hiddenLayers) != 0:
            penultimateLayer = self.hiddenLayers[-1]
        for j in range(numOutput):
            node = OutputNode(outputActivationFunction, self.bias)
            self.harness.registerOutputNode(node)
            for parent in penultimateLayer:
                node.registerParent(parent)
            self.outputLayer.append(node)

    def __str__(self):
        out = "Network:\n\tInput layer:\t" + str(map(str, self.inputLayer))
        for i, layer in enumerate(self.hiddenLayers):
            layerString = str(map(str, layer))
            out += "\n\tHidden layer " + str(i + 1) + ":\t" + layerString
        out += "\n\tOutput layer:\t" + str(map(str, self.outputLayer))
        return out

    def getWeights(self, layer, nodeNumber):
        layers = [self.inputLayer] + self.hiddenLayers + [self.outputLayer]
        return layers[layer][nodeNumber].weights

    def setWeights(self, layer, nodeNumber, newWeights):
        layers = [self.inputLayer] + self.hiddenLayers + [self.outputLayer]
        layers[layer][nodeNumber].weights = newWeights

    def run(self, data):
        """Run data through the network."""
        return self.harness.run(data)

class Perceptron(Network):
    """Perceptron implementation as a special case of the Network class with no
    hidden layers and one output node. Uses a zero-one activation function and
    a bias input by default."""
    def __init__(self, numInputs, bias=True,
        inputActivationFunction=identityActivation,
        outputActivationFunction=zeroOneActivation):
        Network.__init__(self, numInputs, [], 1, bias,
        inputActivationFunction=identityActivation,
        outputActivationFunction=outputActivationFunction)

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
        return [node.process() for node in self.outputs]

class Node(object):
    """Represents a single node in the network."""
    def __init__(self, activationFunc, bias=False):
        self.activationFunc = activationFunc
        # If bias is False, ignore it; else, incorporate the bias node
        self.bias = bias
        self.weights = [1.0] if bias else []
        self.inputs = [bias] if bias else []

    def __str__(self):
        return str(self.__class__.__name__) + ": " + str(self.process())

    def registerParent(self, node):
        """Registers an input node and initializes its weight to 0."""
        self.weights.append(0.0)
        self.inputs.append(node)

    def process(self):
        """Grabs inputs to this node, computes net input, and returns output
        using the activation function. Implemented by each Node subclass."""
        pass


class InputNode(Node):
    """An input node. Only source of input is one input from the network
    harness."""
    def __init__(self, activationFunc, harness):
        Node.__init__(self, activationFunc, False)
        self.harness = harness

    def process(self):
        """Grabs input value from the network harness and passes it through
        the activation function."""
        return self.activationFunc(self.harness.getInputValue(self))


class HiddenNode(Node):
    """A node in a hidden layer. Grabs data from either the input layer or the
    previous hidden layer to compute its value."""
    def process(self):
        """Computes weighted sum of input nodes and returns the value after
        passing through the activation function."""
        weightedInput = 0.0
        for node, weight in zip(self.inputs, self.weights):
            rawInput = node.process()
            weightedInput += rawInput * weight
        return self.activationFunc(weightedInput)


class OutputNode(Node):
    """An output node. Grabs data from either the input layer or the last
    hidden layer to compute its value."""
    def process(self):
        """Computes weighted sum of input nodes and returns the value after
        passing through the activation function."""
        weightedInput = 0.0
        for node, weight in zip(self.inputs, self.weights):
            rawInput = node.process()
            weightedInput += rawInput * weight
        return self.activationFunc(weightedInput)

class BiasNode(Node):
    """Node that always returns a fixed value from its process() method. Only
    one instance should exist per Network."""
    def __init__(self, value=1.0):
        self.value = value

    def process(self):
        return self.value
