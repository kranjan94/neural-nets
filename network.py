from activationFunctions import zeroOneActivation
from activationFunctions import sigmoidActivation
from activationFunctions import arcTanActivation
from networkExceptions import InvalidInputNodeException

class Network(object):
    """Main class for a network. Keeps track of each layer of the network as
    well as its harness. Uses a sigmoid activation function by default,
    though others can be used (see activationFunctions.py)."""
    def __init__(self, numInput, numsHidden, numOutput,
            inputActivationFunction=sigmoidActivation,
            hiddenActivationFunction=sigmoidActivation,
            outputActivationFunction=sigmoidActivation):
        self.harness = NetworkHarness()
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
                node = HiddenNode(hiddenActivationFunction)
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
            node = OutputNode(outputActivationFunction)
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

    def run(self, data):
        """Run data through the network."""
        return self.harness.run(data)

class Perceptron(Network):
    """Perceptron implementation as a special case of the Network class with no
    hidden layers and one output node. Uses a zero-one activation function by
    default."""
    def __init__(self, numInputs, inputActivationFunction=zeroOneActivation,
        outputActivationFunction=zeroOneActivation):
        Network.__init__(self, numInputs, [], 1, inputActivationFunction, None,
            outputActivationFunction)

class NetworkHarness(object):
    """A NetworkHarness provides the framework for a neural network. It feeds
    data to the input nodes and runs data through the network to the output
    layer."""
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.inputNodesToIndices = {}

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
            raise InvalidInputNodeException(str(node) + " is not a registered "
                + "input node with this harness.")
        idx = self.inputNodesToIndices[node]
        return self.inputs[idx]

    def run(self, data):
        """Load a new set of data into the inputs and propogate through the
        network to the outputs."""
        self.inputs = data
        return [node.process() for node in self.outputs]

class Node(object):
    """Represents a single node in the network."""
    def __init__(self, activationFunc):
        self.activationFunc = activationFunc
        self.weights = []
        self.inputs = []

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
        Node.__init__(self, activationFunc)
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
