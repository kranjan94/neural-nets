from network import NetworkHarness, Node, InputNode, HiddenNode, OutputNode, \
    BiasNode
from networkExceptions import NetworkFileException

class NetworkFileReader(object):
    """Utility class for reading .network layout files. A .network file
    specifies the topology (but NOT the weights) for a network, and can be
    used to configure any arbitrary topology.
        -   The first line of the file should be a series of N >= 2 integers.
            The first and last integer specify the size of the input and output
            layers, respectively. The integers in between specify the sizes
            of each hidden layer, in order.
        -   The rest of the lines will read:
                <node1>: <parent1> <parent2> <parent3> ...
            where each <X> is the index of node X. The index of node X is a
            string, 'x1.x2'. x1 is the zero-indexed number of the layer for X,
            so the input layer has x1 = 0 and the output layer has x1 = N - 1.
            x2 is the one-indexed depth in the layer of X, so the topmost node
            has x2 = 1. For example, the 3rd node from the top in the second
            hidden layer would have index '2.3'.
    Lines starting with '%' are treated as comments and will be ignored. Blank
    lines will also be ignored. See examples/exampleLayout.network for an
    example of a .network file.
    """
    @staticmethod
    def read(filename, inputActivationFunction, hiddenActivationFunction,
            outputActivationFunction, bias=False):
        """Returns the layers, nodes, and harness of the new network."""
        if not filename.endswith('.network'):
            raise NetworkFileException(filename + " is not a .network file.")
        lines = open(filename).readlines()
        lines = [line for line in lines if line[0] != '%']  # Ignore comments
        lines = [line for line in lines if line != '\n']  # Ignore blank lines
        layerSizes = None
        try:
            layerSizes = list(map(int, lines[0].split()))
        except ValueError as e:
            raise NetworkFileException("Error reading layout file: " + str(e))
        if len(layerSizes) < 2:
            raise NetworkFileException("Not enough layers specified.")
        harness = NetworkHarness()
        nodes = {}
        bias = BiasNode() if bias else False
        # First, initialize layers with no connections
        layers = []
        for i, size in enumerate(layerSizes):
            layer = []
            for j in range(size):
                index = str(i) + '.' + str(j + 1)
                node = None
                if i == 0:
                    node = InputNode(inputActivationFunction, index, harness)
                    harness.registerInputNode(node)
                elif i == len(layerSizes) - 1:
                    node = OutputNode(outputActivationFunction, index, bias)
                    harness.registerOutputNode(node)
                else:
                    node = HiddenNode(hiddenActivationFunction, index, bias)
                layer.append(node)
                nodes[index] = node
            layers.append(layer)
        # Second, use the input file to make connections.
        try:
            for line in lines[1:]:
                line = line[:-1] if line[-1] == '\n' else line
                nodeIndex, inputIndices = line.split(': ')
                if nodeIndex not in nodes:
                    raise NetworkFileException("Node " + nodeIndex + " not "
                        + "requested.")
                targetNode = nodes[nodeIndex]
                for index in inputIndices.split():
                    if index not in nodes:
                        raise NetworkFileException("Node " + index + " not "
                            + "requested.")
                    parent = nodes[index]
                    targetNode.registerParent(parent)
        except Exception as e:
            raise NetworkFileException("Error reading layout file: " + str(e))
        return (layers, nodes, harness)
