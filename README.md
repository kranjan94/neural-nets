Neural Networks
===========

This is a Python neural network implementation.

A neural network is a type of machine learning algorithm that takes loose inspiration from networks of neurons in animals. It consists of a set of nodes interconnected by weighted edges. Each node has a set of inputs from which it computes an output using an activation function. By using complex topologies of nodes and different types of learning schemes and various other parameters, one can use neural networks to perform a variety of learning and classification tasks.

Example usage is shown in example.py, using training and validation data in the examples directory. You can also run tutorial.py to get a walkthrough of some of the features of the package.

    python example.py
    python tutorial.py

###Constructing a Network
Construction is typically done directly via the `Network` class constructor. The constructor generates a fully connected feed-forward network with any number of hidden layers, adjustable activation functions, and an optional bias input:

    def __init__(self, numInput=0, numsHidden=[], numOutput=0, bias=False, validLabels=None,
        inputActivationFunction=sigmoidActivation, hiddenActivationFunction=sigmoidActivation,
        outputActivationFunction=sigmoidActivation):
`numInput` and `numOutput` are integers, while `numsHidden` is a list of integers denoting the number of nodes in each hidden layer; for example, `numsHidden=[3,4,2]` signals the constructor to include 3 hidden layers with 3, 4, and 2 nodes, respectively from input to output. `bias` should be True if a bias input on all hidden and output nodes is desired. `activationFunctions.py` contains several activation functions that can be used for each of the last three arguments. Note that, by default, there are no hidden nodes and no bias inputs and the activation function for all nodes is a sigmoid function.

`Perceptron` is a subclass of `Network`. It provides a simple way of building a 1-layer network with an arbitrary number of inputs, a bias input, and step activation on the output node. The only required argument is the number of input nodes, so a 2-input perceptron can be constructed with just:

    perceptron = Perceptron(2)

####Constructing Arbitrary Topologies
Instead of using fully connected networks, you can specify arbitrary topologies using the `fromFile` method in `Network`. To use this method, you need to write a .network file. A .network file specifies the topology and all of the connections (and, optionally, the output labels) of a network. To write the file, follow these steps:
* Blank lines and lines beginning with '%' will be ignored, so lines beginning with '%' can be used as comments.
* The first line of the file should be a series of space-delimited integers `I H1 H2 ... HN O`, specifying the number of nodes in the input, each of N hidden, and output layers, respectively.
* Anywhere after the first line, you may have a line starting with 'LABELS: ' followed by one label for each output node:

        LABELS: red blue green yellow
There must be exactly as many labels as output nodes or an exception will be raised.
* Every other line specifies the parents of a single node. Each node has an index `x1.x2` where `x1` is the layer to which the row belongs (0 being the input layer, 1 being the first hidden layer or possibly the output layer, etc.) and `x2` is the number of the node, starting with 1, within that layer. For example, the 4th node from the top in the 2nd hidden layer has index `2.4`. Now, each parent specification line begins with the index of the node being specified followed by a semicolon and a list of the indices of parent nodes: 

        x1.x2: a1.a2 b1.b2 c1.c2 ... 
(where `a1.a2`, `b1.b2`, etc. are the parent nodes). A node need not only have parents in the previous layer; for example, a node in the 3rd hidden layer could have parents in the 1st hidden layer or even in the input layer (in fact, a node may even have parents in later layers, so long as no loops are created). Note that there must be no (directed) loops in the network or an exception will be raised during construction.

`fromFile` takes a .network filename as a required argument and returns the network specified. An example of a .network file is under `examples/exampleLayout.network`.

####Examples

    net = Network(3, [], 2)
Constructs a network with no hidden nodes, 3 inputs, 2 outputs, no bias input, and sigmoid activations on its outputs.

    net = Network(3, [4, 5], 2, bias=True)
Constructs a network with 3 inputs, 2 outputs, a hidden layer with 4 nodes, a hidden layer with 5 nodes, and a bias input and sigmoid activations on all hidden and output nodes.

    net = Network(3, [], 2, validLabels=['red', 'blue'])
Constructs a network like that in the first example, but with labels of 'red' and 'blue' associated with the output nodes.

    net = Perceptron(4)
Constructs a 4-input perceptron as a special instance of a Network.

    net = Network.fromFile('examples/exampleLayout.network')
Constructs the network specified in the exampleLayout file with default settings.
