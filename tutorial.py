import sys
bypass = False
if len(sys.argv) > 1 and sys.argv[1] == "bypass":
    bypass = True

removeWhitespace = lambda s: "".join(s.split())  # Used for comparing code lines

def getCodeLine(line):
    """Prompts a user to enter a line of code. Once the line has been entered,
    returns it. Whitespaces are ignored."""
    print("\n\t" + line)
    if bypass:
        return line
    print("\n>>>"),
    userInput = raw_input()
    while removeWhitespace(userInput) != removeWhitespace(line):
        print("That doesn't look quite right. Try again:\n>>>"),
        userInput = raw_input()
    return userInput

print("\nWelcome to the tutorial for the neural-nets package!")
print("This tutorial will take you through some of the basic functionality"
    + " of the package and show you some of the options available to you.")
print("To exit, use ctrl+c. To bypass the interactive part of the tutorial "
    + "and have the text printed out, run with 'bypass' as a command line "
    + "argument: python tutorial.py bypass")

# Basic network construction

print("\nTo start, we need to import some classes from network.py. Type:")
exec(getCodeLine("from network import Network, Perceptron"))

print("\nGood! The Network class is the base class for all neural network "
    + "instances. The Perceptron class is just a subclass of the Network "
    + "class. The Network initializer takes these arguments, in order:")
print("\t-the number of input nodes")
print("\t-a list [h1, h2, ... ,hN] containing the number of nodes in each"
    + " of N hidden layers")
print("\t-the number of output nodes")
print("\t-a boolean indicating whether or not to include a bias node"
    + " (optional, False by default)")
print("\t-the activation function for the input nodes (optional, a sigmoid"
    + " function by default)")
print("\t-the activation function for the hidden nodes (optional, a sigmoid"
    + " function by default)")
print("\t-the activation function for the output nodes (optional, a sigmoid"
    + " function by default)")
print("For now, let's ignore the optional arguments and build a simple"
    + " network. Try:")
exec(getCodeLine("net = Network(3, [4, 3], 2)"))

print("\nNotice also that we didn't specify anything about the connections "
    + "between nodes. That's because, by default, the Network constructor "
    + "connects each node to every node in the preceding layer.")
print("Now, go ahead and print out the network you just created:")
exec(getCodeLine("print(net)"))

print("\nThe __str__ method for the Network class prints out each of the layers"
    + " of the network. Here, as expected, we have an input layer with 3 nodes,"
    + " one hidden layer with 4 nodes, one hidden layer with 3 nodes, and an "
    + "output layer with 2 nodes.")
print("The __str__ method of each Node gives you the type of the node, its "
    + "index, and the value it's currently outputting. For example, the second "
    + "node in the third line is a hidden node with an index of 2.2 currently "
    + "outputting 0.5 (because the default activation function is a sigmoid "
    + "function, for which 0.0 is sent to 0.5). A node's index, 'x1.x2', "
    + "indicates its layer (x1 = 0 for input, 1 for first hidden, etc.) and its"
    + " position within the layer (x2 = 1 for top node, 2 for 2nd from top, "
    + "etc.).")
print("The Perceptron class has a much simpler constructor. Recall that a "
    + "perceptron is a one-layer neural network with a zero-one (or 'step') "
    + "activation function on its sole output node. As such, the only required"
    + " parameter is the number of input nodes. Let's try 2:")
exec(getCodeLine("perceptron = Perceptron(2)"))

print("\nThis is your new perceptron:\n")
print(perceptron)
print("\nAs we expected, it has 2 input nodes and one output node.")

# Advanced network construction

print("\nYou can also specify that you want a bias input on the non-input "
    + "nodes of the network with the 'bias' argument. Let's give our "
    + "perceptron a bias input:")
exec(getCodeLine("perceptron = Perceptron(2, bias=True)"))

print("\nThe bias node won't show up in your layers; it's a singleton Node "
    + "that every non-input node can get input from, and it always returns "
    + "1.0 as its output.")
print("You can also control the activation functions for the input "
    + "layer, hidden layers, and output layer. This is done via the "
    + "'inputActivationFunction', 'hiddenActivationFunction', and "
    + "'outputActivationFunction' optional arguments. You can find a list "
    + "of currently supported functions in activationFunctions.py. Let's "
    + "grab one of them:")
exec(getCodeLine("from activationFunctions import arctanActivation"))

print("\nWe can, for example, make a network with the arctan activation "
    + "function on the hidden nodes like this:")
exec(getCodeLine("arctanNet = Network(3, [2], 2, hiddenActivationFunction="
    + "arctanActivation)"))

print("\nNote that, since the default activation function is a sigmoid "
    + "function, this network has 3 input nodes with sigmoid activations, "
    + "one hidden layer with 2 nodes with arctan activations, and 2 output "
    + "nodes with sigmoid activations.")
print("One last thing: you can also construct networks with arbitrary "
    + "connections, so long as they have no loops. You can do this with the "
    + "fromFile method in the Network class. This method takes a .network "
    + "file as an argument and returns a Network matching the topology "
    + "specified in the file. You can find instructions on writing these "
    + "files in networkFileReader.py, but we'll use "
    + "examples/exampleLayout.network for now:")
exec(getCodeLine("customNet = Network.fromFile('examples/"
    + "exampleLayout.network')"))

print("\nWe can get a closer look at this network with the nodeDetails method:")
exec(getCodeLine("customNet.nodeDetails()"))

print("\nNotice that the new topology described in exampleLayout.network is "
    + "reflected in the output.")

# Training

print("\nNow, let's try training our perceptron. Our training data is in the "
    + "file examples/train2.txt. This file contains 500 2-dimensional points, "
    + "(x, y), each labeled 1 if y > x and 0 if y <= x. We will attempt to use "
    + "our perceptron to learn this classification. First, we need the train "
    + "function:")
exec(getCodeLine("from train import train"))

print("\nThe train function uses the perceptron weight update algorithm to "
    + "train perceptrons and backpropagation to train general networks. "
    + "Before we proceed, though, we need a learning rate function. In this "
    + "package, learning rates take a time argument; for example, you can "
    + "specify a learning rate that is, say, 1/t, where t is the number of "
    + "samples seen in the training data so far. We'll do just that:")
exec(getCodeLine("from train import inverseTimeLearningRate"))

print("\nYou can find more learning rate functions in train.py.")
print("The train function takes a Network, a training data filename, and a "
    + "learning rate function as required arguments, so we can train our "
    + "perceptron like this:")
exec(getCodeLine("train(perceptron, 'examples/train2.txt', inverseTime"
    + "LearningRate(1))"))

# Validation

print("\nNow, we'll validate our network with the 500 data points in "
    + "examples/validation2.txt. First, we need the validate function:")
exec(getCodeLine("from train import validate"))

print("\nThis is used almost the same way as train:")
exec(getCodeLine("validate(perceptron, 'examples/validation2.txt')"))

print("\nNow, let's try training a general network:")
exec(getCodeLine("net = Network(2, [], 2)"))
exec(getCodeLine("train(net, 'examples/train2.txt', inverseTime"
    + "LearningRate(1))"))

print("\nAnd validating it:")
exec(getCodeLine("validate(net, 'examples/validation2.txt')"))

print("\nAlmost 90% accuracy with the general network and over 99% with the "
    + "perceptron. Not bad!")

print("\nThis concludes the tutorial. You can find more details about the "
    + "options available to you in network.py, train.py, and "
    + "activationFunctions.py. Good luck!")
