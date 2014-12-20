from math import exp, pi, atan

"Various types of activation functions. Each activation function takes a number as input and returns another number."""

def zeroOneActivation(arg):
    return int(arg >= 0)

def identityActivation(arg):
    return arg

def sigmoidActivation(arg):
    return (1 + exp(-1 * arg)) ** -1

def arctanActivation(arg):
    return atan(arg)/pi + 0.5

""" -- Derivatives -- """

def identityDerivative(arg):
    return 1

def sigmoidDerivative(arg):
    return sigmoidActivation(arg) * (1 - sigmoidActivation(arg))

def arctanDerivative(arg):
    return (pi * (arg ** 2 + 1)) ** -1

# This dictionary can be used to get the derivative for an activation function
# e.g. derivative = activationFunctions.DERIVATIVES[node.activationFunc]
DERIVATIVES = {
    zeroOneActivation: None,  # Nondifferentiable
    identityActivation: identityDerivative,
    sigmoidActivation: sigmoidDerivative,
    arctanActivation: arctanDerivative
}
