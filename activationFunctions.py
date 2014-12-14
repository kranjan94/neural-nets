from math import exp
from math import atan
from math import pi

"Each activation function takes a number as input and returns another number."

def zeroOneActivation(arg):
    return int(arg > 0)

def sigmoidActivation(arg):
    return (1 + exp(-1 * arg)) ** -1

def arcTanActivation(arg):
    return atan(arg)/pi + 0.5
