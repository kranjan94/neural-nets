"""Exceptions thrown by Networks and associated classes."""

class InvalidInputNodeException(Exception):
    """Signals that a node that is not a registered input node tried to get
    data from a NetworkHarness."""
    def __init__(self, node):
        self.message = str(node) + " is not a registered input node with " \
            + "this harness."

    def __str__(self):
        return repr(self.message)

class BadInputException(Exception):
    """Signals that the input to a network was not the right size."""
    def __init__(self, expectedLength, actualLength):
        self.message = "Incorrect number of inputs: expected " \
            + str(expectedLength) + ", received " + str(actualLength)

    def __str__(self):
        return repr(self.message)

class NetworkFileException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

class TrainingError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
