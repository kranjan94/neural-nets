from struct import unpack
import sys

"""Reads data from the MNIST raw data files and writes them to training and
validation data files. run with
    python readMNISTData.py
or, to limit to N data points per file,
    python readMNISTData.py N
Note that training on all 60,000 data points may take a while, so it is
recommended to start with 1,000-5,000 points per run.
"""

maxItems = float('inf')  # Use to set max number of items per file
if len(sys.argv) > 1:
    maxItems = int(sys.argv[1])

def showImage(line, width, height):
    """Prints a data point from the MNIST data along with its label."""
    print(' ')
    data = line[:-1]
    label = line[-1]
    for row in range(height):
        rowData = data[:width]
        data = data[width:]
        rowData = list(map(lambda x: '#' if int(x) > 0 else ' ', rowData))
        print(''.join(rowData))
        print('LABEL: ' + label)

def parseData(labelFile, imageFile, targetFile):
    """Parses the image and label data files to write the data to a new file.
    The output file will have, for each data point, 28x28 space-delimited
    pixel values followed by the label for the point."""
    labelFile.read(4)
    numItems = min(unpack('>L', labelFile.read(4))[0], maxItems)
    imageFile.read(8)
    numRows, numCols = unpack('>LL', imageFile.read(8))
    for i in range(numItems):
        label = unpack('>B', labelFile.read(1))[0]
        pixels = []
        for _ in range(numRows * numCols / 4):  # Read data, 4 pixels at a time
            pixels += unpack('>BBBB', imageFile.read(4))
        pixels = list(map(lambda x: int(x > 0), pixels))
        dataPoint = pixels + [label]
        dataPoint = list(map(str, dataPoint))
        targetFile.write(' '.join(dataPoint) + '\n')
        if (i + 1) % 1000 == 0:
            print(str(i + 1) + ' items parsed.')

# Parse training data
trainLabelFile = open('train-labels-idx1-ubyte')
trainImageFile = open('train-images-idx3-ubyte')
trainTargetFile = open('mnistTrain.txt', 'w')
parseData(trainLabelFile, trainImageFile, trainTargetFile)
trainTargetFile.close()

# Parse test data
validateLabelFile = open('t10k-labels-idx1-ubyte')
validateImageFile = open('t10k-images-idx3-ubyte')
validateTargetFile = open('mnistValidate.txt', 'w')
parseData(validateLabelFile, validateImageFile, validateTargetFile)
validateTargetFile.close()
