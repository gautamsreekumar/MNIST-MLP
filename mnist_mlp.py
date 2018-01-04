import numpy as np
import os, struct
from sys import *
from array import array
from cvxopt.base import matrix
import numpy.random as nprandom

def sigmoid(n): # sigmoid function
    return 1.0/(1.0+np.exp(-1*n))

def softmax(n): # softmax function
    temp = n.copy()
    sum_exp = np.sum(np.exp(n))
    for i in range(0, len(n)):
        temp[i] = np.exp(n[i])/sum_exp
    return temp

# reading MNIST dataset
# code taken from Alex Kesling (akesling) in  Github
def read(digits, dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    images =  matrix(0, (len(ind), rows*cols))
    labels = matrix(0, (len(ind), 1))
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i] = lbl[ind[i]]

    return images, labels

# defining neural network
def test_nn(W1, W2, b1, b2):
    m = 10000
    digit_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    images, labels = read(digit_list, dataset = "testing")
    images = np.array(images)
    labels = np.array(labels)
    y = np.zeros((10, m))
    for i in range(0, labels.shape[0]):
        y[labels[i], i] = 1.0

    correct = 0
    for i in range(0, m): # one testing example at a time
        x_temp = images[i].reshape(784, 1)
        y_temp = y[:, i].reshape(10, 1)

        a1 = np.dot(W1, x_temp)+b1
        h1 = sigmoid(a1)
        a2 = np.dot(W2, h1)+b2
        h2 = softmax(a2)
        
        if y_temp[np.argmax(h2)] == 1:
            correct += 1
    accuracy = float(correct)/m
    print "Accuracy = ", accuracy
    return accuracy

digit_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
images, labels = read(digit_list)
images = np.array(images)
labels = np.array(labels)
y = np.zeros((10, 60000))
for i in range(0, labels.shape[0]):
    y[labels[i], i] = 1.0

m = 60000 # number of training examples

alpha = float(argv[1]) # learning rate

# implementing a two layer neural network
pixels = 28
input_layer_size = pixels*pixels # input layer size
layer_1_size = 20 # pixels*pixels # size of hidden layer 1
output_layer_size = 10 # size of the output layer
layer_2_size = 10

scaling_factor = 0.01 # scaling down the initial weights and biases
# these initializations are required
W1 = scaling_factor*nprandom.randn(layer_1_size, input_layer_size) # weight matrix 1
W2 = scaling_factor*nprandom.randn(layer_2_size, layer_1_size) # weight matrix 2
b1 = np.zeros((layer_1_size, 1)) # bias 1
b2 = np.zeros((layer_2_size, 1)) # bias 2
gradW1 = np.zeros((layer_1_size, input_layer_size)) # W1 gradient
gradW2 = np.zeros((layer_2_size, layer_1_size)) # W2 gradient
gradb1 = np.zeros((layer_1_size, 1)) # bias1 gradient
gradb2 = np.zeros((layer_2_size, 1)) # bias2 gradient
a1 = np.zeros((layer_1_size, 1))
a2 = np.zeros((layer_2_size, 1))
h1 = np.zeros((layer_1_size, 1))
h2 = np.zeros((layer_2_size, 1))
delta1 = np.zeros((layer_1_size, 1))
delta2 = np.zeros((layer_2_size, 1))

# there are three layers : one input layer of size 784x1, one hidden layer of size 784x1, and one output layer of size 10x1

reg_factor = 1 # regularization factor. reg_factor = 0 means 'No regularization'
batch_size = 1000 # batch size for mini-batch gradient
total_iteration = 10 # training will be carried out 'total_iteration' number of times
accuracy = np.zeros(total_iteration)

for iteration in range(0, total_iteration):
    for i in range(0, m): # for each training examples
        x_temp = images[i].reshape(784, 1)
        y_temp = y[:, i].reshape(10, 1)

        a1 = np.dot(W1, x_temp)+b1
        h1 = sigmoid(a1)
        a2 = np.dot(W2, h1)+b2
        h2 = softmax(a2)

        delta2 = h2-y_temp
        gradW2 += np.dot(delta2, h1.transpose())
        delta1 = h1*(1-h1)*(np.sum(delta2*W2, 0).reshape((layer_1_size, 1)))
        gradW1 += np.dot(delta1, x_temp.transpose())
        gradb1 += delta1
        gradb2 += delta2
        
        if (i % batch_size) == 0: # mini-batch training
            W1 -= alpha*gradW1/batch_size # updating weight matrix 1
            W2 -= alpha*gradW2/batch_size # updating weight matrix 2
            b1 -= alpha*gradb1/batch_size # updating bias 1
            b2 -= alpha*gradb2/batch_size # updating bias 2
            gradW1 = np.zeros((layer_1_size, input_layer_size)) # W1 gradient
            gradW2 = np.zeros((layer_2_size, layer_1_size)) # W2 gradient
            gradb1 = np.zeros((layer_1_size, 1)) # bias1 gradient
            gradb2 = np.zeros((layer_2_size, 1)) # bias2 gradient
            error_function = sum(-1.0*y_temp*np.log(h2)) # error function
    accuracy[iteration] = test_nn(W1, W2, b1, b2)
    print "Epoch ", iteration+1, "Accuracy ", accuracy[iteration]
