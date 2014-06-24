import os
import sys
import time
import cPickle      as pkl
import numpy        as np
import numpy.random as npr

import kayak

num_epochs = 10
batch_size = 256
learn_rate = 0.0001

mnist_filename = '~/Data/MNIST/mnist.pkl'

with open(os.path.expanduser(mnist_filename)) as fh:
    train_set, valid_set, test_set = pkl.load(fh)

# Load up the training data and construct one-hot encoding.
train_X = train_set[0]
N, D    = train_X.shape
train_Y = np.zeros((N,10))
train_Y[np.arange(N),train_set[1]] = 1

# Create the batcher.
batcher = kayak.Batcher(batch_size, N)

# Create input and target objects.
inputs  = kayak.Inputs(train_X, batcher)
targets = kayak.Targets(train_Y, batcher)

# Simple logistic regression to start.
weights = kayak.Parameter( 0.01*npr.randn(D, 10) )
biases  = kayak.Parameter( 0.01*npr.randn(1, 10) )

# Compute the output.
output  = kayak.LogSoftMax( kayak.ElemAdd( kayak.MatMult( inputs, weights ), biases ) )

# Loss function.
loss    = kayak.LogMultinomialLoss( output, targets )

# Overall objective.
objective = kayak.MatSum( loss )

# Check gradients.
#print "weights:", kayak.util.checkgrad(weights, objective)
#print "biases: ", kayak.util.checkgrad(biases, objective)

for epoch in xrange(num_epochs):
    t0 = time.time()
    overall = 0.0
    for batch in batcher:
        overall += objective.value(True)
        grad_weights = objective.grad(weights)
        grad_biases  = objective.grad(biases)

        weights.add( -learn_rate * grad_weights )
        biases.add( -learn_rate * grad_biases )
    print overall, time.time() - t0


