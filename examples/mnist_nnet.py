import sys
import time
import data
import numpy        as np
import numpy.random as npr

sys.path.append('..')

import kayak

batch_size     = 256
learn_rate     = 0.01
momentum       = 0.9
layer1_sz      = 500
layer2_sz      = 500
layer1_dropout = 0.25
layer2_dropout = 0.25

npr.seed(1)

# Load in the MNIST data.
train_images, train_labels, test_images, test_labels = data.mnist()

# Turn the uint8 images into floating-point vectors.
train_images = np.reshape(train_images,
                          (train_images.shape[0],
                           train_images.shape[1]*train_images.shape[2]))/255.0

# Use one-hot coding for the labels.
train_labels = kayak.util.onehot(train_labels)
test_labels  = kayak.util.onehot(test_labels)

# Hand the training data off to a cross-validation object.
# This will create ten folds and allow us to easily iterate.
CV = kayak.CrossValidator(10, train_images, train_labels)

# Here I define a nice little training function that takes inputs and targets.
def train(inputs, targets):

    # Create a batcher object.
    batcher = kayak.Batcher(batch_size, inputs.shape[0])

    # Inputs and targets need access to the batcher.
    X = kayak.Inputs(inputs, batcher)
    T = kayak.Targets(targets, batcher)

    # First-layer weights and biases, with random initializations.
    W1 = kayak.Parameter( 0.1*npr.randn( inputs.shape[1], layer1_sz ))
    B1 = kayak.Parameter( 0.1*npr.randn(1, layer1_sz) )

    # First hidden layer: ReLU + Dropout
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), layer1_dropout)

    # Second-layer weights and biases, with random initializations.
    W2 = kayak.Parameter( 0.1*npr.randn( layer1_sz, layer2_sz ))
    B2 = kayak.Parameter( 0.1*npr.randn(1, layer2_sz) )

    # Second hidden layer: ReLU + Dropout
    H2 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(H1, W2), B2)), layer2_dropout)

    # Output layer weights and biases, with random initializations.
    W3 = kayak.Parameter( 0.1*npr.randn( layer2_sz, 10 ))
    B3 = kayak.Parameter( 0.1*npr.randn(1, 10) )

    # Output layer.
    Y = kayak.LogSoftMax( kayak.ElemAdd(kayak.MatMult(H2, W3), B3) )

    # The training loss is negative multinomial log likelihood.
    loss = kayak.MatSum(kayak.LogMultinomialLoss(Y, T))

    # Use momentum for the gradient-based optimization.
    mom_grad_W1 = np.zeros(W1.shape())
    mom_grad_W2 = np.zeros(W2.shape())
    mom_grad_W3 = np.zeros(W3.shape())

    # Loop over epochs.
    for epoch in xrange(10):

        # Track the total loss.
        total_loss = 0.0

        # Loop over batches -- using batcher as iterator.
        for batch in batcher:

            # Compute the loss of this minibatch by asking the Kayak
            # object for its value and giving it reset=True.
            total_loss += loss.value(True)

            # Now ask the loss for its gradient in terms of the
            # weights and the biases -- the two things we're trying to
            # learn here.
            grad_W1 = loss.grad(W1)
            grad_B1 = loss.grad(B1)
            grad_W2 = loss.grad(W2)
            grad_B2 = loss.grad(B2)
            grad_W3 = loss.grad(W3)
            grad_B3 = loss.grad(B3)
            
            # Use momentum on the weight gradients.
            mom_grad_W1 = momentum*mom_grad_W1 + (1.0-momentum)*grad_W1
            mom_grad_W2 = momentum*mom_grad_W2 + (1.0-momentum)*grad_W2
            mom_grad_W3 = momentum*mom_grad_W3 + (1.0-momentum)*grad_W3

            # Now make the actual parameter updates.
            W1.add( -learn_rate * mom_grad_W1 )
            B1.add( -learn_rate * grad_B1 )
            W2.add( -learn_rate * mom_grad_W2 )
            B2.add( -learn_rate * grad_B2 )
            W3.add( -learn_rate * mom_grad_W3 )
            B3.add( -learn_rate * grad_B3 )

        print epoch, total_loss

    # After we've trained, we return a sugary little function handle
    # that makes things easy.  Basically, what we're doing here is
    # handing the output object (not the loss!) a dictionary where the
    # key is the Kayak input object 'X' (that is the features being
    # used here for logistic regression) and the value in that
    # dictionary is being determined by the argument to the lambda
    # expression.  The point here is that we wind up with a function
    # handle the can be called with a numpy object and it produces the
    # target values for novel data, using the parameters we just learned.
    return lambda x: Y.value(True, inputs={ X: x })

# Loop over our cross validation folds.
for ii, fold in enumerate(CV):
    print "Fold %d" % (ii+1)
    
    # Get the training and validation data, according to this fold.
    train_images, train_labels = fold.train()
    valid_images, valid_labels = fold.valid()

    # Train on these data and get a prediction function back.
    t0 = time.time()
    pred_func = train(train_images, train_labels)
    print "train():", time.time()-t0

    # Make predictions on the validation data.
    valid_preds = np.argmax(pred_func( valid_images ), axis=1)

    # How did we do?
    print np.mean(valid_preds == np.argmax(valid_labels, axis=1))
    
