import sys
import data
import numpy        as np
import numpy.random as npr

sys.path.append('..')

import kayak

batch_size = 256
learn_rate = 0.01
momentum   = 0.9

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
    X    = kayak.Inputs(inputs, batcher)
    T    = kayak.Targets(targets, batcher)

    # Weights and biases, with random initializations.
    W    = kayak.Parameter( 0.1*npr.randn( inputs.shape[1], 10 ))
    B    = kayak.Parameter( 0.1*npr.randn(1,10) )

    # Nothing fancy here: inputs times weights, plus bias, then softmax.
    Y    = kayak.LogSoftMax( kayak.ElemAdd( kayak.MatMult(X, W), B ) )

    # The training loss is negative multinomial log likelihood.
    loss = kayak.MatSum(kayak.LogMultinomialLoss(Y, T))

    # Use momentum for the gradient-based optimization.
    mom_grad_W = np.zeros(W.shape())

    # Loop over epochs.
    for epoch in xrange(25):

        # Track the total loss and the overall gradient.
        total_loss   = 0.0
        total_grad_W = np.zeros(W.shape())

        # Loop over batches -- using batcher as iterator.
        for batch in batcher:

            # Compute the loss of this minibatch by asking the Kayak
            # object for its value and giving it reset=True.
            total_loss += loss.value(True)

            # Now ask the loss for its gradient in terms of the
            # weights and the biases -- the two things we're trying to
            # learn here.
            grad_W = loss.grad(W)
            grad_B = loss.grad(B)
            
            # Use momentum on the weight gradient.
            mom_grad_W = momentum*mom_grad_W + (1.0-momentum)*grad_W

            # Now make the actual parameter updates.
            W.add( -learn_rate * mom_grad_W )
            B.add( -learn_rate * grad_B )

            # Keep track of the gradient to see if we're converging.
            total_grad_W += grad_W

        print epoch, total_loss, np.sum(total_grad_W**2)

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
    pred_func = train(train_images, train_labels)

    # Make predictions on the validation data.
    valid_preds = np.argmax(func( valid_images ), axis=1)

    # How did we do?
    print np.mean(valid_preds == np.argmax(valid_labels, axis=1))
    
