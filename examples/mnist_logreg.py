import sys
import data
import numpy        as np
import numpy.random as npr

sys.path.append('..')

import kayak

batch_size = 256
learn_rate = 0.01
momentum   = 0.9

train_images, train_labels, test_images, test_labels = data.mnist()
train_images = np.reshape(train_images,
                          (train_images.shape[0],
                           train_images.shape[1]*train_images.shape[2]))/255.0

train_labels = kayak.util.onehot(train_labels)
test_labels  = kayak.util.onehot(test_labels)

CV = kayak.CrossValidator(10, train_images, train_labels)

def train(inputs, targets):
    batcher = kayak.Batcher(batch_size, inputs.shape[0])

    X    = kayak.Inputs(inputs, batcher)
    T    = kayak.Targets(targets, batcher)
    W    = kayak.Parameter( 0.1*npr.randn( inputs.shape[1], 10 ))
    B    = kayak.Parameter( 0.1*npr.randn(1,10) )
    Y    = kayak.LogSoftMax( kayak.ElemAdd( kayak.MatMult(X, W), B ) )
    loss = kayak.MatSum(kayak.LogMultinomialLoss(Y, T))

    mom_grad_W = np.zeros(W.shape())
    for ii in xrange(5):
        total_loss   = 0.0
        total_grad_W = np.zeros(W.shape())
        for batch in batcher:
            total_loss += loss.value(True)
            grad_W = loss.grad(W)
            grad_B = loss.grad(B)
            
            mom_grad_W = momentum*mom_grad_W + (1.0-momentum)*grad_W

            W.add( -learn_rate * mom_grad_W )
            B.add( -learn_rate * grad_B )

            total_grad_W += grad_W

        print ii, total_loss, np.sum(total_grad_W**2)

    return lambda x: Y.value(True, inputs={ X: x })

for ii, fold in enumerate(CV):
    print "Fold %d" % (ii+1)

    train_images, train_labels = fold.train()
    valid_images, valid_labels = fold.valid()

    func = train(train_images, train_labels)

    #valid_preds = func( valid_images )
