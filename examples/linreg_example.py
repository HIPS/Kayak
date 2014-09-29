import numpy        as np
import numpy.random as npr

import sys
sys.path.append('..')

import kayak

N = 10000
D = 5
P = 3
learn = 0.00001
batch_size = 500

# Random inputs.
X = npr.randn(N,D)
true_W = npr.randn(D,P)
Y = np.dot(X, true_W) + 0.1*npr.randn(N,P)

kyk_batcher = kayak.Batcher(batch_size, N)

# Build network.
kyk_inputs = kayak.Inputs(X, kyk_batcher)

# Labels.
kyk_targets = kayak.Targets(Y, kyk_batcher)

# Weights.
W = 0.01*npr.randn(D,P)
kyk_W = kayak.Parameter(W)

# Linear layer.
kyk_out = kayak.MatMult( kyk_inputs, kyk_W )

# Elementwise Loss.
kyk_el_loss = kayak.L2Loss(kyk_out, kyk_targets)

# Sum the losses.
kyk_loss = kayak.MatSum( kyk_el_loss )

for ii in xrange(100):

    for batch in kyk_batcher:
        loss = kyk_loss.value
        print loss, np.sum((kyk_W.value - true_W)**2)
        grad = kyk_loss.grad(kyk_W)
        kyk_W.value -= learn * grad
