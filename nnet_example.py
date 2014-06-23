import time
import numpy        as np
import numpy.random as npr

import kayak
import kayak.util

N  = 1000
D  = 50
H1 = 10
P  = 1
batch_size = 256

# Random data.
X = npr.randn(N, D)
Y = npr.randn(N, P)

batcher = kayak.Batcher(batch_size, N)

# Build network.
kyk_inputs = kayak.Inputs(X, batcher)

# Labels.
kyk_targets = kayak.Targets(Y, batcher)

# First layer weights and biases.
kyk_W1 = kayak.Parameter( npr.randn(D, H1) )
kyk_B1 = kayak.Parameter( npr.randn(1,H1) )

# First layer weight mult plus biases, then nonlinearity.
kyk_H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult( kyk_inputs, kyk_W1 ), kyk_B1)), drop_prob=0.5)

# Second layer weights and bias.
kyk_W2 = kayak.Parameter( npr.randn(H1, P) )
kyk_B2 = kayak.Parameter( npr.randn(1,P) )

# Second layer multiplication.
kyk_out = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult( kyk_H1, kyk_W2 ), kyk_B2)), drop_prob=0.5)

# Elementwise Loss.
kyk_el_loss = kayak.L2Loss(kyk_out, kyk_targets)

# Sum the losses.
kyk_loss = kayak.MatSum( kyk_el_loss )

# Roll in the weight regularization.
kyk_obj = kayak.ElemAdd( kyk_loss, kayak.L1Norm(kyk_W1, scale=100.0), kayak.L1Norm(kyk_W2, scale=100.0))

print "W2:", kayak.util.checkgrad(kyk_W2, kyk_obj)
print "B2:", kayak.util.checkgrad(kyk_B2, kyk_obj)
print "W1:", kayak.util.checkgrad(kyk_W1, kyk_obj)
print "B1:", kayak.util.checkgrad(kyk_B1, kyk_obj)


t0 = time.time()
for ii in xrange(10):

    for batch in batcher:

        val = kyk_obj.value(True)
        grad_W1 = kyk_obj.grad(kyk_W1)
        grad_B1 = kyk_obj.grad(kyk_B1)
        grad_W2 = kyk_obj.grad(kyk_W2)
        grad_B2 = kyk_obj.grad(kyk_B2)

t1 = time.time()
print t1-t0
