import numpy        as np
import numpy.random as npr

import kayak

def load_train():
    N = 100
    D = 5
    return npr.randn(N,D), npr.rand(N,1)

# Load training data.
X, Y = load_train()
N, D = X.shape

# Batch size
batch_size = 11

# Size of hidden layers.
H1 = 10

kyk_batcher = kayak.Batcher(batch_size, N)

# Build network.
kyk_inputs = kayak.Inputs(X, kyk_batcher)

# First-layer weights.
W1 = npr.randn(D,H1)
kyk_l1_wts = kayak.Constant(W1)

# Linear layer.
kyk_layer1_lin = kayak.MatMult( kyk_inputs, kyk_l1_wts )

# Add bias.
# Explicit indices for broadcasting.
B1 = npr.randn(1,H1)
kyk_l1_bias = kayak.Parameter(B1)
kyk_l1_a = kayak.MatAdd( kyk_layer1_lin, kyk_l1_bias )

# Apply relu.
kyk_l1_b = kayak.ReLU( kyk_l1_a )

# Apply dropout.
#rng = npr.RandomState()
#rate = 0.5
#kyk_l1_c = kayak.Dropout( kyk_l1_b, rng, rate )

# Output weights.
W2 = npr.randn(H1,1)
kyk_out_wts = kayak.Parameter(W2)

# Linear output layer.
#kyk_out_a = kayak.MatMult( kyk_l1_c, kyk_out_wts )
kyk_out_a = kayak.MatMult( kyk_l1_b, kyk_out_wts)

# Output bias.
B2 = npr.randn(1,1)
kyk_out_bias = kayak.Parameter(B2)
kyk_out_b = kayak.MatAdd( kyk_out_a, kyk_out_bias )

# Labels.
kyk_targets = kayak.Targets(Y, kyk_batcher)

# Elementwise Loss.
kyk_el_loss = kayak.L2Loss(kyk_out_b, kyk_targets)

# Sum the losses.
#kyk_loss = kayak.MatSum( kyk_el_loss )

#while kyk.batcher.next():
#    
#    # Get an evaluation object.
#    kyk_eval = kyk_loss.eval()

    # Now get gradients in terms of all these things.
#    kyk_l1_wts_grad = kyk_loss.grad( kyk_l1_wts )

    # Now get the actual values.
#    l1_wts_grad = kyk_l1_wts_grad.value()

    # Now update these guys.
#    kyk_l1_wts.value() -= rate * l1_wts_grad

    # Do this for everyone...



