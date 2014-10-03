import numpy        as np
import numpy.random as npr

import matplotlib.pyplot as plt

import sys
sys.path.append('..')

import kayak

N = 10000
D = 5
P = 1
learn = 0.00001
batch_size = 500

# Random inputs.
X = npr.randn(N,D)
true_W = npr.randn(D,P)
lam = np.exp(np.dot(X, true_W))
Y = npr.poisson(lam)

kyk_batcher = kayak.Batcher(batch_size, N)

# Build network.
kyk_inputs = kayak.Inputs(X, kyk_batcher)

# Labels.
kyk_targets = kayak.Targets(Y, kyk_batcher)

# Weights.
W = 0.01*npr.randn(D,P)
kyk_W = kayak.Parameter(W)

# Linear layer.
kyk_activation = kayak.MatMult( kyk_inputs, kyk_W)

# Exponential inverse-link function.
kyk_lam = kayak.ElemExp(kyk_activation)

# Poisson negative log likelihood.
kyk_nll = kyk_lam - kayak.ElemLog(kyk_lam) * kyk_targets

# Sum the losses.
kyk_loss = kayak.MatSum( kyk_nll )

for ii in xrange(100):

    for batch in kyk_batcher:
        loss = kyk_loss.value
        print loss, np.sum((kyk_W.value - true_W)**2)
        grad = kyk_loss.grad(kyk_W)
        kyk_W.value -= learn * grad

# Plot the true and inferred rate for a subset of data.
T_slice = slice(0,100)
kyk_inputs.value = X[T_slice,:]
plt.figure()
plt.plot(lam[T_slice], 'k')
plt.plot(kyk_lam.value, '--r')
plt.show()