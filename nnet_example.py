import numpy        as np
import numpy.random as npr

import kayak
import kayak.util

N  = 100
D  = 5
H1 = 7
P  = 3

# Random data.
X = npr.randn(N, D)
Y = npr.randn(N, P)

# Build network.
kyk_inputs = kayak.Inputs(X)

# Labels.
kyk_targets = kayak.Targets(Y)

# First layer weights and biases.
kyk_W1 = kayak.Parameter( npr.randn(D, H1) )
kyk_B1 = kayak.Parameter( npr.randn(1,H1) )

# First layer weight mult plus biases, then nonlinearity.
kyk_H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult( kyk_inputs, kyk_W1 ), kyk_B1)), drop_prob=0.5)
#kyk_H1 = kayak.ElemAdd(kayak.MatMult( kyk_inputs, kyk_W1 ), kyk_B1)

# Second layer weights and bias.
kyk_W2 = kayak.Parameter( npr.randn(H1, P) )
kyk_B2 = kayak.Parameter( npr.randn(1,P) )

# Second layer multiplication.
kyk_out = kayak.ElemAdd(kayak.MatMult( kyk_H1, kyk_W2 ), kyk_B2)

# Elementwise Loss.
kyk_el_loss = kayak.L2Loss(kyk_out, kyk_targets)

# Sum the losses.
kyk_loss = kayak.MatSum( kyk_el_loss )

print "W2:", kayak.util.checkgrad(kyk_W2, kyk_loss)
print "B2:", kayak.util.checkgrad(kyk_B2, kyk_loss)
print "W1:", kayak.util.checkgrad(kyk_W1, kyk_loss)
print "B1:", kayak.util.checkgrad(kyk_B1, kyk_loss)

#UU = kayak.Parameter(npr.randn(1,1))
#WW = kayak.Parameter(npr.randn(7,5))
#XX = kayak.Parameter(npr.randn(5,1))
#YY = kayak.SoftReLU(kayak.MatAdd(kayak.MatMult(WW, XX), UU))
#ZZ = kayak.MatSum(YY)
#print kayak.util.checkgrad(XX, ZZ)


#for ii in xrange(100):
#
#    for batch in kyk_batcher:

#        loss = kyk_loss.value(True)
#        print loss, np.sum((kyk_W.value() - true_W)**2)
#        grad = kyk_loss.grad(kyk_W)
#        kyk_W.add( -learn * grad )
