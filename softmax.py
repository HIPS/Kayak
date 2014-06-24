import numpy        as np
import numpy.random as npr

import kayak

T = kayak.Targets(npr.rand(10,5) < 0.5)
X = kayak.Parameter(npr.randn(10,5))
Y = kayak.LogMultinomialLoss( kayak.LogSoftMax(X), T)
O = kayak.MatSum(Y)

print O.value()
print kayak.util.checkgrad(X, O)
