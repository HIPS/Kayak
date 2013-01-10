import kayak

import numpy        as np
import numpy.random as npr

# Load some features, let's say.
# X = kayak.load('feature_file')

# Also load some labels.
# Y = kayak.load('label_file')

# Declare some weights.
# W = kayak.new( put the size here?, then type? )

# Compute the log of the resulting softmax.
# log_probs = kayak.log_softmax(kayak.dot(X, W))

# Compute the overall objective.
# objective = kayak.sum(kayak.el_mult(log_probs, Y))

# Set the weights to something.
# W.set( random things )

# Compute the gradient in terms of the weights.
# gradient = objective.gradient(W)

# Compute the Hessian times a vector.
# hess_vec = kayak.dot(gradient.gradient(W), vec)
# hess_vec = objective.hessian(W, vec)

# Perhaps use the gradient to compute an update.
# W.set( W.eval() + learn_rate * gradient.eval() )

np_X = npr.randn(5,4)
np_Y = npr.randn(4,3)

X = kayak.variables(np_X)
Y = kayak.variables(np_Y)

Z = kayak.mat_mult(X, Y)

# this is all wrong
# it should output, e.g., a 4d tensor
# need to figure out the incoming thing
print np_X
print np_Y
print Z.gradient(X).value().shape
print Z.gradient(Y).value().shape
