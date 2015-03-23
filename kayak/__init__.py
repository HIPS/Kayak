# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import sys
import hashlib
import numpy as np

EPSILON = sys.float_info.epsilon

from differentiable import Differentiable
from root_nodes     import Constant, Parameter, DataNode, Inputs, Targets
from batcher        import Batcher
from matrix_ops     import MatAdd, MatMult, MatElemMult, MatSum, MatMean, Transpose, Reshape, Concatenate, Identity, TensorMult, ListToArray, MatDet
from elem_ops       import ElemAdd, ElemMult, ElemExp, ElemLog, ElemPower, ElemAbs
from nonlinearities import SoftReLU, HardReLU, LogSoftMax, TanH, Logistic, InputSoftMax, SoftMax
from losses         import L2Loss, LogMultinomialLoss
from dropout        import Dropout
from regularizers   import L2Norm, L1Norm, Horseshoe, NExp
from crossval       import CrossValidator
from convolution    import Convolve1d, Pool, TopKPool
from indexing       import Take
from stacking       import Hstack
from generic_ops    import Blank

