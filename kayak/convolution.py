# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np

import util

from . import Differentiable
import sys

class Convolve1d(Differentiable):

    def __init__(self, A, B, ncolors=1):
        super(Convolve1d, self).__init__([A,B])
        self.A    = A
        self.B    = B
        self.ncolors = ncolors

    def _compute_value(self):
        A = self.A.value
        B = self.B.value
        filtersize = B.shape[0]/self.ncolors

        # Broadcast to get color channels
        A = np.reshape(A, (A.shape[0], self.ncolors, -1))
        D = A.shape[-1] - filtersize + 1
        output = np.zeros((A.shape[0], D, B.shape[1]))

        offset = 0
        for j in xrange(D):
            output[:,j,:] = np.dot(A[:, :,offset:offset+filtersize].reshape((A.shape[0],-1)), B)
            offset += 1

        return output.reshape((A.shape[0], D*B.shape[1]))

    def _local_grad(self, parent, d_out_d_self):
        filtersize = self.B.shape[0]/self.ncolors
        if parent == 0:
            output     = np.zeros((self.A.shape[0], self.A.shape[1]))
            B          = self.B.value
            output = output.reshape((output.shape[0], self.ncolors, -1))
            outgrad = d_out_d_self.reshape(d_out_d_self.shape[0], -1, B.shape[-1])
            for i in xrange(outgrad.shape[1]):
                output[:,:,i:i+filtersize] += np.dot(outgrad[:,i,:], B.T).reshape((output.shape[0], self.ncolors, filtersize)) 

            return output.reshape((output.shape[0], -1))

        elif parent == 1:
            output     = np.zeros((self.B.shape[0], self.B.shape[1]))
            A          = self.A.value
            A          = np.reshape(A, (A.shape[0], self.ncolors, -1))
            filtersize = self.B.shape[0]/self.ncolors
            outgrad    = np.reshape(d_out_d_self, (d_out_d_self.shape[0], -1, self.B.shape[1]))
            offset = 0
            for j in xrange(outgrad.shape[1]):
                output += np.dot(A[:,:,offset:offset+filtersize].reshape((A.shape[0],-1)).T, outgrad[:,j,:])
                offset += 1
            return output
        else:   
            raise Exception("Not a parent of me")            


