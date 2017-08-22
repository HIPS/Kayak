Don't use this: use [Autograd](http://github.com/hips/autograd) instead!
=======================================

Kayak: Library for Deep Neural Networks
=======================================

This is a library that implements some useful modules and provides
automatic differentiation utilities for learning deep neural networks.
It is similar in spirit to tools like
[Theano](http://deeplearning.net/software/theano/) and
[Torch](http://torch.ch/).  The objective of Kayak is to be simple to
use and extend, for rapid prototyping in Python.  It is unlikely to be
faster than these other tools, although it is competitive and
sometimes faster in performance when the architectures are highly
complex.  It will certainly not be faster on convolutional
architectures for visual object detection and recognition tasks than,
e.g., [Alex Krizhevsky's CUDA
Convnet](https://code.google.com/p/cuda-convnet2/) or
[Caffe](http://caffe.berkeleyvision.org/).  The point of Kayak is to
be able to experiment in Python with patterns that look a lot like
what you're already used to with Numpy.  It makes it easy to manage
batches of data and compute gradients with backpropagation.

There are some examples in the 'examples' directory, but the main idea
looks like this:

    import kayak
    import numpy.random as npr

    X = ... your feature matrix ...
    Y = ... your label matrix ...

    # Create Kayak objects for features and labels.
    inputs  = kayak.Inputs(X)
    targets = kayak.Targets(Y)

    # Create Kayak objects first-layer weights and biases.  Initialize
    # them with random Numpy matrices.
    weights_1 = kayak.Parameter(npr.randn( input_dims, hidsize_1 ))
    biases_1  = kayak.Parameter(npr.randn( 1, hidsize_1 ))

    # Create Kayak objects that implement a network layer.  First,
    # multiply the features by weights and add biases.
    hiddens_1a = kayak.ElemAdd(kayak.MatMult( inputs, weights_1 ), biases_1)

    # Then, apply a "relu" (rectified linear) nonlinearity.
    # Alternatively, you can apply your own favorite nonlinearity, or
    # add one for an idea that you want to try out.
    hiddens_1b = kayak.HardReLU(hiddens_1a)

    # Now, apply a "dropout" layer to prevent co-adaptation.  Got a
    # new idea for dropout?  It's super easy to extend Kayak with it.
    hiddens_1 = kayak.Dropout(hiddens_1b, drop_prob=0.5)

    # Okay, with that layer constructed, let's make another one the
    # same way: linear transformation + bias with ReLU and dropout.
    # First, create the second-layer parameters.
    weights_2 = kayak.Parameter(npr.randn(hidsize_1, hidsize_2))
    biases_2  = kayak.Parameter(npr.randn(1, hidsize_2))

    # This time, let's compose all the steps, just to show we can.
    hiddens_2 = kayak.Dropout( kayak.HardReLU( kayak.ElemAdd( \
                    kayak.MatMult( hiddens_1, weights_2), biases_2)), drop_prob=0.5)

    # Make the output layer linear.
    weights_out = kayak.Parameter(npr.randn(hidsize_2, 1))
    biases_out  = kayak.Parameter(npr.randn())
    out         = kayak.ElemAdd( kayak.MatMult( hiddens_2, weights_out), biases_out)

    # Apply a loss function.  In this case, we'll just do squared loss.
    loss = kayak.MatSum( kayak.L2Loss( out, targets ))

    # Maybe roll in an L1 norm for the first layer and an L2 norm for the others?
    objective = kayak.ElemAdd(loss,
                              kayak.L1Norm(weights_1, weight=100.0),
                              kayak.L2Norm(weights_2, weight=50.0),
                              kayak.L2Norm(weights_out, weight=3.0))

    # This is the fun part and is the whole point of Kayak.  You can
    # now get the gradient of anything in terms of anything else.
    # Probably, if you're doing neural networks, you want the gradient
    # of the parameters in terms of the overall objective. That way
    # you can go off and do some kind of optimization.
    weights_1_grad   = objective.grad(weights_1)
    biases_1_grad    = objective.grad(biases_1)
    weights_2_grad   = objective.grad(weights_2)
    biases_2_grad    = objective.grad(biases_2)
    weights_out_grad = objective.grad(weights_out)
    biases_out-grad  = objective.grad(biases_out)

    ... use the gradients for learning ...
    ... probably this whole thing would be in a loop ...
    ... in practice you'd probably also use minibatches ...

This is a work in progress and we welcome contributions. Some
nosetests are implemented.  We're working on documentation.  Whatever
docs come into existence will end up at
[http://hips.gihub.io/Kayak](http://hips.gihub.io/Kayak).

This project is primarily develped by the [Harvard Intelligent
Probabilistic Systems (HIPS)](http://hips.seas.harvard.edu) group in
the [Harvard School of Engineering and Applied Sciences
(SEAS)](http://www.seas.harvard.edu).  The primary developers to date
have been Ryan Adams, David Duvenaud, Scott Linderman, Dougal
Maclaurin, and Jasper Snoek.

Kayak is Copyrighted by The President and Fellows of Harvard
University, and is distributed under an MIT license, which can be
found in the license.txt file but is also below:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
