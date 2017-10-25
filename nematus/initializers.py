'''
Parameter initializers
'''

import numpy

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano_util import floatX

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype(floatX)


def glorot_normal(nin, nout=None, ksize=3, gain=1.0):
    if gain == "relu":
        gain = numpy.sqrt(2)
    if nout is None:
        nout = nin
    std = gain * numpy.sqrt(2.0 / ((nin + nout) * ksize))
    W = std * numpy.random.rand(nout, nin, ksize, 1)
    return W.astype(floatX)

