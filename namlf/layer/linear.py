'''
Linear layer
'''

import numpy as np
from namlf.layer.base import Layer

class LinearLayer(Layer):

  def __init__(self, input_size, output_size):
    self._input_size = input_size
    self._output_size = output_size
    
    # initialize weights and bias
    # TODO: create function w/ stub in base layer def'n
    self._params = {}
    self._params['W'] = np.random.normal(loc=0.0, 
                  scale=1/np.sqrt(input_size),
                  size=(input_size, output_size))
    self._params['b'] = np.zeros(output_size)

    # initialize derivatives
    # TODO: do in base layer def'n
    self._derivs = {}
    self._derivs['W'] = np.zeros(self._params['W'].shape)
    self._derivs['b'] = np.zeros(self._params['b'].shape)

  def forward(self, x):
    
    self._x = x # save for backward gradient
    
    W = self._params['W']
    b = self._params['b']
    
    h = np.dot(x, W) + b
    return h

  def backward(self, dLdh):
    self._dLdW = np.tensordot(self._x.T, dLdh, axes=1)
    self._dLdb = np.ones(self._params['b'].shape)
