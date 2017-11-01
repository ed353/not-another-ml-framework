'''
Super-class for all implemented layers
'''

from abc import ABC, abstractmethod

class Layer(ABC):
  
  def __init__(self):
    self._params = {}
    self._derivs = {}
    super(Layer, self).__init__()

  @abstractmethod
  def forward(self, x):
    pass

  @abstractmethod
  def backward(self, dLdh):
    pass

  def weight_update(self, update_func):
    for ky in self._params.keys():
      w = self._params[ky]
      dw = self._derivs[ky]
      self._params[ky] = update_func(w, dw)

'''
Class for layer parameters
'''
# TODO: add definitions for creation? deriv? update?
class Parameter:

  def __init__(self):
    pass

  
