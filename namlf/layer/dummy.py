'''
Dummy layer for testing
'''

from base import Layer

class DummyLayer(Layer):
  
  def __init__(self):
    super(DummyLayer, self).__init__()

  def forward(self, x):
    print('dummy forward pass')

  def backward(self, dLdh):
    print('dummy backward pass')

  def _weight_update(self, update_func):
    super(DummyLayer, self)._weight_update(update_func)
