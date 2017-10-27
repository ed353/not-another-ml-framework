'''
Super-class for all implemented layers
'''

class Layer:
  
  def __init__(self):
    pass

  def forward(self, x):
    h = x
    return h

  def backward(self, dLdh):
    dLdW = dLdh
    dLdx = dLdh
    return (dLdW, dLdx)
