import numpy as np

import sys
sys.path.append('..')
from namlf.layer.linear import LinearLayer

BATCH_SIZE = 16
INPUT_SIZE = 20
OUTPUT_SIZE = 10

if __name__=='__main__':

  x = np.random.random(size=(BATCH_SIZE, INPUT_SIZE))
  lyr = LinearLayer(INPUT_SIZE, OUTPUT_SIZE)
  y = lyr.forward(x)
  
  dLdy = np.random.random(size=y.shape)
  lyr.backward(dLdy)
