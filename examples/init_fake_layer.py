import numpy as np

import sys
sys.path.append('..')
from namlf.layer.fake import FakeLayer

BATCH_SIZE = 16
INPUT_SIZE = 20
OUTPUT_SIZE = 10

if __name__=='__main__':

  x = np.random.random(size=(BATCH_SIZE, INPUT_SIZE))
  lyr = FakeLayer()
