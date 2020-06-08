import sys
from main import main
import pandas as pd
import numpy as np

def apply_target(data_target):
  return np.where(data_target=='M', -1, 1)

main(
  sys.argv,
  (1, 11),
  (0, 1),
  apply_target,
  [10, 10, 10, 1],
  5000,
  0.02,
  5
)
