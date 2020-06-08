import pandas as pd
import numpy as np
from dataset import normalize

def apply_target(data_target):
  return np.where(data_target=='M', -1, 1)

def targetNormalize(target_data):
  data = target_data.copy()
  normalize(data)
  return data.to_numpy()
