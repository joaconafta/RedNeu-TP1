from main import main
import pandas as pd
import numpy as np

def apply_target(data_target):
  return np.where(data_target=='M', -1, 1)

S = [10,9,1]

main(
  (1,11),
  (0,1),
  apply_target,
  S
)

# import main
# elegir parametros del modelo
# ejecutar perceptron
