import sys
from dataset import load_dataset
from os import path
import numpy as np

def initialize(S):
  L = len(S)

  W = [0]
  for i in range(L-1):
    W.append(
      np.random.normal(
        0, S[0]**(-1/2), (S[i]+1, S[i+1])
      )
    )

  dW = [0]
  for i in range(L-1):
    dW.append(np.zeros_like(W[i+1]))

  return (W, dW)

def train(input, target, S):
  (W, dW) = initialize(S)
  print(W)

def main(input_break, target_break, apply_target, S):
  if (len(sys.argv) != 3):
    print('Error en la entrada')

  [_, model, data] = sys.argv

  (input, target) = load_dataset(
    data,
    input_break,
    target_break,
    apply_target
  )

  if (not path.exists(model)):
    train(input, target, S)
