import sys
from dataset import load_dataset

def main(input_break, target_break, apply_target):
  if (len(sys.argv) != 3):
    print('Error en la entrada')

  [_, model, data] = sys.argv

  (input, target) = load_dataset(
    data,
    input_break,
    target_break,
    apply_target
  )
