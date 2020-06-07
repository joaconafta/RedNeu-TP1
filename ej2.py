import sys
from main import main
from dataset import normalize

def targetNormalize(target_data):
  data = target_data.copy()
  normalize(data)
  return data.to_numpy()

main(
  sys.argv,
  (0, 8),
  (8, 10),
  targetNormalize,
  [8, 7, 7, 4, 2],
  1000,
  0.01
)
