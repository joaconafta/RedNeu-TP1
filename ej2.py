import sys
from main import main
from util import targetNormalize

main(
  sys.argv,
  (0, 8),
  (8, 10),
  targetNormalize,
  [8, 7, 7, 4, 2],
  1000,
  0.01
)
