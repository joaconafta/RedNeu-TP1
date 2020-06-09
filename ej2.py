import sys
from main import main
from util import targetNormalize

main(
  sys.argv,
  (0, 8),
  (8, 10),
  targetNormalize,
  [8, 8, 2],
  5000,
  0.01,
  5
)