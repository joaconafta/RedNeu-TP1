import sys
from main import main
from util import apply_target

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
