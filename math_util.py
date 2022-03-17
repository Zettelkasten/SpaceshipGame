import math

import numpy as np


def angle_to_vec2(angle) -> np.ndarray:
  c, s = math.cos(angle), math.sin(angle)
  return np.asarray((c, -s))
