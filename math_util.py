import math

import numpy as np
from numpy.typing import ArrayLike


def angle_to_vec2(angle) -> np.ndarray:
  c, s = math.cos(angle), math.sin(angle)
  return np.asarray((c, s))


def transform2d(pos: ArrayLike, scale: ArrayLike, angle: ArrayLike) -> tuple:
  scale = np.asarray(scale)
  if len(scale.shape) == 0:
    scale = np.array([scale, scale], dtype="f4")
  c, s = math.cos(angle), math.sin(angle)
  return (
    scale[0] *  c, scale[0] * s, 0.0,
    scale[1] * -s, scale[1] * c, 0.0,
    pos[0],        pos[1],       1.0)


def circle(num_vertices) -> np.ndarray:
  data = np.zeros((num_vertices, 2), dtype="f4")
  for i in range(num_vertices):
    angle = 2 * math.pi * i / num_vertices
    data[i, 0] = math.cos(angle)
    data[i, 1] = math.sin(angle)
  return data
