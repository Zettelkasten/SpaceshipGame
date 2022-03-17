from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List

import numpy as np
import tkinter as tk

from numpy.typing import ArrayLike


class TransformedCanvas:
  def __init__(self, canvas: tk.Canvas|None):
    self.canvas = canvas
    self.stack: List[np.ndarray] = [np.identity(3)]

  @property
  def current_transform(self) -> np.ndarray:
    return self.stack[-1]

  def push(self):
    self.stack.append(self.current_transform.copy())

  def pop(self):
    assert len(self.stack) > 1
    self.stack.pop()

  @contextmanager
  def local_transform(self):
    try:
      yield self.push()
    finally:
      self.pop()

  def transform_pos(self, pos: ArrayLike):
    pos = np.asarray(pos)
    assert pos.shape == (2,)
    return (self.current_transform @ np.asarray((pos[0], pos[1], 1)))[:2]

  def _do_transform(self, matrix: np.ndarray):
    np.matmul(
      self.current_transform,
      matrix,
      out=self.current_transform)

  def do_translate(self, pos: ArrayLike):
    pos = np.asarray(pos)
    assert pos.shape == (2,)
    self._do_transform(np.asarray(((1, 0, pos[0]), (0, 1, pos[1]), (0, 0, 1))))

  def do_scale(self, scale: float | ArrayLike):
    scale = np.asarray(scale)
    if scale.shape == ():
      scale = np.asarray((scale, scale))
    assert scale.shape == (2,)
    self._do_transform(np.asarray(((scale[0], 0, 0), (0, scale[1], 0), (0, 0, 1))))

  def do_rotate(self, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    self._do_transform(np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1))))

  def draw_rect(self, pos: ArrayLike, size: ArrayLike, **kwargs):
    return self.canvas.create_rectangle(*self.transform_pos(pos), *size, **kwargs)

  def draw_oval(self, pos: ArrayLike, size: ArrayLike, **kwargs):
    pos, size = np.asarray(pos), np.asarray(size)
    x1 = self.transform_pos(pos - size / 2)
    x2 = self.transform_pos(pos + size / 2)
    return self.canvas.create_oval(*x1, *x2, **kwargs)

  @contextmanager
  def poly(self, **kwargs):
    poly = _PolyContext(self)
    try:
      yield poly
    finally:
      self.canvas.create_polygon(*np.reshape(poly.points, (-1,)), **kwargs)


class _PolyContext:
  def __init__(self, canvas: TransformedCanvas):
    self.canvas = canvas
    self.points: List[np.ndarray] = []

  def add_point(self, point: ArrayLike):
    self.points.append(self.canvas.transform_pos(point))
