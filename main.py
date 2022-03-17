from __future__ import annotations

from tkinter import *
from time import time
import importlib

import numpy as np

import game as game_
from game import Game
from transform import TransformedCanvas

root = Tk()
window_dims = np.asarray([600, 600])
canvas = Canvas(root, width=window_dims[0], height=window_dims[1])
transformed_canvas = TransformedCanvas(canvas)

last_tick_time: float = time()
tick_duration = 1 / 60  # 60 fps

game: Game | None = None


def loop_callback():
  global last_tick_time
  current_time = time()
  if last_tick_time + tick_duration < current_time:
    do_tick(current_time - last_tick_time)
    last_tick_time = current_time

  root.after(1, loop_callback)


def init_game():
  global game
  game = Game(window_dims)


def do_tick(delta):
  importlib.reload(game_)
  canvas.delete("all")
  game.update(delta)
  game.draw(transformed_canvas)


if __name__ == '__main__':
  canvas.pack()
  root.after(1, loop_callback)
  init_game()

  root.bind("<KeyPress>", lambda e: game.key_down(e))
  root.bind("<KeyRelease>", lambda e: game.key_up(e))
  root.mainloop()
