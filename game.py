import math
from abc import abstractmethod, ABC
import random
from typing import Set, List

import numpy as np

import math_util
from transform import TransformedCanvas


class GameObject(ABC):
  @abstractmethod
  def draw(self, canvas: TransformedCanvas):
    pass

  @abstractmethod
  def update(self, delta: float):
    pass


class Game(GameObject):
  def __init__(self, window_dims: np.ndarray):
    self.scale = 40
    self.space_dims = window_dims / self.scale
    self.space_wrap_size = 1

    self.ship = Ship(self, np.asarray([5, 5], dtype="float32"), 1)
    self.bullets: List[Bullet] = []
    self.remove_bullets: List[Bullet] = []
    self.asteroids: List[Asteroid] = []
    self.remove_asteroids: List[Asteroid] = []

    self.keys_pressed: Set[str] = set()

  def draw(self, canvas: TransformedCanvas):
    with canvas.local_transform():
      canvas.do_scale(self.scale)
      self.ship.draw(canvas)
      for bullet in self.bullets:
        bullet.draw(canvas)
      for asteroid in self.asteroids:
        asteroid.draw(canvas)

  def update(self, delta: float):
    self.ship.update(delta)
    for bullet in self.bullets:
      bullet.update(delta)
    for bullet in self.remove_bullets:
      if bullet in self.bullets:
        self.bullets.remove(bullet)
    self.remove_bullets.clear()
    for asteroid in self.asteroids:
      asteroid.update(delta)
    for asteroid in self.remove_asteroids:
      if asteroid in self.asteroids:
        self.asteroids.remove(asteroid)
    self.remove_asteroids.clear()

  def key_down(self, event):
    self.keys_pressed.add(event.keysym)
    self.ship.key_down(event)
    if event.keysym == 'q':
      self.asteroids.append(Asteroid(
        self,
        pos=self.space_dims * np.random.random((2,)),
        velo=np.random.normal(0, 0.5, size=(2,)),
        scale=random.gauss(1, 0.6)))

  def key_up(self, event):
    if event.keysym not in self.keys_pressed:
      return
    self.keys_pressed.remove(event.keysym)

  def configure(self, event):
    # resize
    window_dims = np.asarray((event.width, event.height))
    self.space_dims = window_dims / self.scale

  def spawn_bullet(self, pos: np.ndarray, velo: np.ndarray, scale: float):
    self.bullets.append(Bullet(self, pos.copy(), velo.copy(), scale))

  def wrap_pos(self, pos: np.ndarray) -> np.ndarray:
    return (pos + self.space_wrap_size) % (self.space_dims + 2 * self.space_wrap_size) - self.space_wrap_size

  def dist(self, from_pos: np.ndarray, to_pos: np.ndarray):
    # calculates to_pos - from_pos but with screen wrapping.
    size = self.space_dims + 1 * self.space_wrap_size
    offsets = [-1, 0, 1]
    x_distances = [np.abs(to_pos[0] - from_pos[0] + offset * size[0]) for offset in offsets]
    x_offset = offsets[np.argmin(x_distances)]
    y_distances = [np.abs(to_pos[1] - from_pos[1] + offset * size[1]) for offset in offsets]
    y_offset = offsets[np.argmin(y_distances)]
    return to_pos - from_pos + size * np.asarray([x_offset, y_offset])


class Ship(GameObject):
  def __init__(self, game: Game, pos: np.ndarray, scale: float):
    self.game = game
    self.pos = pos
    self.scale = scale
    self.velo = np.zeros((2,))
    self.angle = 0
    self.angle_speed = math.pi  # turn 180 deg per second
    self.friction = 0.4

    self.shoot_cooldown = 0

  def draw(self, canvas: TransformedCanvas):
    with canvas.local_transform():
      canvas.do_translate(self.pos)
      canvas.do_scale(self.scale)
      canvas.do_rotate(-self.angle)
      with canvas.poly(fill="", outline="black") as poly:
        poly.add_point([1, 0])
        canvas.do_rotate(0.75 * math.pi)
        poly.add_point([1, 0])
        poly.add_point([0, 0])
        canvas.do_rotate(0.5 * math.pi)
        poly.add_point([1, 0])

  def update(self, delta: float):
    self.game.dist(from_pos=np.asarray([0, 0]), to_pos=np.asarray([17,17]))
    if 'a' in self.game.keys_pressed:
      angle_velo = 1
    elif 'd' in self.game.keys_pressed:
      angle_velo = -1
    else:
      angle_velo = 0
    if 'w' in self.game.keys_pressed:
      self.velo = 10 * math_util.angle_to_vec2(self.angle)
    if 's' in self.game.keys_pressed:
      self.velo *= 0.9
    self.velo *= self.friction ** delta
    self.pos += delta * self.velo
    self.angle += delta * self.angle_speed * angle_velo

    # shoot
    if "space" in self.game.keys_pressed:
      if self.shoot_cooldown <= 0:
        dir = math_util.angle_to_vec2(self.angle + random.gauss(0, 0.05 * math.pi))
        self.game.spawn_bullet(
          pos=self.pos + self.scale * dir,
          velo=self.velo + dir + np.ones((2,)) * random.gauss(0.1, 0.05),
          scale=random.gauss(0.2, 0.1))
        self.shoot_cooldown += random.gauss(0.02, 0.01)
      else:
        self.shoot_cooldown -= delta

    self.pos = self.game.wrap_pos(self.pos)

  def key_down(self, event):
    pass

  def key_up(self, event):
    pass


class Bullet(GameObject):
  def __init__(self, game: Game, pos: np.ndarray, velo: np.ndarray, scale: float):
    self.game = game
    self.pos = pos
    self.scale = scale
    self.velo = velo
    self.lifetime = 20#5

  def draw(self, canvas: TransformedCanvas):
    with canvas.local_transform():
      canvas.do_translate(self.pos)
      canvas.do_scale(self.scale)
      canvas.draw_oval([0, 0], [1, 1])

  def update(self, delta: float):
    self.pos += delta * self.velo
    self.pos = self.game.wrap_pos(self.pos)

    mid = self.game.ship.pos
    dist = self.game.dist(self.pos, mid)
    self.velo += dist * delta * 10 / (1e-2 + np.linalg.norm(dist))
    self.velo *= 0.995
    for asteroid in self.game.asteroids:
      if asteroid.collides_with(self):
        self.game.remove_asteroids.append(asteroid)
        self.game.remove_bullets.append(self)

    self.lifetime -= delta
    if self.lifetime <= 0:
      self.game.remove_bullets.append(self)


class Asteroid(GameObject):
  def __init__(self, game: Game, pos: np.ndarray, velo: np.ndarray, scale: float):
    self.game = game
    self.pos = pos
    self.scale = scale
    self.velo = velo
    vertices = 10
    self.angles_list = np.random.normal(1, 0.2, size=vertices)
    self.angles_list /= np.sum(self.angles_list)  # normalize to sum=2pi
    self.angles_list *= 2 * math.pi
    self.radius_list = np.random.normal(1, 0.3, size=vertices)

  def draw(self, canvas: TransformedCanvas):
    with canvas.local_transform():
      canvas.do_translate(self.pos)
      canvas.do_scale(self.scale)
      with canvas.poly(fill="", outline="black") as poly:
        for angle, radius in zip(self.angles_list, self.radius_list):
          poly.add_point([radius, 0])
          canvas.do_rotate(angle)

  def update(self, delta: float):
    self.pos += delta * self.velo
    self.pos = self.game.wrap_pos(self.pos)

  def collides_with(self, bullet: Bullet) -> bool:
    return np.linalg.norm(self.pos - bullet.pos) <= self.scale  # ignore size of bullet, that feels too OP
