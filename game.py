import math
from abc import abstractmethod, ABC
import random
from typing import Set, List

import moderngl as gl
import moderngl_window as mglw
import numpy as np

import math_util


class GameObject(ABC):
  @abstractmethod
  def render(self, time: float, frame_time: float):
    pass

  @abstractmethod
  def update(self, time: float, frame_time: float):
    pass


class Game(mglw.WindowConfig, GameObject):
  # for OpenGL
  gl_version = (3, 3)
  samples = 4

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.world_transform = np.identity(3)
    self.scale = 20
    self.ctx.multisample = True

    self.space_wrap_size = 1

    self.ship = Ship(self, np.asarray([5, 5], dtype="float32"), 1)
    self.bullets = Bullets(self)

    self.keys_pressed: Set[str] = set()  # TODO

  @property
  def space_dims(self) -> np.ndarray:
    window_size = np.asarray(self.wnd.viewport_size)
    return window_size / self.scale

  def render(self, time: float, frame_time: float):
    window_size = np.asarray(self.wnd.viewport_size)
    self.world_transform = math_util.transform2d([-1, -1], 2 * self.scale / window_size, 0)

    self.ctx.clear(0.0, 0.0, 0.0, 0.0)
    self.ship.render(time=time, frame_time=frame_time)
    self.bullets.render(time=time, frame_time=frame_time)

    # tie updates + rendering for now
    self.update(time=time, frame_time=frame_time)

  def update(self, time: float, frame_time: float):
    self.ship.update(time=time, frame_time=frame_time)
    self.bullets.update(time=time, frame_time=frame_time)

  def key_event(self, key, action, modifiers):
    keys = self.wnd.keys
    if action == keys.ACTION_PRESS:
      self.keys_pressed.add(key)
    elif action == keys.ACTION_RELEASE and key in self.keys_pressed:
      self.keys_pressed.remove(key)

  def key_up(self, event):
    if event.keysym not in self.keys_pressed:
      return
    self.keys_pressed.remove(event.keysym)

  def spawn_bullet(self, pos: np.ndarray, velo: np.ndarray, scale: float):
    self.bullets.spawn(pos=pos, velo=velo, scale=scale)

  def wrap_pos(self, pos: np.ndarray) -> np.ndarray:
    return (pos + self.space_wrap_size) % (self.space_dims + 2 * self.space_wrap_size) - self.space_wrap_size

  def dist(self, from_pos: np.ndarray, to_pos: np.ndarray):
    # calculates to_pos - from_pos but with screen wrapping.
    return to_pos - from_pos
    # TODO re-add the wrapping, by porting this to numpy.
    # old wrapping code:
    # size = self.space_dims + 1 * self.space_wrap_size
    # offsets = [-1, 0, 1]
    # x_distances = [np.abs(to_pos[0] - from_pos[0] + offset * size[0]) for offset in offsets]
    # x_offset = offsets[np.argmin(x_distances)]
    # y_distances = [np.abs(to_pos[1] - from_pos[1] + offset * size[1]) for offset in offsets]
    # y_offset = offsets[np.argmin(y_distances)]
    # return to_pos - from_pos + size * np.asarray([x_offset, y_offset])


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

    with open("ship_vertex.glsl") as vertex_shader_file, open("fragment.glsl") as fragment_shader_file:
      self.prog = self.game.ctx.program(
        vertex_shader=vertex_shader_file.read(),
        fragment_shader=fragment_shader_file.read())
    vertices = np.array([
      1.0, 0.0,
      -0.4, 0.3,
      0.0, 0.0,
      -0.4, -0.3], dtype="f4")
    self.vbo = self.game.ctx.buffer(vertices)  # noqa
    self.vao = self.game.ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_vert')])

  def render(self, time: float, frame_time: float):
    self.prog["world_transform"].value = self.game.world_transform
    self.prog["object_transform"].value = math_util.transform2d(pos=self.pos, scale=self.scale, angle=self.angle)
    self.vao.render(gl.LINE_LOOP)

  def update(self, time: float, frame_time: float):
    keys = self.game.wnd.keys
    if keys.A in self.game.keys_pressed:
      angle_velo = 1
    elif keys.D in self.game.keys_pressed:
      angle_velo = -1
    else:
      angle_velo = 0
    if keys.W in self.game.keys_pressed:
      self.velo = 10 * math_util.angle_to_vec2(self.angle)
    if keys.S in self.game.keys_pressed:
      self.velo *= 0.9

    self.velo *= self.friction ** frame_time
    self.pos += frame_time * self.velo
    self.angle += frame_time * self.angle_speed * angle_velo

    # shoot
    if keys.SPACE in self.game.keys_pressed:
      if self.shoot_cooldown <= 0:
        dir = math_util.angle_to_vec2(self.angle + random.gauss(0, 0.05 * math.pi))
        self.game.spawn_bullet(
          pos=self.pos + self.scale * dir,
          velo=self.velo + dir + np.ones((2,)) * random.gauss(0.1, 0.05),
          scale=random.gauss(0.2, 0.1))
        self.shoot_cooldown += random.gauss(0.02, 0.01)
      else:
        self.shoot_cooldown -= frame_time

    self.pos = self.game.wrap_pos(self.pos)

  def key_down(self, event):
    pass

  def key_up(self, event):
    pass


BulletId = int


class Bullets(GameObject):
  def __init__(self, game: Game):
    self.game = game
    self.pos = np.zeros((0, 2), dtype="f4")
    self.velo = np.zeros((0, 2), dtype="f4")
    self.scale = np.zeros((0,), dtype="f4")
    self.lifetime = np.zeros((0,), dtype="f4")

    with open("bullet_vertex.glsl") as vertex_shader_file, open("fragment.glsl") as fragment_shader_file:
      self.prog = self.game.ctx.program(
        vertex_shader=vertex_shader_file.read(),
        fragment_shader=fragment_shader_file.read())
    vertices = math_util.circle(20)
    self.vbo = self.game.ctx.buffer(vertices)  # noqa
    self.instance_buffer_size = 1000
    self.instance_buffer = self.game.ctx.buffer(reserve=4 * 3 * self.instance_buffer_size, dynamic=True)
    self.vao = self.game.ctx.vertex_array(self.prog, [
      (self.vbo, '2f /v', 'in_vert'),
      (self.instance_buffer, '2f 1f /i', 'object_pos', 'scale')
    ])

  def spawn(self, pos, velo, scale, lifetime: float = 2000) -> BulletId:
    self.pos = np.append(self.pos, np.expand_dims(pos, axis=0), axis=0)
    self.velo = np.append(self.velo, np.expand_dims(velo, axis=0), axis=0)
    self.scale = np.append(self.scale, np.expand_dims(scale, axis=0), axis=0)
    self.lifetime = np.append(self.lifetime, np.expand_dims(lifetime, axis=0), axis=0)
    return self.pos.shape[0] - 1

  def destroy(self, bullet_id: BulletId):
    self.pos = self.pos[bullet_id-1:bullet_id+1]
    self.velo = self.velo[bullet_id-1:bullet_id+1]
    self.scale = self.scale[bullet_id-1:bullet_id+1]
    self.lifetime = self.lifetime[bullet_id-1:bullet_id+1]

  def render(self, time: float, frame_time: float):
    self.prog["world_transform"].value = self.game.world_transform
    num_instances = self.pos.shape[0]
    all_data = np.concatenate([self.pos, np.expand_dims(self.scale, axis=1)], axis=1).astype("f4")
    for buffer_begin in range(0, num_instances, self.instance_buffer_size):
      self.instance_buffer.write(all_data[buffer_begin:buffer_begin+self.instance_buffer_size])
      self.vao.render(gl.LINE_LOOP, instances=self.pos.shape[0])

  def update(self, time: float, frame_time: float):
    self.pos += frame_time * self.velo
    self.pos = self.game.wrap_pos(self.pos)

    mid = self.game.ship.pos
    dist = self.game.dist(self.pos, mid)
    self.velo += dist * frame_time * 100 / (1e-2 + np.linalg.norm(dist))
    self.velo *= 0.995

    self.lifetime -= frame_time

    alive_list = self.lifetime > 0
    if not np.all(alive_list):
      self.pos = self.pos[alive_list, :]
      self.velo = self.velo[alive_list, :]
      self.scale = self.scale[alive_list]
      self.lifetime = self.lifetime[alive_list]

    self.game.wnd.title = f"Bullet count {self.pos.shape[0]}, FPS {int(1 / frame_time)}"


if __name__ == "__main__":
  Game.run()
