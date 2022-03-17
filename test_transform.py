import math

import numpy as np

from transform import TransformedCanvas


def test_translate():
  t = TransformedCanvas(canvas=None)
  t.do_translate([3, 5])
  np.testing.assert_equal(t.transform_pos([1, 1]), [1 + 3, 5 + 1])


def test_rotate():
  t = TransformedCanvas(canvas=None)
  with t.local_transform():
    np.testing.assert_almost_equal(t.transform_pos([3, 5]), [3, 5])
    t.do_rotate(math.pi / 2)
    np.testing.assert_almost_equal(t.transform_pos([3, 5]), [-5, 3])
    t.do_rotate(math.pi / 2)
    np.testing.assert_almost_equal(t.transform_pos([3, 5]), [-3, -5])
    t.do_rotate(math.pi / 2)
    np.testing.assert_almost_equal(t.transform_pos([3, 5]), [5, -3])
    t.do_rotate(math.pi / 2)
    np.testing.assert_almost_equal(t.transform_pos([3, 5]), [3, 5])
    np.testing.assert_almost_equal(t.current_transform, np.identity(3))
