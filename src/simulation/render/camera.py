"""Perspective camera for the simulated-lab software renderer.

Conventions (kept deliberately explicit, because this project also carries a
SECOND, different convention that must not be confused with this one):

  * LAB WORLD frame - right-handed, +X right, +Y up, +Z toward the viewer
    (out of the backdrop). Metres. This is the frame every mesh in
    ``src/simulation/render/meshes.py`` is authored in.
  * VIEW frame - OpenGL style, camera at the origin looking down -Z, +Y up.
    Positive *depth* is therefore ``-z_view``.
  * SCREEN - pixels, origin top-left, +Y down (OpenCV's convention, since the
    rendered buffer is handed straight to cv2).

The project's PERCEPTION code uses an entirely separate camera frame (+Y down,
+Z forward-away, see ``BoundingBox3D.project_to_2d``). Converting between the
two is the sole job of ``perception_to_lab`` in ``src/simulation/lab_sim.py`` -
nothing else should mix them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


@dataclass
class Camera:
    """Pinhole perspective camera with a look-at rig."""

    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_deg: float
    width: int
    height: int
    near: float = 0.05
    far: float = 60.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32)
        self.target = np.asarray(self.target, dtype=np.float32)
        self.up = np.asarray(self.up, dtype=np.float32)
        self._rebuild()

    def _rebuild(self) -> None:
        forward = _normalize(self.target - self.position)      # camera looks along +forward
        right = _normalize(np.cross(forward, self.up))
        true_up = np.cross(right, forward)

        # Rows are the view-space basis vectors expressed in world coordinates.
        # View looks down -Z, so the third row is -forward.
        self._R = np.stack([right, true_up, -forward]).astype(np.float32)
        self._t = (-self._R @ self.position).astype(np.float32)

        f = 1.0 / math.tan(math.radians(self.fov_y_deg) * 0.5)
        self.fy = 0.5 * self.height * f
        self.fx = self.fy                                       # square pixels
        self.cx = 0.5 * self.width
        self.cy = 0.5 * self.height

    def moved(self, position=None, target=None, fov_y_deg=None) -> "Camera":
        """A copy with one or more rig parameters replaced."""
        return Camera(
            position=np.asarray(position if position is not None else self.position, dtype=np.float32),
            target=np.asarray(target if target is not None else self.target, dtype=np.float32),
            up=self.up.copy(),
            fov_y_deg=float(fov_y_deg if fov_y_deg is not None else self.fov_y_deg),
            width=self.width, height=self.height, near=self.near, far=self.far,
        )

    def to_view(self, points_world: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 3) view space."""
        return points_world.astype(np.float32) @ self._R.T + self._t

    def directions_to_view(self, dirs_world: np.ndarray) -> np.ndarray:
        """Rotate directions (normals, light vectors) into view space - no translation.

        The view rotation is orthonormal, so normals transform with R itself
        rather than its inverse-transpose.
        """
        return np.atleast_2d(dirs_world).astype(np.float32) @ self._R.T

    def project_view(self, points_view: np.ndarray) -> tuple:
        """(N, 3) view space -> ((N, 2) screen pixels, (N,) positive depth).

        Depth is clamped away from zero so callers can divide freely; vertices
        behind the near plane must already have been clipped by the caller
        (``clip_triangle_near`` in raster.py) for the result to be meaningful.
        """
        depth = np.maximum(-points_view[:, 2], 1e-6)
        screen = np.empty((len(points_view), 2), dtype=np.float32)
        screen[:, 0] = self.fx * points_view[:, 0] / depth + self.cx
        screen[:, 1] = self.cy - self.fy * points_view[:, 1] / depth
        return screen, depth

    def project(self, points_world: np.ndarray) -> tuple:
        """(N, 3) world -> ((N, 2) screen pixels, (N,) positive depth)."""
        return self.project_view(self.to_view(points_world))
