"""Deferred lighting pass over a rasterized G-buffer.

Runs once per frame over the whole buffer rather than per triangle, so lights,
fog, and tone mapping cost the same whether the scene has 200 triangles or
2000. Colours are BGR in [0, 1] to match OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.simulation.render import raster as R
from src.simulation.render.camera import Camera


@dataclass
class Light:
    """A directional key/fill/rim light, or a point light when ``point`` is set."""

    direction: np.ndarray                     # world-space direction the light TRAVELS
    color: np.ndarray                         # BGR in [0, 1]
    intensity: float = 1.0
    point: Optional[np.ndarray] = None        # world position; overrides `direction`
    radius: float = 4.0                       # falloff scale for point lights

    def __post_init__(self) -> None:
        d = np.asarray(self.direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        self.direction = d / n if n > 1e-9 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self.color = np.asarray(self.color, dtype=np.float32)
        if self.point is not None:
            self.point = np.asarray(self.point, dtype=np.float32)


@dataclass
class Environment:
    """Ambient and atmospheric terms."""

    sky_color: np.ndarray                     # ambient from above (BGR)
    ground_color: np.ndarray                  # bounce from below (BGR)
    fog_color: np.ndarray
    fog_start: float = 2.0
    fog_end: float = 9.0
    fog_density: float = 0.85
    exposure: float = 1.0

    def __post_init__(self) -> None:
        self.sky_color = np.asarray(self.sky_color, dtype=np.float32)
        self.ground_color = np.asarray(self.ground_color, dtype=np.float32)
        self.fog_color = np.asarray(self.fog_color, dtype=np.float32)


def shade_rows(gb_rows: np.ndarray, depth_rows: np.ndarray, camera: Camera,
               lights: List[Light], env: Environment,
               shadow_rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Shade a flat list of G-buffer samples -> (N, 3) linear BGR.

    Row-based rather than image-based so callers can shade only the pixels that
    actually changed this frame; the static lab is baked once and reused, which
    is what keeps the per-frame cost proportional to the moving actors rather
    than to the screen.
    """
    if len(gb_rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    albedo = gb_rows[:, R.GB_ALBEDO]
    normal = gb_rows[:, R.GB_NORMAL]
    world = gb_rows[:, R.GB_WORLD]
    spec_k = gb_rows[:, R.GB_SPEC][:, None]
    shin = gb_rows[:, R.GB_SHIN][:, None]
    emis = gb_rows[:, R.GB_EMIS][:, None]
    cover = gb_rows[:, R.GB_COVER] > 0.5

    N = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    view_vec = camera.position[None, :] - world
    V = view_vec / np.maximum(np.linalg.norm(view_vec, axis=1, keepdims=True), 1e-6)

    # Two-sided shading: flip normals facing away from the camera so single-sided
    # quads (walls, backdrop) never render pitch black.
    flip = np.sum(N * V, axis=1, keepdims=True) < 0.0
    N = np.where(flip, -N, N)

    up_frac = N[:, 1:2] * 0.5 + 0.5
    ambient = env.sky_color * up_frac + env.ground_color * (1.0 - up_frac)

    direct = np.zeros_like(albedo)
    specular = np.zeros_like(albedo)
    for light in lights:
        if light.point is not None:
            to_light = light.point[None, :] - world
            dist = np.linalg.norm(to_light, axis=1, keepdims=True)
            L = to_light / np.maximum(dist, 1e-6)
            atten = 1.0 / (1.0 + (dist / light.radius) ** 2)
        else:
            L = -light.direction[None, :]
            atten = 1.0

        ndotl = np.clip(np.sum(N * L, axis=1, keepdims=True), 0.0, None)
        direct += light.color * (light.intensity * ndotl * atten)

        H = L + V
        H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-6)
        ndoth = np.clip(np.sum(N * H, axis=1, keepdims=True), 0.0, None)
        specular += light.color * (light.intensity * spec_k * atten * (ndoth ** shin))

    if shadow_rows is not None:
        sm = shadow_rows[:, None]
        direct *= sm
        specular *= sm

    color = albedo * (ambient + direct) + specular + albedo * emis

    d = np.where(np.isfinite(depth_rows), depth_rows, env.fog_end)
    fog = np.clip((d - env.fog_start) / max(env.fog_end - env.fog_start, 1e-6), 0.0, 1.0)
    fog = ((fog ** 1.4) * env.fog_density)[:, None]
    color = color * (1.0 - fog) + env.fog_color * fog

    color = np.where(cover[:, None], color, env.fog_color)
    return (color * env.exposure).astype(np.float32)


def shade(rast: R.Rasterizer, camera: Camera, lights: List[Light], env: Environment,
          shadow_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Whole-buffer convenience wrapper - used to bake the static lab once."""
    h, w = rast.depth.shape
    rows = shade_rows(
        rast.gbuffer.reshape(-1, R.GB_CHANNELS),
        rast.depth.reshape(-1),
        camera, lights, env,
        None if shadow_mask is None else shadow_mask.reshape(-1),
    )
    return rows.reshape(h, w, 3)


def tonemap(linear_bgr: np.ndarray, gamma: float = 1.0 / 2.2,
            contrast: float = 1.06) -> np.ndarray:
    """Filmic-ish roll-off + gamma -> uint8 BGR ready for cv2.

    Reinhard-style compression keeps specular highlights from clipping to flat
    white, which is what makes a software render read as 'lit' rather than
    'painted'.
    """
    x = np.clip(linear_bgr, 0.0, None)
    x = x / (1.0 + x * 0.72)
    x = np.power(np.clip(x, 0.0, 1.0), gamma)
    x = np.clip((x - 0.5) * contrast + 0.5, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)
