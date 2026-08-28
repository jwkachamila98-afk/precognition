"""Procedural BGR textures for the simulated lab.

Generated with numpy + cv2 rather than shipped as image assets, so the repo
stays asset-free and the lab renders identically on any checkout. All functions
are deterministic (fixed seed) so the cached static background is stable across
runs, and results are memoised because the lab geometry is built once.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

_RNG_SEED = 0xC0FFEE


def _rng() -> np.random.Generator:
    return np.random.default_rng(_RNG_SEED)


def _speckle(height: int, width: int, amount: float, rng) -> np.ndarray:
    """Blurred monochrome grain as an (H, W, 1) array, ready to add to a BGR base.

    cv2.GaussianBlur drops a trailing singleton channel, so the axis is restored
    explicitly rather than relying on the blur to preserve it.
    """
    noise = rng.normal(0.0, amount, (height, width)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 0.8)[..., None]


@lru_cache(maxsize=None)
def epoxy_floor(size: int = 512, cell: int = 64) -> np.ndarray:
    """Dark polished-epoxy lab floor with a faint inlaid reference grid."""
    rng = _rng()
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[:] = (34.0, 29.0, 26.0)                       # BGR: cool near-black
    base += _speckle(size, size, 5.0, rng)

    grid = np.zeros((size, size), dtype=np.float32)
    for i in range(0, size + 1, cell):
        cv2.line(grid, (i, 0), (i, size), 1.0, 1, cv2.LINE_AA)
        cv2.line(grid, (0, i), (size, i), 1.0, 1, cv2.LINE_AA)
    grid = cv2.GaussianBlur(grid, (0, 0), 0.6)
    base += grid[..., None] * np.array([46.0, 34.0, 18.0], dtype=np.float32)

    # Wide safety-marking stripe, the kind painted around a real robot cell.
    cv2.line(base, (0, size - cell), (size, size - cell), (28.0, 96.0, 118.0), 3, cv2.LINE_AA)
    return np.clip(base, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def wall_panel(size: int = 512) -> np.ndarray:
    """Modular acoustic wall panels with recessed seams."""
    rng = _rng()
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[:] = (62.0, 57.0, 52.0)
    base += _speckle(size, size, 3.5, rng)

    step = size // 4
    for i in range(0, size + 1, step):
        cv2.line(base, (i, 0), (i, size), (30.0, 27.0, 24.0), 3, cv2.LINE_AA)
        cv2.line(base, (i + 2, 0), (i + 2, size), (84.0, 78.0, 72.0), 1, cv2.LINE_AA)
        cv2.line(base, (0, i), (size, i), (30.0, 27.0, 24.0), 3, cv2.LINE_AA)
        cv2.line(base, (0, i + 2), (size, i + 2), (84.0, 78.0, 72.0), 1, cv2.LINE_AA)
    return np.clip(base, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def brushed_steel(size: int = 512) -> np.ndarray:
    """Brushed stainless worktop - anisotropic horizontal grain."""
    rng = _rng()
    grain = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (63, 1), 0)        # smear along X only
    grain = grain / (np.abs(grain).max() + 1e-6)
    base = np.full((size, size, 3), (84.0, 81.0, 77.0), dtype=np.float32)
    base += grain[..., None] * 20.0
    return np.clip(base, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def calibration_target(size: int = 256, squares: int = 8) -> np.ndarray:
    """A checkerboard calibration target - the one prop that says 'vision lab'."""
    board = np.zeros((size, size, 3), dtype=np.float32)
    board[:] = 235.0
    step = size // squares
    for r in range(squares):
        for c in range(squares):
            if (r + c) % 2 == 0:
                board[r * step:(r + 1) * step, c * step:(c + 1) * step] = 22.0
    cv2.rectangle(board, (0, 0), (size - 1, size - 1), (150.0, 145.0, 140.0), 6)
    cv2.circle(board, (size // 2, size // 2), max(4, step // 6), (40.0, 190.0, 255.0), -1, cv2.LINE_AA)
    return np.clip(board, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def monitor_screen(width: int = 256, height: int = 160) -> np.ndarray:
    """A wall monitor showing a plausible telemetry trace."""
    scr = np.zeros((height, width, 3), dtype=np.float32)
    scr[:] = (26.0, 18.0, 12.0)
    for gy in range(0, height, 16):
        cv2.line(scr, (0, gy), (width, gy), (48.0, 38.0, 30.0), 1)
    for gx in range(0, width, 24):
        cv2.line(scr, (gx, 0), (gx, height), (48.0, 38.0, 30.0), 1)

    xs = np.arange(width)
    for k, (amp, freq, phase, color) in enumerate([
        (0.20, 0.055, 0.0, (255.0, 226.0, 40.0)),
        (0.13, 0.031, 1.7, (120.0, 255.0, 150.0)),
    ]):
        ys = height * (0.5 + amp * np.sin(freq * xs + phase) * np.sin(0.011 * xs + k))
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(scr, [pts], False, color, 2, cv2.LINE_AA)

    cv2.rectangle(scr, (0, 0), (width - 1, height - 1), (70.0, 60.0, 50.0), 2)
    cv2.putText(scr, "TELEMETRY", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (200.0, 200.0, 205.0), 1, cv2.LINE_AA)
    return np.clip(scr, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def rack_face(width: int = 128, height: int = 256) -> np.ndarray:
    """Equipment-rack front: stacked 1U modules with status LEDs."""
    rng = _rng()
    face = np.full((height, width, 3), (40.0, 36.0, 33.0), dtype=np.float32)
    face += _speckle(height, width, 3.0, rng)
    unit = 26
    for i, y in enumerate(range(4, height - unit, unit)):
        cv2.rectangle(face, (5, y), (width - 6, y + unit - 5), (55.0, 50.0, 46.0), -1)
        cv2.rectangle(face, (5, y), (width - 6, y + unit - 5), (24.0, 22.0, 20.0), 1)
        for k in range(3):
            on = (i + k) % 4 != 0
            col = (110.0, 255.0, 140.0) if on else (60.0, 55.0, 52.0)
            cv2.circle(face, (16 + k * 11, y + (unit - 5) // 2), 2, col, -1, cv2.LINE_AA)
        cv2.rectangle(face, (width - 40, y + 6), (width - 12, y + unit - 11),
                      (70.0, 64.0, 58.0), -1)
    return np.clip(face, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def backdrop_panel(size: int = 256) -> np.ndarray:
    """Shadow-box backdrop: darker at the top, so the staged object separates
    from the wall behind it without needing a second light."""
    rng = _rng()
    grad = np.linspace(0.55, 1.0, size, dtype=np.float32)[:, None]
    base = np.full((size, size, 3), (150.0, 146.0, 142.0), dtype=np.float32) * grad[..., None]
    base += _speckle(size, size, 2.5, rng)
    # Faint vertical seam lines, as on a real fabric-covered shadow box.
    for x in range(0, size + 1, size // 3):
        cv2.line(base, (x, 0), (x, size), (0.0, 0.0, 0.0), 1, cv2.LINE_AA)
    return np.clip(base, 0, 255).astype(np.uint8)
