"""Liquid-glass surfaces and frame-rate-independent motion (src/ui/glass.py).

The HUD used flat rounded rectangles blended at a fixed alpha. That reads as a
sticker laid on the picture. Real glass does three things a flat blend does not:

  1. It *refracts* what is behind it - the backdrop is blurred, not merely
     darkened, so the panel has depth and the busy camera feed stops competing
     with the text sitting on it.
  2. It catches a *specular* highlight along its upper edge, which is what makes
     a surface read as raised rather than painted on.
  3. It casts a soft *shadow*, separating it from the layer below.

Large-radius blur is the expensive part, so it is done by downscaling, blurring
small, and scaling back up: a box of blurred low-frequency content is visually
indistinguishable from a true wide Gaussian and costs a fraction of it.

Corner masks and edge rings are cached by geometry, so a panel whose size does
not change between frames pays for its mask exactly once.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Supersampling factor for corner masks. Rounded corners drawn directly at final
# size stair-step badly at the radii this design uses; drawn 4x and averaged down
# they come out clean.
_SS = 4

_mask_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
_ring_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
_grad_cache: Dict[Tuple[int, int], np.ndarray] = {}
_shadow_cache: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
_tint_cache: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
_sheen_cache: Dict[tuple, np.ndarray] = {}


def _blend_region(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> None:
    """dst = src*alpha + dst*(1-alpha) over a small region, in place."""
    if dst.size == 0:
        return
    dst[:] = (src.astype(np.float32) * alpha
              + dst.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def composite_rounded(roi: np.ndarray, body: np.ndarray, radius: int) -> None:
    """Draw `body` into `roi` with rounded corners, in place.

    A rounded mask is exactly 1.0 everywhere except four corner squares, so
    per-pixel alpha over the whole panel is almost entirely wasted work: the
    interior is a straight copy and only the corners need blending. On a
    300x250 panel that is the difference between 1.5 ms and 0.06 ms, which is
    what makes several glass layers affordable at video rates.
    """
    h, w = roi.shape[:2]
    r = int(max(0, min(radius, min(w, h) // 2)))
    if r == 0:
        np.copyto(roi, body)
        return
    mask = rounded_mask(w, h, r)
    np.copyto(roi[r:h - r, :], body[r:h - r, :])                   # centre band
    np.copyto(roi[0:r, r:w - r], body[0:r, r:w - r])               # top edge
    np.copyto(roi[h - r:h, r:w - r], body[h - r:h, r:w - r])       # bottom edge
    for ys, xs in ((slice(0, r), slice(0, r)), (slice(0, r), slice(w - r, w)),
                   (slice(h - r, h), slice(0, r)), (slice(h - r, h), slice(w - r, w))):
        _blend_region(roi[ys, xs], body[ys, xs], mask[ys, xs])


def _stroke_edges(roi: np.ndarray, ring: np.ndarray, colour: np.ndarray, radius: int) -> None:
    """Apply a coverage ring only within the border bands it can occupy."""
    h, w = roi.shape[:2]
    band = int(max(3, radius * 0.34 + 2))
    band = min(band, h // 2, w // 2)
    if band <= 0:
        return
    tile = np.empty((0, 0, 3), np.float32)
    for ys in (slice(0, band), slice(h - band, h)):
        sub, a = roi[ys, :], ring[ys, :]
        sub[:] = np.clip(sub.astype(np.float32) * (1.0 - a) + colour * a, 0, 255).astype(np.uint8)
    mid = slice(band, h - band)
    for xs in (slice(0, band), slice(w - band, w)):
        sub, a = roi[mid, xs], ring[mid, xs]
        sub[:] = np.clip(sub.astype(np.float32) * (1.0 - a) + colour * a, 0, 255).astype(np.uint8)
    del tile


def rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Antialiased rounded-rectangle coverage mask, float32 in [0, 1], HxWx1."""
    radius = int(max(0, min(radius, min(w, h) // 2)))
    key = (w, h, radius)
    cached = _mask_cache.get(key)
    if cached is not None:
        return cached

    big = np.zeros((h * _SS, w * _SS), dtype=np.uint8)
    r = radius * _SS
    cv2.rectangle(big, (r, 0), (w * _SS - r, h * _SS), 255, -1)
    cv2.rectangle(big, (0, r), (w * _SS, h * _SS - r), 255, -1)
    for cx, cy in ((r, r), (w * _SS - r, r), (r, h * _SS - r), (w * _SS - r, h * _SS - r)):
        cv2.circle(big, (cx, cy), r, 255, -1)
    mask = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mask = mask[:, :, None]
    _mask_cache[key] = mask
    return mask


def edge_ring(w: int, h: int, radius: int, thickness: int = 1) -> np.ndarray:
    """The panel's outline as a soft coverage ring, float32 HxWx1.

    Taken as the difference between the mask and an inset copy of itself, so the
    stroke follows the rounded corners exactly instead of being drawn twice.
    """
    key = (w, h, radius, thickness)
    cached = _ring_cache.get(key)
    if cached is not None:
        return cached
    outer = rounded_mask(w, h, radius)
    k = 2 * thickness + 1
    inner = cv2.erode(outer[:, :, 0], np.ones((k, k), np.uint8))[:, :, None]
    ring = np.clip(outer - inner, 0.0, 1.0)
    _ring_cache[key] = ring
    return ring


def _vertical_gradient(w: int, h: int) -> np.ndarray:
    """1 at the top edge falling to 0 at the bottom, float32 HxWx1."""
    key = (w, h)
    cached = _grad_cache.get(key)
    if cached is not None:
        return cached
    g = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None, None]
    g = np.repeat(g, w, axis=1)
    _grad_cache[key] = g
    return g


def wide_blur(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """A wide, cheap blur: shrink, blur small, grow back.

    A true Gaussian at the radius this needs would dominate the frame budget.
    Downsampling discards exactly the high frequencies the blur would remove
    anyway, so the result is visually equivalent at a fraction of the cost.
    """
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return img.copy()
    scale = max(0.06, min(0.30, 0.22 / max(strength, 0.2)))
    sw, sh = max(2, int(w * scale)), max(2, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, 2.4 * strength))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def drop_shadow(
    canvas: np.ndarray, rect: Tuple[int, int, int, int], radius: int,
    spread: int = 18, opacity: float = 0.40, dy: int = 6,
) -> None:
    """Darken a soft halo beneath a panel, in place.

    Without this the glass floats on nothing; the shadow is what tells the eye
    the panel is a layer above the picture rather than a hole cut into it.
    """
    x1, y1, x2, y2 = rect
    h, w = canvas.shape[:2]
    px1, py1 = max(0, x1 - spread), max(0, y1 - spread + dy)
    px2, py2 = min(w, x2 + spread), min(h, y2 + spread + dy)
    if px2 <= px1 or py2 <= py1:
        return
    sw, sh = px2 - px1, py2 - py1
    ox1, oy1 = x1 - px1, y1 - py1 + dy
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    ox1, oy1 = max(0, ox1), max(0, oy1)
    ox2, oy2 = min(sw, ox2), min(sh, oy2)
    if ox2 <= ox1 or oy2 <= oy1:
        return

    # The falloff depends only on geometry, never on what is behind the panel,
    # so it is computed once per size and thereafter is a single multiply.
    key = (sw, sh, ox1 * 4096 + oy1, (ox2 - ox1) * 4096 + (oy2 - oy1), radius * 64 + spread)
    occluder = _shadow_cache.get(key)
    if occluder is None:
        occluder = np.zeros((sh, sw), dtype=np.float32)
        occluder[oy1:oy2, ox1:ox2] = rounded_mask(ox2 - ox1, oy2 - oy1, radius)[:, :, 0]
        # Blurred small: a shadow is nothing but low frequencies.
        ds = max(0.10, min(0.5, 24.0 / max(sw, sh)))
        tw, th = max(2, int(sw * ds)), max(2, int(sh * ds))
        small = cv2.resize(occluder, (tw, th), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(1.0, spread * 0.45 * ds))
        occluder = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_LINEAR)
        # Cache the finished attenuation factor, not the coverage: applying the
        # shadow then costs one multiply.
        occluder = np.repeat(occluder[:, :, None], 3, axis=2)
        _shadow_cache[key] = occluder
    akey = key + (int(opacity * 100),)
    atten = _shadow_cache.get(akey)
    if atten is None:
        atten = (1.0 - occluder * opacity).astype(np.float32)
        _shadow_cache[akey] = atten
    roi = canvas[py1:py2, px1:px2]
    cv2.multiply(roi, atten, dst=roi, dtype=cv2.CV_8U)


def glass(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    radius: int = 22,
    tint: Tuple[int, int, int] = (18, 22, 30),
    tint_strength: float = 0.62,
    blur: float = 1.0,
    highlight: float = 0.55,
    shadow: bool = True,
    border: Tuple[int, int, int] = (150, 168, 196),
) -> None:
    """Composite a liquid-glass panel onto `canvas`, in place.

    Order matters: shadow first (it must fall on the untouched backdrop), then
    the refracted body, then the specular edge on top of everything.
    """
    x1, y1, x2, y2 = rect
    h, w = canvas.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    pw, ph = x2 - x1, y2 - y1
    if pw <= 2 or ph <= 2:
        return

    if shadow:
        drop_shadow(canvas, (x1, y1, x2, y2), radius)

    roi = canvas[y1:y2, x1:x2]
    body = wide_blur(roi, blur)

    # Refracted backdrop, pulled toward the tint. Keeping a share of the blurred
    # scene is what separates glass from a painted panel: the panel still
    # carries the colour of whatever is behind it. addWeighted rather than numpy
    # arithmetic - it is the same operation an order of magnitude faster.
    tkey = (pw, ph, int(tint[0]), int(tint[1]), int(tint[2]))
    tint_img = _tint_cache.get(tkey)
    if tint_img is None:
        tint_img = np.full((ph, pw, 3), tint, dtype=np.uint8)
        _tint_cache[tkey] = tint_img
    body = cv2.addWeighted(body, 1.0 - tint_strength, tint_img, tint_strength, 0.0)

    # A faint sheen down the surface, brightest at the top edge.
    if highlight > 0.0:
        skey = (pw, ph, int(highlight * 100))
        sheen = _sheen_cache.get(skey)
        if sheen is None:
            sheen = (_vertical_gradient(pw, ph) * (26.0 * highlight)).astype(np.uint8)
            sheen = np.repeat(sheen, 3, axis=2)
            _sheen_cache[skey] = sheen
        body = cv2.add(body, sheen)

    composite_rounded(roi, body, radius)

    # Specular rim: bright along the top, fading to nothing by the bottom, so
    # the panel catches light from above like a real raised surface would.
    if highlight > 0.0:
        rkey = (pw, ph, radius, int(highlight * 100))
        ring = _sheen_cache.get(("ring",) + rkey)
        if ring is None:
            ring = edge_ring(pw, ph, radius, 1) * (0.25 + 0.75 * _vertical_gradient(pw, ph)) * highlight
            _sheen_cache[("ring",) + rkey] = ring
        _stroke_edges(roi, ring, np.array(border, dtype=np.float32), radius)


def rounded_blit(
    canvas: np.ndarray, image: np.ndarray, rect: Tuple[int, int, int, int],
    radius: int = 22, shadow: bool = True,
) -> None:
    """Place an image into `rect` with rounded corners and a soft shadow."""
    x1, y1, x2, y2 = rect
    pw, ph = x2 - x1, y2 - y1
    if pw <= 2 or ph <= 2:
        return
    if shadow:
        drop_shadow(canvas, rect, radius, spread=26, opacity=0.50, dy=10)
    fitted = cv2.resize(image, (pw, ph), interpolation=cv2.INTER_LINEAR)
    composite_rounded(canvas[y1:y2, x1:x2], fitted, radius)


class Motion:
    """Frame-rate-independent easing for animated values.

    Every animated quantity is addressed by name and eased toward its target
    with an exponential decay evaluated against real elapsed time, so motion
    looks identical at 12 fps and at 60 - which matters here, where the frame
    rate swings with what perception is doing. A fixed per-frame lerp would
    speed up and slow down with the workload.
    """

    def __init__(self) -> None:
        self._values: Dict[str, float] = {}
        self._last = time.perf_counter()
        self._dt = 1.0 / 60.0

    def tick(self, dt: Optional[float] = None) -> float:
        """Advance the clock once per rendered frame. Returns the delta.

        `dt` may be supplied to drive the easing deterministically, which is
        what makes frame-rate independence testable rather than merely claimed.
        """
        now = time.perf_counter()
        self._dt = float(np.clip(now - self._last if dt is None else dt, 1e-4, 0.25))
        self._last = now
        return self._dt

    def to(self, key: str, target: float, speed: float = 9.0) -> float:
        """Ease the named value toward `target`. Higher speed settles sooner."""
        current = self._values.get(key)
        if current is None:
            self._values[key] = float(target)
            return float(target)
        alpha = 1.0 - math.exp(-speed * self._dt)
        current += (float(target) - current) * alpha
        self._values[key] = current
        return current

    def set(self, key: str, value: float) -> None:
        """Snap a value without easing - use when a jump is intended."""
        self._values[key] = float(value)

    def get(self, key: str, default: float = 0.0) -> float:
        return self._values.get(key, default)


def ease_out_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 4.0 * t ** 3 if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_out_back(t: float, overshoot: float = 1.32) -> float:
    """Overshoots slightly then settles - for panels arriving on screen."""
    t = float(np.clip(t, 0.0, 1.0))
    c3 = overshoot + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + overshoot * (t - 1.0) ** 2
