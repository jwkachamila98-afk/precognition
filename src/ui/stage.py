"""Widescreen stage composition and layout (src/ui/stage.py).

Everything used to be drawn onto the 640x480 camera frame, which HighGUI then
padded out to whatever the window was. On a 21:9 display that left roughly a
third of the screen as dead grey, and it meant every label and number was
rasterised at 480 lines and scaled up about 3x - which is why the type looked
soft and the panels looked like stickers.

This module inverts that. A stage canvas is built at the display's own size and
aspect; the camera feed becomes a card placed on it; and the chrome is drawn
directly onto the stage at full resolution. The space that used to be grey
becomes two things: a rail carrying the panels that used to overlap the video,
and a backdrop made from the camera feed itself - scaled to cover, blurred wide
and dimmed. Nothing is empty, and nothing sits on top of anything else.

Layout is computed, not hardcoded. Every panel asks the layout for its rect, so
two panels cannot claim the same pixels - which is how the depth inset ended up
underneath the status bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.ui import glass as G

Rect = Tuple[int, int, int, int]

# The camera card holds the sensor's own aspect; everything else is measured in
# fractions of the stage so the layout survives any window size.
_VIDEO_ASPECT = 4.0 / 3.0
_MAX_RAIL_W = 430
_MIN_RAIL_W = 250


@dataclass(frozen=True)
class Layout:
    """Non-overlapping slots for one stage size. All rects are (x1, y1, x2, y2)."""

    width: int
    height: int
    video: Rect
    status: Rect
    telemetry: Rect
    depth: Rect
    hotkeys: Rect
    learning: Optional[Rect]
    margin: int
    gap: int
    scale: float

    @property
    def video_size(self) -> Tuple[int, int]:
        return self.video[2] - self.video[0], self.video[3] - self.video[1]

    def rects(self) -> Tuple[Rect, ...]:
        base = (self.video, self.status, self.telemetry, self.depth, self.hotkeys)
        return base + ((self.learning,) if self.learning else ())


def compute_layout(width: int, height: int, telemetry_h: int = 0,
                   learning_h: int = 0, depth_h: int = 0) -> Layout:
    """Place the camera card and its rails, centred as a group.

    On a wide display the video is flanked by TWO rails. A single rail left a
    third of an ultrawide screen carrying nothing while the depth preview and
    the shortcut list were squeezed into whatever vertical space the telemetry
    card had not taken - so the panels that needed room were starved next to the
    largest empty region on the display. Splitting them across both sides gives
    every card its natural size and leaves no dead band.

    Narrow displays cannot afford two rails and fall back to one on the right.
    """
    margin = int(round(height * 0.032))
    gap = int(round(height * 0.020))
    status_h = int(round(np.clip(height * 0.092, 62, 104)))

    video_h = height - 2 * margin - status_h - gap
    video_w = int(round(video_h * _VIDEO_ASPECT))

    rail_w = int(np.clip(width * 0.24, _MIN_RAIL_W, _MAX_RAIL_W))
    two_rails = (video_w + 2 * rail_w + 2 * gap) <= (width - 2 * margin)

    n_rails = 2 if two_rails else 1
    group_w = video_w + n_rails * (rail_w + gap)
    if group_w > width - 2 * margin:
        video_w = width - 2 * margin - n_rails * (rail_w + gap)
        video_h = int(round(video_w / _VIDEO_ASPECT))
        group_w = video_w + n_rails * (rail_w + gap)

    x0 = max(margin, (width - group_w) // 2)
    y0 = margin

    left_x = x0 if two_rails else None
    vx = (x0 + rail_w + gap) if two_rails else x0
    video = (vx, y0, vx + video_w, y0 + video_h)
    status = (vx, video[3] + gap, vx + video_w, video[3] + gap + status_h)
    rail_bottom = status[3]
    rail_h = rail_bottom - y0

    scale = rail_w / 430.0

    rx = video[2] + gap
    tel_h = int(telemetry_h) if telemetry_h > 0 else int(rail_h * 0.72)
    tel_h = int(min(tel_h, rail_h - 2 * gap - 80))
    telemetry = (rx, y0, rx + rail_w, y0 + tel_h)

    if two_rails:
        # Right rail: telemetry, with the shortcut list filling the remainder.
        hot_top = telemetry[3] + gap
        hotkeys = (rx, hot_top, rx + rail_w, rail_bottom)
        # Left rail: the depth preview is sized to its content (the heatmap's
        # own aspect) and the learning card takes everything under it - its
        # trend line grows into the extra height, so neither card carries a
        # band of empty glass.
        if depth_h > 0:
            d_h = int(np.clip(depth_h, 100, rail_h - gap - 160))
            depth = (left_x, y0, left_x + rail_w, y0 + d_h)
        else:
            learn_h = int(learning_h) if learning_h > 0 else int(rail_h * 0.46)
            learn_h = int(np.clip(learn_h, 120, rail_h - gap - 140))
            depth = (left_x, y0, left_x + rail_w, rail_bottom - learn_h - gap)
        learning = (left_x, depth[3] + gap, left_x + rail_w, rail_bottom)
    else:
        remaining = rail_h - tel_h - 2 * gap
        depth_h = int(min(rail_w * 0.60, remaining * 0.56))
        depth = (rx, telemetry[3] + gap, rx + rail_w, telemetry[3] + gap + depth_h)
        hotkeys = (rx, depth[3] + gap, rx + rail_w, rail_bottom)
        learning = None

    return Layout(width=width, height=height, video=video, status=status,
                  telemetry=telemetry, depth=depth, hotkeys=hotkeys,
                  learning=learning, margin=margin, gap=gap, scale=scale)


class Stage:
    """The composition surface: backdrop, camera card, and chrome on top."""

    def __init__(self, width: int, height: int, telemetry_h: int = 0,
                 learning_h: int = 0, depth_h: int = 0) -> None:
        self.width = int(width)
        self.height = int(height)
        self._telemetry_h = int(telemetry_h)
        self._learning_h = int(learning_h)
        self._depth_h = int(depth_h)
        self.layout = compute_layout(self.width, self.height, telemetry_h,
                                     learning_h, depth_h)
        self._canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._backdrop = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._vignette: Optional[np.ndarray] = None
        # The backdrop is defocused to the point of carrying no detail, so
        # rebuilding it every frame buys nothing visible while costing a
        # full-stage upscale. Refreshing it every few frames and copying the
        # cached result is markedly cheaper and looks identical in motion.
        self._backdrop_period = 3
        self._backdrop_age = 10 ** 6
        # Panels in the rail sit on the backdrop, never on the live video, so
        # their glass body is identical for as long as the backdrop is. Caching
        # the composited body and re-blitting it turns the expensive part of a
        # panel into a memcpy on two of every three frames; only the text on top
        # is redrawn each time.
        self._backdrop_gen = 0
        self._panel_cache: dict = {}
        self._panel_gen: dict = {}
        # The backdrop is built at this width and upscaled. It carries no detail
        # worth preserving - it exists to be out of focus - so rendering it at
        # roughly a tenth of the stage costs almost nothing and looks smoother
        # than blurring at full size.
        self._bd_w = 192
        self._bd_h = max(2, int(self._bd_w * self.height / max(self.width, 1)))

    def resize(self, width: int, height: int, telemetry_h: int = 0,
               learning_h: int = 0, depth_h: int = 0) -> None:
        if ((int(width), int(height), int(telemetry_h), int(learning_h), int(depth_h))
                == (self.width, self.height, self._telemetry_h, self._learning_h,
                    self._depth_h)):
            return
        self.__init__(width, height, telemetry_h, learning_h, depth_h)

    def _vignette_mask(self) -> np.ndarray:
        """A soft darkening toward the edges, baked at backdrop resolution."""
        if self._vignette is not None:
            return self._vignette
        ys = np.linspace(-1.0, 1.0, self._bd_h, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, self._bd_w, dtype=np.float32)[None, :]
        r = np.sqrt(xs ** 2 + ys ** 2) / 1.414
        v = np.clip(1.0 - 0.62 * r ** 1.7, 0.22, 1.0).astype(np.float32)
        self._vignette = np.repeat(v[:, :, None], 3, axis=2)
        return self._vignette

    def compose_backdrop(self, frame: np.ndarray) -> np.ndarray:
        """Fill the stage with a blurred, dimmed enlargement of the feed.

        The colour of the room therefore washes across the whole display and
        moves as the scene moves, so the area outside the camera card reads as
        depth of field rather than as unused screen.
        """
        self._backdrop_age += 1
        if self._backdrop_age >= self._backdrop_period:
            self._backdrop_age = 0
            small = cv2.resize(frame, (self._bd_w, self._bd_h), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 7.0)
            small = cv2.multiply(small, self._vignette_mask() * 0.52, dtype=cv2.CV_8U)
            cv2.resize(small, (self.width, self.height), dst=self._backdrop,
                       interpolation=cv2.INTER_LINEAR)
            self._backdrop_gen += 1
        np.copyto(self._canvas, self._backdrop)
        return self._canvas

    def place_video(self, video_image: np.ndarray, radius: int = 26) -> Rect:
        """Blit the annotated camera image into its card slot."""
        G.rounded_blit(self._canvas, video_image, self.layout.video, radius=radius)
        return self.layout.video

    def panel_glass(self, rect: Rect, **kwargs) -> None:
        """Composite a glass panel, reusing the last body while the backdrop
        behind it is unchanged."""
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            return
        key = ("g", rect, tuple(sorted((k, str(v)) for k, v in kwargs.items())))
        if self._panel_gen.get(key) == self._backdrop_gen:
            np.copyto(self._canvas[y1:y2, x1:x2], self._panel_cache[key])
            return
        G.glass(self._canvas, rect, **kwargs)
        self._panel_cache[key] = self._canvas[y1:y2, x1:x2].copy()
        self._panel_gen[key] = self._backdrop_gen

    def panel_static(self, key: str, rect: Rect, draw) -> None:
        """Render a panel whose contents never change at most once per backdrop.

        The shortcut card is fixed text on fixed glass; redrawing it - twelve
        capsules and twenty-four strings - every frame is pure waste.
        """
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            return
        ck = ("s", key, rect)
        if self._panel_gen.get(ck) == self._backdrop_gen:
            np.copyto(self._canvas[y1:y2, x1:x2], self._panel_cache[ck])
            return
        draw(self._canvas, rect)
        self._panel_cache[ck] = self._canvas[y1:y2, x1:x2].copy()
        self._panel_gen[ck] = self._backdrop_gen

    def invalidate(self) -> None:
        """Force every cached panel to redraw on the next frame."""
        self._panel_gen.clear()

    @property
    def canvas(self) -> np.ndarray:
        return self._canvas
