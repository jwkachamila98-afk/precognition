"""Real typography for the stage (src/ui/typography.py).

Every piece of text in this interface was drawn with OpenCV's Hershey fonts -
single-stroke plotter lettering from the 1960s, the one element no amount of
layout work could make look designed. The machine this runs on ships the actual
San Francisco face Apple sets its own interfaces in, and Pillow can rasterise
it, so there is no reason to keep drawing letters with a pen plotter.

Glyphs are rendered once per (text, size, weight) into a cached alpha mask and
tinted at blit time, so any colour reuses the same sprite. After the first
frame a line of text costs a small-region numpy blend - comparable to
cv2.putText and far below the cost of the window blit.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except Exception:                                       # pragma: no cover
    _PIL = False

# The genuine article first; Helvetica Neue is the closest thing macOS ships
# in a plain ttc if SF ever moves.
_FONT_PATHS = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
)

# SFNS.ttf is a variable font; these are its named weight instances.
_WEIGHT_NAMES = {
    "regular": "Regular", "medium": "Medium",
    "semibold": "Semibold", "bold": "Bold",
}

_faces: Dict[Tuple[int, str], object] = {}
_metrics: Dict[Tuple[int, str], Tuple[int, int]] = {}
_sprites: "OrderedDict[Tuple[str, int, str], Tuple[np.ndarray, int]]" = OrderedDict()
_SPRITE_CACHE_MAX = 768
_font_path: Optional[str] = None


def available() -> bool:
    """Whether real type can be rendered on this machine."""
    return _PIL and _resolve_font() is not None


def _resolve_font() -> Optional[str]:
    global _font_path
    if _font_path is not None:
        return _font_path or None
    import os
    for path in _FONT_PATHS:
        if os.path.exists(path):
            _font_path = path
            return path
    _font_path = ""
    return None


def _face(px: int, weight: str):
    key = (px, weight)
    face = _faces.get(key)
    if face is not None:
        return face
    path = _resolve_font()
    face = ImageFont.truetype(path, px)
    # A variable font carries every weight in one file; ask for the named
    # instance and accept Regular if this build of FreeType cannot oblige.
    try:
        face.set_variation_by_name(_WEIGHT_NAMES.get(weight, "Regular"))
    except Exception:
        pass
    _faces[key] = face
    _metrics[key] = face.getmetrics()
    return face


def _sprite(text: str, px: int, weight: str) -> Tuple[np.ndarray, int]:
    """Alpha mask for a run of text, plus its ascent. Cached, LRU-evicted."""
    key = (text, px, weight)
    hit = _sprites.get(key)
    if hit is not None:
        _sprites.move_to_end(key)
        return hit
    face = _face(px, weight)
    ascent, descent = _metrics[(px, weight)]
    left, _, right, _ = face.getbbox(text)
    width = max(1, right - left + 2)
    height = ascent + descent + 2
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).text((-left + 1, 0), text, font=face, fill=255)
    alpha = np.asarray(img, dtype=np.float32) / 255.0
    entry = (alpha, ascent)
    _sprites[key] = entry
    if len(_sprites) > _SPRITE_CACHE_MAX:
        _sprites.popitem(last=False)
    return entry


def measure(text: str, px: int, weight: str = "regular") -> int:
    """Advance width in pixels."""
    if not text or not available():
        return 0
    return _sprite(text, px, weight)[0].shape[1]


def draw(canvas: np.ndarray, text: str, org: Tuple[int, int], px: int,
         colour: Tuple[int, int, int], weight: str = "regular",
         align: str = "left") -> int:
    """Draw `text` with its BASELINE at org, like cv2.putText. Returns width.

    `align` moves the anchor: "left" (default), "right", or "center".
    """
    if not text:
        return 0
    if not available():                                  # pragma: no cover
        import cv2
        cv2.putText(canvas, text, (int(org[0]), int(org[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, px / 30.0, colour, 1, cv2.LINE_AA)
        return int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 30.0, 1)[0][0])

    alpha, ascent = _sprite(text, px, weight)
    h, w = alpha.shape
    x = int(org[0])
    if align == "right":
        x -= w
    elif align == "center":
        x -= w // 2
    y = int(org[1]) - ascent

    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + w), min(ch, y + h)
    if x2 <= x1 or y2 <= y1:
        return w
    a = alpha[y1 - y:y2 - y, x1 - x:x2 - x][:, :, None]
    roi = canvas[y1:y2, x1:x2]
    tint = np.asarray(colour, dtype=np.float32)
    roi[:] = (roi.astype(np.float32) * (1.0 - a) + tint * a).astype(np.uint8)
    return w


def draw_tracked(canvas: np.ndarray, text: str, org: Tuple[int, int], px: int,
                 colour: Tuple[int, int, int], tracking: float = 0.14,
                 weight: str = "semibold") -> int:
    """Letter-spaced small caps, the SF way of setting a section label.

    Tracking is a fraction of the size, applied between glyphs, and the text is
    uppercased - this is specifically the treatment for eyebrow headings.
    """
    text = text.upper()
    if not available():                                  # pragma: no cover
        return draw(canvas, text, org, px, colour, weight)
    x = float(org[0])
    gap = px * tracking
    for ch in text:
        x += draw(canvas, ch, (int(round(x)), org[1]), px, colour, weight) + gap
    return int(x - org[0])


def line_height(px: int, weight: str = "regular") -> int:
    """Ascent + descent for the face at this size."""
    if not available():
        return int(px * 1.3)
    _face(px, weight)
    ascent, descent = _metrics[(px, weight)]
    return ascent + descent
