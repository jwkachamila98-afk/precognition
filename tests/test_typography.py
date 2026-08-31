"""Real type on the stage (tests/test_typography.py).

Every label was drawn with OpenCV's Hershey fonts - single-stroke plotter
lettering. The machine ships the actual San Francisco face, Pillow rasterises
it, and glyph sprites are cached so a line of text costs a small-region blend.
"""

import numpy as np
import pytest

from src.ui import typography as T

pytestmark = pytest.mark.skipif(not T.available(), reason="no usable font on this host")


def test_text_lands_where_it_was_asked_to():
    canvas = np.zeros((80, 400, 3), np.uint8)
    width = T.draw(canvas, "Precognition", (10, 50), 24, (255, 255, 255))
    assert width > 60
    ys, xs = np.nonzero(canvas.max(axis=2) > 32)
    assert xs.min() >= 8 and xs.max() <= 10 + width + 2
    # Baseline anchoring: most ink sits above the baseline, a descender below.
    assert ys.min() < 50 - 10, "cap height should rise well above the baseline"
    assert ys.max() > 50, "the 'g' descender should drop below it"


def test_alignment_anchors():
    canvas = np.zeros((60, 400, 3), np.uint8)
    w = T.measure("hello", 20)
    T.draw(canvas, "hello", (390, 40), 20, (255, 255, 255), align="right")
    _, xs = np.nonzero(canvas.max(axis=2) > 32)
    assert xs.max() <= 391 and xs.min() >= 390 - w - 2


def test_colour_is_applied_at_blit_time_from_one_cached_sprite():
    """The cache must be colour-independent, or every tint refills it."""
    canvas = np.zeros((60, 200, 3), np.uint8)
    before = len(T._sprites)
    T.draw(canvas, "tint-check", (5, 40), 18, (10, 132, 255))
    after_first = len(T._sprites)
    T.draw(canvas, "tint-check", (5, 40), 18, (88, 209, 48))
    assert len(T._sprites) == after_first, "a second colour must not add a sprite"
    assert after_first > before


def test_clipping_at_canvas_edges_never_raises():
    canvas = np.zeros((40, 100, 3), np.uint8)
    for org in ((-50, 20), (90, 20), (10, -5), (10, 200), (10, 39)):
        T.draw(canvas, "edge case text", org, 18, (255, 255, 255))
    assert canvas.shape == (40, 100, 3)


def test_the_cache_is_bounded():
    for i in range(T._SPRITE_CACHE_MAX + 64):
        T.measure(f"unique-{i}", 12)
    assert len(T._sprites) <= T._SPRITE_CACHE_MAX


def test_tracked_headings_are_uppercased_and_spread():
    canvas = np.zeros((40, 400, 3), np.uint8)
    plain = T.measure("SESSION", 12, "semibold")
    tracked = T.draw_tracked(canvas, "session", (5, 25), 12, (255, 255, 255))
    assert tracked > plain, "tracking should widen the run"


def test_non_ascii_renders_as_glyphs_not_question_marks():
    """The old Hershey path turned curly quotes into literal '???'. Real type
    has the glyphs, so typographic punctuation is allowed again."""
    canvas = np.zeros((60, 300, 3), np.uint8)
    w = T.draw(canvas, "“water cup”", (5, 40), 18, (255, 255, 255))
    assert w > T.measure("water cup", 18), "quotes should occupy real width"
    assert canvas.max() > 0
