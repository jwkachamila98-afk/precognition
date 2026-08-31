"""Tests for the stage composition layer (tests/test_ui_stage.py).

These target the failures the previous HUD actually shipped with: panels drawn
on top of one another, a card whose contents ran past its own bottom edge, and
geometry that wrote outside the canvas when a rect was clipped.
"""

import cv2
import numpy as np
import pytest

from src.ui import glass as G
from src.ui import hud
from src.ui.stage import Stage, compute_layout

SIZES = [(1920, 804), (3440, 1440), (2560, 1080), (1600, 900),
         (1440, 900), (1280, 800), (1024, 640), (900, 600), (800, 600)]


def _layout(w, h):
    probe = compute_layout(w, h)
    return compute_layout(w, h, telemetry_h=hud.telemetry_height(probe.scale),
                          learning_h=hud.learning_height(probe.scale))


@pytest.mark.parametrize("size", SIZES)
def test_no_two_panels_ever_overlap(size):
    """The depth inset used to be drawn underneath the status bar. Slots come
    from one layout pass now, so two panels cannot claim the same pixels."""
    L = _layout(*size)
    rects = L.rects()
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            overlap = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
            assert not overlap, f"{size}: {a} overlaps {b}"


@pytest.mark.parametrize("size", SIZES)
def test_every_panel_is_on_screen_and_non_degenerate(size):
    w, h = size
    for r in _layout(w, h).rects():
        assert 0 <= r[0] < r[2] <= w, f"{size}: {r} outside width"
        assert 0 <= r[1] < r[3] <= h, f"{size}: {r} outside height"


@pytest.mark.parametrize("size", SIZES)
def test_cards_are_allocated_at_least_the_height_they_measure(size):
    """A card is measured before the layout runs. If the rail cannot honour the
    measurement the card must be clipped by the layout, never by drawing over
    whatever sits below it."""
    L = _layout(*size)
    needed = hud.telemetry_height(L.scale)
    got = L.telemetry[3] - L.telemetry[1]
    assert got <= needed + 2, "telemetry was given more room than it asked for"
    assert got > 0.55 * needed, f"{size}: telemetry squeezed to {got} of {needed}"


def test_wide_displays_get_two_rails_and_narrow_ones_do_not():
    assert _layout(1920, 804).learning is not None, "an ultrawide should use both sides"
    assert _layout(1024, 640).learning is None, "a narrow display cannot afford two rails"


def test_glass_survives_rects_that_hang_off_the_canvas():
    """Clipped panels must not raise or write outside the buffer."""
    canvas = np.full((240, 320, 3), 90, np.uint8)
    guard = canvas.copy()
    for rect in [(-60, -40, 120, 90), (260, 190, 420, 300), (-10, 100, 5, 140),
                 (300, -20, 700, 400), (10, 10, 11, 11)]:
        G.glass(canvas, rect)
        hud.pill(canvas, rect, hud.C["blue"])
        hud.progress_track(canvas, rect, 0.5, hud.C["teal"])
    assert canvas.shape == guard.shape


def test_panel_cache_matches_an_uncached_draw():
    """The cached glass body must be pixel-identical to drawing it again -
    otherwise the saving shows up as a visible seam every third frame."""
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    rect = (40, 40, 300, 260)
    kw = dict(radius=20, tint=(22, 26, 34), tint_strength=0.6, blur=1.0, highlight=0.5)

    a = Stage(800, 500)
    a.compose_backdrop(frame)
    a.panel_glass(rect, **kw)
    first = a.canvas[rect[1]:rect[3], rect[0]:rect[2]].copy()

    a.compose_backdrop(frame)          # same backdrop generation -> cache hit
    a.panel_glass(rect, **kw)
    cached = a.canvas[rect[1]:rect[3], rect[0]:rect[2]].copy()

    assert np.array_equal(first, cached)


def test_backdrop_fills_the_whole_canvas():
    """The complaint that started this: a third of the display was dead grey.
    Every pixel must carry backdrop, including outside the content group."""
    frame = np.full((480, 640, 3), (40, 90, 160), np.uint8)
    stage = Stage(1600, 670)
    canvas = stage.compose_backdrop(frame)
    for probe in [(5, 5), (1595, 5), (5, 665), (1595, 665), (800, 335)]:
        assert canvas[probe[1], probe[0]].max() > 0, f"{probe} left unpainted"


def test_motion_easing_is_frame_rate_independent():
    """The same elapsed time must produce the same easing whether it arrived in
    a few long frames or many short ones - the frame rate here swings with what
    perception is doing, and a per-frame lerp would animate at a different speed
    every time the workload changed."""
    slow, fast = G.Motion(), G.Motion()
    slow.to("v", 0.0)
    fast.to("v", 0.0)
    for _ in range(6):                      # 6 frames of 50 ms = 300 ms
        slow.tick(0.05)
        slow.to("v", 1.0, speed=8.0)
    for _ in range(30):                     # 30 frames of 10 ms = 300 ms
        fast.tick(0.01)
        fast.to("v", 1.0, speed=8.0)
    assert slow.get("v") == pytest.approx(fast.get("v"), abs=0.03)
    assert 0.85 < slow.get("v") < 1.0, "easing should be well along after 300 ms"


def test_a_card_never_draws_into_the_card_below_it():
    """A card's own content must not reach its neighbour. Panels do cast a soft
    shadow a little past their edge - that is deliberate, and a later panel
    paints over it - so the invariant is about the neighbour's interior, not
    about a hard bounding box."""
    frame = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    stage = Stage(1920, 804, telemetry_h=hud.telemetry_height(1.0),
                  learning_h=hud.learning_height(1.0))
    stage.compose_backdrop(frame)
    L = stage.layout
    motion = G.Motion()

    neighbour = L.hotkeys if L.hotkeys[1] > L.telemetry[3] else L.depth
    inset = 26                                   # clear of the shadow falloff
    interior = (neighbour[0] + inset, neighbour[1] + inset,
                neighbour[2] - inset, neighbour[3] - inset)
    baseline = stage.canvas.copy()
    hud.draw_telemetry_card(
        stage, L.telemetry, motion, fps=24.0, latency_ms=88.0, phase_value="IDLE",
        target="coffee cup", voice_status="IDLE", adaptation_active=True, reward=0.4,
        error=0.04, loss=0.06, gripper=0.5, robot_connected=True, hand_conf=0.9,
        is_recording=True, recorded_frames=99, scale=L.scale)

    changed = np.any(stage.canvas != baseline, axis=2)
    assert changed.any(), "the card drew nothing at all"

    # Nothing of the telemetry card reaches the next card's interior.
    spill = changed[interior[1]:interior[3], interior[0]:interior[2]]
    assert not spill.any(), f"telemetry drew {int(spill.sum())} px inside {neighbour}"

    # And its own content stays within its rect once the shadow is allowed for.
    ys, xs = np.nonzero(changed)
    assert xs.min() >= L.telemetry[0] - inset and xs.max() < L.telemetry[2] + inset
    assert ys.min() >= L.telemetry[1] - inset and ys.max() < L.telemetry[3] + inset


def test_the_spoken_intent_is_shown_and_stays_inside_its_card():
    """The interface showed the parsed target noun but never the utterance it
    came from, in a system whose whole premise is conditioning on what was
    said."""
    frame = np.random.default_rng(5).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    stage = Stage(1920, 804, telemetry_h=hud.telemetry_height(1.0),
                  learning_h=hud.learning_height(1.0))
    stage.compose_backdrop(frame)
    L = stage.layout
    baseline = stage.canvas.copy()

    hud.draw_telemetry_card(
        stage, L.telemetry, G.Motion(), fps=24.0, latency_ms=88.0,
        phase_value="USER_EXECUTING", target="coffee cup", voice_status="IDLE",
        adaptation_active=True, reward=0.4, error=0.04, loss=0.06, gripper=0.5,
        robot_connected=True, hand_conf=0.9, is_recording=False, recorded_frames=0,
        scale=L.scale,
        utterance="I am going to pick up this coffee cup from the desk over there",
        intent_conditioned=True)

    changed = np.any(stage.canvas != baseline, axis=2)
    ys, xs = np.nonzero(changed)
    inset = 26                                   # clear of the shadow falloff
    assert ys.max() < L.telemetry[3] + inset, "a long utterance overflowed the card"
    assert xs.max() < L.telemetry[2] + inset


def test_long_utterances_are_wrapped_and_elided_not_run_off_the_edge():
    # A width narrower than the text, or there is nothing to wrap: the string
    # below measures 331 px at this size.
    width = 250
    lines = hud.wrap(chr(34) + "pick up the small black remote control on the table"
                     + chr(34), 0.38, width)
    assert len(lines) >= 2, "a long line should wrap"
    for line in lines:
        assert cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0] <= width


def test_no_non_ascii_glyphs_are_drawn():
    """OpenCV's Hershey fonts are ASCII-only - typographic quotes rendered as
    a literal '???' on screen."""
    import inspect
    source = inspect.getsource(hud)
    for bad in ("“", "”", "‘", "’", "—", "·"):
        assert bad not in source, f"non-ASCII {bad!r} will not render in a Hershey font"
