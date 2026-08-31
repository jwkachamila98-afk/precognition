"""Stage chrome, set like a Mac pro app (src/ui/hud.py).

The design language is the one Apple uses for its own dark pro tools - Final
Cut, Logic: near-black surfaces, San Francisco type, one restrained accent,
hairline rules, and no drawn borders. Hierarchy comes from type weight and
tracking, not from boxes around things.

All text goes through src/ui/typography (real SF glyphs, cached sprites);
nothing in this module may touch OpenCV's Hershey fonts. Numbers that update
every frame are eased through `Motion` so they read steadily, and every card is
MEASURED before layout so content can never overrun its card.

Colours are the macOS dark-mode system set, in BGR.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.ui import glass as G
from src.ui import typography as T

Rect = Tuple[int, int, int, int]

# macOS dark-mode system colours, BGR.
C = {
    "label": (247, 245, 245),        # primary text
    "secondary": (162, 158, 155),
    "tertiary": (122, 118, 115),
    "quaternary": (94, 91, 88),
    "blue": (255, 132, 10),          # the single accent
    "green": (88, 209, 48),
    "red": (58, 69, 255),
    "orange": (10, 159, 255),
    "purple": (242, 90, 191),
    "teal": (255, 210, 100),
    "separator": (54, 52, 50),
    "hairline": (44, 42, 41),
    "fill": (40, 38, 37),
}

# The one glass recipe every card shares, so the surfaces read as one material.
CARD = dict(tint=(19, 18, 17), tint_strength=0.80, blur=1.2, highlight=0.13)


def _p(base: float, s: float) -> int:
    """A dimension at the current card scale, never below one pixel."""
    return max(1, int(round(base * s)))


def wrap(text: str, px: int, max_px: int, weight: str = "regular") -> List[str]:
    """Break text into lines that fit `max_px`, measured in the real face."""
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if T.measure(trial, px, weight) <= max_px or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


# ---------------------------------------------------------------- primitives

def pill(canvas, rect: Rect, colour, radius=None, opacity=0.20) -> None:
    """A tinted capsule - the shape a piece of live state wears."""
    h, w = canvas.shape[:2]
    x1, y1 = max(0, rect[0]), max(0, rect[1])
    x2, y2 = min(w, rect[2]), min(h, rect[3])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    r = min(radius if radius is not None else (y2 - y1) // 2,
            (x2 - x1) // 2, (y2 - y1) // 2)
    roi = canvas[y1:y2, x1:x2]
    tint = np.full_like(roi, colour, dtype=np.uint8)
    blended = cv2.addWeighted(tint, opacity, roi, 1.0 - opacity, 0.0)
    G.composite_rounded(roi, blended, r)


def progress_track(canvas, rect: Rect, frac: float, colour, radius=None) -> None:
    """A rounded track with a rounded fill - never a bare rectangle."""
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, rect[0]), max(0, rect[1])
    x2, y2 = min(cw, rect[2]), min(ch, rect[3])
    if x2 - x1 < 3 or y2 - y1 < 2:
        return
    h = y2 - y1
    r = min(radius if radius is not None else h // 2, (x2 - x1) // 2, h // 2)
    roi = canvas[y1:y2, x1:x2]
    track = np.full_like(roi, C["fill"], dtype=np.uint8)
    G.composite_rounded(roi, cv2.addWeighted(track, 0.85, roi, 0.15, 0), r)
    fw = int(round((x2 - x1) * float(np.clip(frac, 0.0, 1.0))))
    if fw >= 2 * r + 1:
        sub = canvas[y1:y2, x1:x1 + fw]
        G.composite_rounded(sub, np.full_like(sub, colour, dtype=np.uint8), r)
    elif fw > 0:
        cv2.circle(canvas, (x1 + r, y1 + r), r, colour, -1, cv2.LINE_AA)


def live_dot(canvas, centre, colour, phase: float, radius: float) -> None:
    """A soft breathing indicator: the halo swells, the core stays put."""
    cx, cy = int(centre[0]), int(centre[1])
    pulse = 0.5 + 0.5 * math.sin(phase)
    r_halo = int(radius * (1.9 + 0.9 * pulse))
    x1, y1 = max(0, cx - r_halo), max(0, cy - r_halo)
    x2, y2 = min(canvas.shape[1], cx + r_halo), min(canvas.shape[0], cy + r_halo)
    if x2 > x1 and y2 > y1:
        sub = canvas[y1:y2, x1:x2]
        tint = np.full_like(sub, colour, dtype=np.uint8)
        a = 0.08 + 0.12 * (1.0 - pulse)
        blended = cv2.addWeighted(tint, a, sub, 1 - a, 0)
        m = G.rounded_mask(x2 - x1, y2 - y1, min(x2 - x1, y2 - y1) // 2)
        sub[:] = (blended.astype(np.float32) * m
                  + sub.astype(np.float32) * (1 - m)).astype(np.uint8)
    cv2.circle(canvas, (cx, cy), int(radius), colour, -1, cv2.LINE_AA)


def sparkline(canvas, rect: Rect, series: Sequence[float], colour,
              lo: float = -1.0, hi: float = 1.0, thickness: int = 2) -> None:
    """A trend line for a short history - shape matters here, not precision."""
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 6:
        return
    cv2.line(canvas, (x1, y1 + h // 2), (x2, y1 + h // 2), C["hairline"], 1, cv2.LINE_AA)
    if not series:
        return
    vals = list(series)[-48:]
    if len(vals) == 1:
        vals = vals * 2
    span = max(hi - lo, 1e-6)
    pts = []
    for i, v in enumerate(vals):
        px_ = x1 + int(round(i * (w - 1) / (len(vals) - 1)))
        norm = float(np.clip((v - lo) / span, 0.0, 1.0))
        pts.append([px_, y2 - 1 - int(round(norm * (h - 2)))])
    poly = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [poly], False, colour, thickness, cv2.LINE_AA)
    cv2.circle(canvas, tuple(poly[-1]), thickness + 1, colour, -1, cv2.LINE_AA)


def _rule(canvas, x1: int, x2: int, y: int) -> None:
    cv2.line(canvas, (x1, y), (x2, y), C["hairline"], 1, cv2.LINE_AA)


def _row(canvas, label: str, value: str, x1: int, x2: int, y: int, colour, s: float,
         value_weight: str = "medium") -> None:
    T.draw(canvas, label, (x1, y), _p(13, s), C["tertiary"])
    T.draw(canvas, value, (x2, y), _p(13, s), colour, weight=value_weight, align="right")


def _eyebrow(canvas, label: str, x: int, y: int, s: float, colour=None) -> int:
    return T.draw_tracked(canvas, label, (x, y), _p(11, s), colour or C["quaternary"])


def _well(canvas, rect: Rect, s: float) -> None:
    """A subtle recessed zone, the pro-app substitute for a bordered box."""
    pill(canvas, rect, C["fill"], radius=_p(10, s), opacity=0.55)


# --------------------------------------------------------------- phase names

_PHASE_COLOUR = {
    "IDLE": "tertiary", "FORESEEING": "orange", "WAIT_USER": "blue",
    "USER_EXECUTING": "green", "ADAPTING": "purple", "RESTARTING": "purple",
    "AUTONOMOUS_DEMO": "teal",
}
_PHASE_SHORT = {
    "WAIT_USER": "READY", "USER_EXECUTING": "EXECUTING",
    "AUTONOMOUS_DEMO": "AUTO DEMO", "FORESEEING": "FORESEEING",
}


def phase_colour(phase_value: str) -> Tuple[int, int, int]:
    return C[_PHASE_COLOUR.get(phase_value.upper(), "tertiary")]


def phase_label(phase_value: str) -> str:
    up = phase_value.upper()
    return _PHASE_SHORT.get(up, up.replace("_", " "))


# ------------------------------------------------------------ telemetry card

# Height of each element at scale 1.0, so the card can be measured before the
# layout runs and its content can never overrun its own bottom edge.
_H = {"head": 86, "tiles": 86, "sec": 26, "row": 26, "gauge": 50,
      "rule": 20, "rec": 26, "quote": 50}
_QUOTE_LINES = 2


def _telemetry_items(*, phase_value, target, voice_status, adaptation_active,
                     error, loss, gripper, robot_connected, hand_conf,
                     is_recording, recorded_frames, utterance=None,
                     intent_conditioned=False, action=None):
    """The card's contents, in order. One source of truth for draw and measure."""
    v_txt, v_col = {
        "LISTENING": ("Listening…", C["teal"]),
        "TRANSCRIBING": ("Transcribing…", C["orange"]),
        "FAILED": ("Not heard — try again", C["red"]),
    }.get(voice_status, ("Hold V to speak", C["quaternary"]))

    items = [
        ("head",),
        ("tiles",),
        ("sec", "Intent"),
        ("quote", utterance or "", intent_conditioned),
        ("row", "Action", (action or "reach and grasp")[:26], C["purple"]),
        ("row", "Target", (target or "standby")[:22], C["orange"]),
        ("row", "Voice", v_txt, v_col),
        ("rule",),
        ("sec", "Adaptation"),
        ("row", "Learning", "Online" if adaptation_active else "Paused",
         C["green"] if adaptation_active else C["red"]),
        ("row", "Discrepancy", f"{error:.3f}", C["label"]),
        ("row", "Net loss", f"{loss:.3f}", C["label"]),
        ("gauge", "Gripper", f"{gripper * 100:.0f}%", float(gripper), C["teal"]),
        ("rule",),
        ("sec", "System"),
        ("row", "Robot", "7-DOF OK" if robot_connected else "Offline",
         C["green"] if robot_connected else C["red"]),
    ]
    if hand_conf is not None:
        items.append(("row", "Hand", f"Tracked · {hand_conf * 100:.0f}%", C["green"]))
    else:
        items.append(("row", "Hand", "Searching…", C["orange"]))
    if is_recording:
        items.append(("rec", f"{recorded_frames} frames"))
    return items


def telemetry_height(scale: float = 1.0, is_recording: bool = False) -> int:
    """What the card needs, so the layout can allocate it before drawing."""
    items = _telemetry_items(
        phase_value="IDLE", target="x", voice_status="IDLE", adaptation_active=True,
        error=0.0, loss=0.0, gripper=0.0, robot_connected=True, hand_conf=None,
        is_recording=is_recording, recorded_frames=0, utterance="x", action="x")
    return int(round((sum(_H[i[0]] for i in items) + 30) * scale))


def draw_telemetry_card(stage, rect: Rect, motion, *, fps, latency_ms, phase_value,
                        target, voice_status, adaptation_active, reward, error,
                        loss, gripper, robot_connected, hand_conf: Optional[float],
                        is_recording, recorded_frames, scale: float,
                        utterance: Optional[str] = None,
                        intent_conditioned: bool = False,
                        action: Optional[str] = None) -> None:
    """The primary readout: quiet labels left, weighted values right."""
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=_p(18, s), **CARD)

    pad = _p(22, s)
    lx, rx = x1 + pad, x2 - pad
    y = y1 + _p(14, s)

    items = _telemetry_items(
        phase_value=phase_value, target=target, voice_status=voice_status,
        adaptation_active=adaptation_active, error=error, loss=loss, gripper=gripper,
        robot_connected=robot_connected, hand_conf=hand_conf,
        is_recording=is_recording, recorded_frames=recorded_frames,
        utterance=utterance, intent_conditioned=intent_conditioned, action=action)

    for item in items:
        kind = item[0]
        step = _p(_H[kind], s)
        if kind == "head":
            T.draw(canvas, "Precognition", (lx, y + _p(28, s)), _p(24, s),
                   C["label"], weight="semibold")
            ph_col = phase_colour(phase_value)
            ph_txt = phase_label(phase_value)
            pw = T.measure(ph_txt, _p(11, s), "semibold") + _p(20, s)
            pill(canvas, (rx - pw, y + _p(8, s), rx, y + _p(8, s) + _p(23, s)),
                 ph_col, opacity=0.22)
            T.draw(canvas, ph_txt, (rx - pw // 2, y + _p(24, s)), _p(11, s),
                   ph_col, weight="semibold", align="center")
            T.draw(canvas, "Visuomotor hand policy", (lx, y + _p(50, s)),
                   _p(12, s), C["tertiary"])
            live_dot(canvas, (rx - _p(5, s), y + _p(46, s)),
                     C["green"] if fps >= 15 else C["orange"],
                     time.time() * 3.0, 3.2 * s)
            _rule(canvas, lx, rx, y + _p(70, s))
        elif kind == "tiles":
            tile_h = _p(66, s)
            gap = _p(10, s)
            tw = (rx - lx - 2 * gap) // 3
            e_fps = motion.to("fps", float(fps), 6.0)
            e_lat = motion.to("lat", float(latency_ms), 6.0)
            e_rew = motion.to("reward", float(reward), 7.0)
            for i, (val, cap, col) in enumerate([
                (f"{e_fps:.1f}", "fps", C["label"]),
                (f"{e_lat:.0f}", "ms", C["label"]),
                (f"{e_rew:+.2f}", "reward",
                 C["green"] if e_rew > 0.5 else C["orange"] if e_rew > 0 else C["red"]),
            ]):
                tx = lx + i * (tw + gap)
                _well(canvas, (tx, y, tx + tw, y + tile_h), s)
                T.draw(canvas, val, (tx + tw // 2, y + _p(36, s)), _p(22, s),
                       col, weight="medium", align="center")
                T.draw_tracked(canvas, cap,
                               (tx + tw // 2 - T.measure(cap.upper(), _p(9, s),
                                                         "semibold") // 2 - _p(2, s),
                                y + _p(56, s)), _p(9, s), C["quaternary"])
        elif kind == "sec":
            _eyebrow(canvas, item[1], lx, y + _p(15, s), s)
        elif kind == "quote":
            _, said, conditioned = item
            if not said:
                T.draw(canvas, "Nothing heard yet", (lx, y + _p(18, s)),
                       _p(13, s), C["quaternary"])
                T.draw(canvas, "Hold V and say what you'll do",
                       (lx, y + _p(38, s)), _p(12, s), C["quaternary"])
            else:
                lines = wrap(f"“{said}”", _p(14, s), rx - lx - _p(6, s))
                for n, line in enumerate(lines[:_QUOTE_LINES]):
                    if n == _QUOTE_LINES - 1 and len(lines) > _QUOTE_LINES:
                        line = line.rstrip(" ”") + "…”"
                    T.draw(canvas, line, (lx, y + _p(18, s) + n * _p(18, s)),
                           _p(14, s), C["label"])
                if conditioned:
                    cv2.circle(canvas, (lx + _p(3, s), y + _p(43, s)),
                               _p(2, s), C["teal"], -1, cv2.LINE_AA)
                    T.draw(canvas, "Conditioning the policy",
                           (lx + _p(11, s), y + _p(47, s)), _p(11, s), C["teal"])
        elif kind == "row":
            _, label, value, colour = item
            _row(canvas, label, value, lx, rx, y + _p(18, s), colour, s)
        elif kind == "gauge":
            _, label, value_txt, frac, colour = item
            T.draw(canvas, label, (lx, y + _p(15, s)), _p(13, s), C["tertiary"])
            T.draw(canvas, value_txt, (rx, y + _p(15, s)), _p(13, s), colour,
                   weight="medium", align="right")
            gy = y + _p(26, s)
            progress_track(canvas, (lx, gy, rx, gy + _p(7, s)),
                           motion.to("grip", float(frac), 10.0), colour)
        elif kind == "rule":
            _rule(canvas, lx, rx, y + _p(11, s))
        elif kind == "rec":
            base = y + _p(18, s)
            cv2.circle(canvas, (lx + _p(4, s), base - _p(4, s)), _p(4, s),
                       C["red"], -1, cv2.LINE_AA)
            T.draw(canvas, "Recording", (lx + _p(15, s), base), _p(13, s), C["tertiary"])
            T.draw(canvas, item[1], (rx, base), _p(13, s), C["red"],
                   weight="medium", align="right")
        y += step


# -------------------------------------------------------------- other panels

def draw_status_bar(stage, rect: Rect, motion, *, title, body, colour,
                    progress: Optional[float], scale: float) -> None:
    """The instruction bar under the video: what to do, in plain language."""
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=_p(16, s), **CARD)
    pad = _p(24, s)
    T.draw_tracked(canvas, title, (x1 + pad, y1 + int((y2 - y1) * 0.40)),
                   _p(11, s), colour)
    T.draw(canvas, body, (x1 + pad, y1 + int((y2 - y1) * 0.74)),
           _p(16, s), C["label"], weight="medium")
    if progress is not None:
        eased = motion.to("phase_progress", float(np.clip(progress, 0, 1)), 7.0)
        bar_h = _p(4, s)
        progress_track(canvas, (x1 + pad, y2 - _p(10, s) - bar_h,
                                x2 - pad, y2 - _p(10, s)), eased, colour)


def draw_depth_card(stage, rect: Rect, heatmap: Optional[np.ndarray], scale: float) -> None:
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=_p(16, s), **CARD)
    pad = _p(16, s)
    _eyebrow(canvas, "Depth", x1 + pad, y1 + _p(26, s), s)
    T.draw(canvas, "metres", (x2 - pad, y1 + _p(26, s)), _p(11, s),
           C["quaternary"], align="right")
    slot = (x1 + pad, y1 + _p(36, s), x2 - pad, y2 - pad)
    if heatmap is None:
        T.draw(canvas, "No depth", ((x1 + x2) // 2, (slot[1] + slot[3]) // 2),
               _p(13, s), C["quaternary"], align="center")
        return
    sw, sh = slot[2] - slot[0], slot[3] - slot[1]
    ih, iw = heatmap.shape[:2]
    k = min(sw / max(iw, 1), sh / max(ih, 1))
    dw, dh = max(2, int(iw * k)), max(2, int(ih * k))
    ox, oy = slot[0] + (sw - dw) // 2, slot[1] + (sh - dh) // 2
    G.rounded_blit(canvas, heatmap, (ox, oy, ox + dw, oy + dh),
                   radius=_p(10, s), shadow=False)


def draw_hotkey_card(stage, rect: Rect, entries: Sequence[Tuple[str, str]],
                     scale: float) -> None:
    """Fixed content on fixed glass, rendered at most once per backdrop."""
    stage.panel_static("hotkeys", rect,
                       lambda cv_, r: _draw_hotkeys(cv_, r, entries, scale))


def _draw_hotkeys(canvas, rect: Rect, entries: Sequence[Tuple[str, str]],
                  scale: float) -> None:
    x1, y1, x2, y2 = rect
    s = scale
    G.glass(canvas, rect, radius=_p(16, s), **CARD)
    pad = _p(16, s)
    _eyebrow(canvas, "Shortcuts", x1 + pad, y1 + _p(26, s), s)

    top = y1 + _p(38, s)
    avail = max(0, (y2 - _p(10, s)) - top)
    step = _p(23, s)
    rows = max(1, avail // max(step, 1))
    cols = 2
    shown = list(entries)[: rows * cols]
    per_col = max(1, (len(shown) + cols - 1) // cols)
    col_w = (x2 - x1 - 2 * pad) // cols
    for i, (key, desc) in enumerate(shown):
        col, rowi = divmod(i, per_col)
        cx = x1 + pad + col * col_w
        cy = top + rowi * step + _p(13, s)
        kw = max(T.measure(key.upper(), _p(10, s), "medium") + _p(12, s), _p(20, s))
        kh = _p(17, s)
        pill(canvas, (cx, cy - kh + _p(4, s), cx + kw, cy + _p(4, s)),
             C["separator"], radius=_p(4, s), opacity=0.85)
        T.draw(canvas, key.upper(), (cx + kw // 2, cy), _p(10, s),
               C["secondary"], weight="medium", align="center")
        T.draw(canvas, desc, (cx + kw + _p(8, s), cy), _p(11, s), C["tertiary"])


def draw_banner(canvas, video_rect: Rect, motion, *, title, colour,
                progress: Optional[float], scale: float) -> None:
    """A floating capsule over the video - the live phase, at a glance."""
    vx1, vy1, vx2, vy2 = video_rect
    s = scale
    txt_w = T.measure(title, _p(13, s), "medium")
    pct_w = T.measure("100%", _p(12, s), "medium") if progress is not None else 0
    bw = _p(46, s) + txt_w + (_p(16, s) + pct_w if pct_w else 0) + _p(18, s)
    bh = _p(36, s)
    cx = (vx1 + vx2) // 2
    bx1, by1 = cx - bw // 2, vy1 + _p(16, s)
    rect = (bx1, by1, bx1 + bw, by1 + bh)
    G.glass(canvas, rect, radius=bh // 2, tint=(16, 15, 14), tint_strength=0.82,
            blur=1.3, highlight=0.12)
    live_dot(canvas, (bx1 + _p(20, s), by1 + bh // 2), colour, time.time() * 3.2, 3.6 * s)
    T.draw(canvas, title, (bx1 + _p(34, s), by1 + int(bh * 0.68)), _p(13, s),
           C["label"], weight="medium")
    if progress is not None:
        eased = motion.to("banner_progress", float(np.clip(progress, 0, 1)), 7.0)
        T.draw(canvas, f"{eased * 100:.0f}%", (bx1 + bw - _p(16, s), by1 + int(bh * 0.68)),
               _p(12, s), colour, weight="medium", align="right")


def learning_height(scale: float = 1.0) -> int:
    """What the learning card needs, so the rail can size it like telemetry."""
    return int(round(352 * scale))


def draw_learning_card(stage, rect: Rect, *, trials: int, init_err_mm: float,
                       cur_err_mm: float, rewards: Sequence[float],
                       scale: float, updates: int = 0,
                       pulse: float = 0.0) -> None:
    """Co-adaptation across trials: is the network actually getting better?

    `updates` is the count of RWR gradient steps the policy has taken and
    `pulse` is 1.0 at the instant of the latest one, decaying to 0 - together
    they let the audience SEE the network training, not merely be told so.
    """
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=_p(16, s), **CARD)
    pad = _p(18, s)
    lx, rx = x1 + pad, x2 - pad
    ey = y1 + _p(26, s)
    ew = _eyebrow(canvas, "Learning", lx, ey, s)

    if updates > 0 or pulse > 0.0:
        # The training heartbeat: flashes with each gradient step and carries
        # the running count, so "it is learning" is a visible event.
        p = float(np.clip(pulse, 0.0, 1.0))
        dot_col = tuple(int(c1 * p + c2 * (1 - p))
                        for c1, c2 in zip(C["teal"], C["quaternary"]))
        live_dot(canvas, (lx + ew + _p(12, s), ey - _p(4, s)), dot_col,
                 time.time() * 2.0, (2.4 + 1.8 * p) * s)
        T.draw(canvas, f"{updates} updates", (rx, ey), _p(11, s),
               C["teal"] if p > 0.25 else C["tertiary"], align="right")

    if trials <= 0:
        T.draw(canvas, "No trials yet", (lx, y1 + _p(58, s)), _p(14, s), C["secondary"])
        T.draw(canvas, "Complete a foresee–execute cycle",
               (lx, y1 + _p(80, s)), _p(12, s), C["quaternary"])
        T.draw(canvas, "to start tracking adaptation.",
               (lx, y1 + _p(98, s)), _p(12, s), C["quaternary"])
        return

    reduction = 0.0 if init_err_mm <= 0 else (init_err_mm - cur_err_mm) / init_err_mm * 100.0
    col = C["green"] if reduction > 0 else C["red"]
    y = y1 + _p(66, s)
    T.draw(canvas, f"{reduction:+.1f}%", (lx, y), _p(28, s), col, weight="semibold")
    T.draw(canvas, f"{trials} trials", (rx, y), _p(12, s), C["tertiary"], align="right")
    T.draw(canvas, "Error reduction vs first attempt", (lx, y + _p(19, s)),
           _p(11, s), C["quaternary"])

    y += _p(46, s)
    _row(canvas, "First", f"{init_err_mm:.1f} mm", lx, rx, y, C["tertiary"], s)
    y += _p(25, s)
    _row(canvas, "Latest", f"{cur_err_mm:.1f} mm", lx, rx, y, C["label"], s)

    y += _p(24, s)
    mean_r = float(np.mean(list(rewards))) if len(rewards) else 0.0
    _row(canvas, "Mean reward", f"{mean_r:+.3f}", lx, rx, y, C["label"], s)

    y += _p(22, s)
    if y + _p(46, s) <= y2 - pad:
        T.draw(canvas, "Episode reward", (lx, y + _p(12, s)), _p(11, s), C["quaternary"])
        band_top = y + _p(18, s)
        band_bottom = min(y2 - pad, band_top + _p(92, s))
        sparkline(canvas, (lx, band_top, rx, band_bottom), list(rewards),
                  C["teal"], thickness=max(1, _p(2, s)))
