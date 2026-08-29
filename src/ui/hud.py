"""Stage chrome: the panels that sit on the composition surface (src/ui/hud.py).

Drawn at stage resolution rather than on the 640x480 feed, so type is rasterised
at the size it is displayed instead of being scaled up threefold.

Numbers that update every frame - frame rate, latency, reward, gripper aperture -
are eased through `Motion` rather than printed raw. A readout that jitters
between 12.6 and 19.4 several times a second is unreadable and reads as
instability in the system rather than in the measurement, and easing costs
nothing because the underlying value is still logged exactly.

Colours are the macOS dark-mode system set, in BGR.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.ui import glass as G

Rect = Tuple[int, int, int, int]

# macOS system colours, BGR.
C = {
    "label": (248, 250, 252),
    "secondary": (176, 184, 198),
    "tertiary": (128, 138, 156),
    "quaternary": (92, 100, 116),
    "blue": (255, 132, 10),
    "green": (88, 209, 48),
    "red": (58, 69, 255),
    "orange": (10, 159, 255),
    "purple": (242, 90, 191),
    "teal": (224, 200, 64),
    "separator": (74, 82, 98),
    "fill": (52, 58, 72),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_HEAVY = cv2.FONT_HERSHEY_DUPLEX


def text(canvas, s, org, scale, colour, weight=1, font=_FONT) -> int:
    cv2.putText(canvas, s, (int(org[0]), int(org[1])), font, scale, colour, weight, cv2.LINE_AA)
    return int(cv2.getTextSize(s, font, scale, weight)[0][0])


def text_right(canvas, s, right_x, y, scale, colour, weight=1, font=_FONT) -> int:
    w = cv2.getTextSize(s, font, scale, weight)[0][0]
    cv2.putText(canvas, s, (int(right_x - w), int(y)), font, scale, colour, weight, cv2.LINE_AA)
    return w


def text_centre(canvas, s, cx, y, scale, colour, weight=1, font=_FONT) -> int:
    w = cv2.getTextSize(s, font, scale, weight)[0][0]
    cv2.putText(canvas, s, (int(cx - w / 2), int(y)), font, scale, colour, weight, cv2.LINE_AA)
    return w


def tracked(canvas, s, org, scale, colour, spacing=2.0, weight=1) -> float:
    """Letter-spaced text. OpenCV has no tracking control, so section headings
    are stamped a glyph at a time - wide spacing is what distinguishes a heading
    from body copy without needing a second typeface."""
    x, y = float(org[0]), int(org[1])
    for ch in s:
        cv2.putText(canvas, ch, (int(round(x)), y), _FONT, scale, colour, weight, cv2.LINE_AA)
        x += cv2.getTextSize(ch, _FONT, scale, weight)[0][0] + spacing
    return x


def pill(canvas, rect: Rect, colour, radius=None, opacity=0.20) -> None:
    """A tinted capsule - the shape Apple uses for a piece of live state."""
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
    ring = G.edge_ring(x2 - x1, y2 - y1, r, 1)
    roi[:] = np.clip(roi.astype(np.float32) * (1 - ring)
                     + np.array(colour, np.float32) * ring, 0, 255).astype(np.uint8)


def progress_track(canvas, rect: Rect, frac: float, colour, radius=None) -> None:
    """A rounded progress track with a rounded fill - never a bare rectangle."""
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, rect[0]), max(0, rect[1])
    x2, y2 = min(cw, rect[2]), min(ch, rect[3])
    if x2 - x1 < 3 or y2 - y1 < 2:
        return
    h = y2 - y1
    r = min(radius if radius is not None else h // 2, (x2 - x1) // 2, h // 2)
    roi = canvas[y1:y2, x1:x2]
    track = np.full_like(roi, C["fill"], dtype=np.uint8)
    G.composite_rounded(roi, cv2.addWeighted(track, 0.75, roi, 0.25, 0), r)
    fw = int(round((x2 - x1) * float(np.clip(frac, 0.0, 1.0))))
    if fw >= 2 * r + 1:
        sub = canvas[y1:y2, x1:x1 + fw]
        G.composite_rounded(sub, np.full_like(sub, colour, dtype=np.uint8), r)
    elif fw > 0:
        cv2.circle(canvas, (x1 + r, y1 + r), r, colour, -1, cv2.LINE_AA)


def stat_tile(canvas, rect: Rect, value: str, caption: str, colour, s: float) -> None:
    """One large number over a small caption, on its own inner surface."""
    x1, y1, x2, y2 = rect
    G.glass(canvas, rect, radius=int(14 * s), tint=(30, 34, 44), tint_strength=0.42,
            blur=0.7, highlight=0.30, shadow=False)
    cx = (x1 + x2) // 2
    text_centre(canvas, value, cx, y1 + int((y2 - y1) * 0.56), 0.62 * s, colour, 1, _FONT_HEAVY)
    tw = sum(cv2.getTextSize(ch, _FONT, 0.32 * s, 1)[0][0] + 1.6 * s for ch in caption)
    tracked(canvas, caption, (cx - tw / 2, y2 - int(11 * s)), 0.32 * s, C["tertiary"], 1.6 * s)


def section(canvas, s_label: str, x: int, y: int, s: float) -> None:
    tracked(canvas, s_label, (x, y), 0.34 * s, C["quaternary"], 2.2 * s)


def rule(canvas, x1: int, x2: int, y: int) -> None:
    cv2.line(canvas, (x1, y), (x2, y), C["separator"], 1, cv2.LINE_AA)


def row(canvas, label: str, value: str, x1: int, x2: int, y: int, colour, s: float) -> None:
    text(canvas, label, (x1, y), 0.40 * s, C["tertiary"])
    text_right(canvas, value, x2, y, 0.42 * s, colour)


def live_dot(canvas, centre, colour, phase: float, radius: float) -> None:
    """A soft pulsing indicator. The halo breathes; the core stays put, so it
    reads as a steady light rather than a blinking one."""
    cx, cy = int(centre[0]), int(centre[1])
    pulse = 0.5 + 0.5 * math.sin(phase)
    halo = np.zeros((1, 1), np.uint8)  # placeholder to keep intent obvious
    del halo
    overlay_r = int(radius * (1.9 + 0.9 * pulse))
    sub_x1, sub_y1 = max(0, cx - overlay_r), max(0, cy - overlay_r)
    sub_x2, sub_y2 = min(canvas.shape[1], cx + overlay_r), min(canvas.shape[0], cy + overlay_r)
    if sub_x2 > sub_x1 and sub_y2 > sub_y1:
        sub = canvas[sub_y1:sub_y2, sub_x1:sub_x2]
        tint = np.full_like(sub, colour, dtype=np.uint8)
        a = 0.10 + 0.14 * (1.0 - pulse)
        blended = cv2.addWeighted(tint, a, sub, 1 - a, 0)
        m = G.rounded_mask(sub_x2 - sub_x1, sub_y2 - sub_y1, min(sub_x2 - sub_x1, sub_y2 - sub_y1) // 2)
        sub[:] = (blended.astype(np.float32) * m + sub.astype(np.float32) * (1 - m)).astype(np.uint8)
    cv2.circle(canvas, (cx, cy), int(radius), colour, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Composed panels
# --------------------------------------------------------------------------

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


# Height of each telemetry element at scale 1.0. The card is measured from
# this table before the layout runs, so its content can never overrun its own
# bottom edge and print over the panel below - which is exactly what happened
# when the card was sized to "whatever space was left".
_H = {"head": 76, "tiles": 90, "sec": 26, "chip": 30, "row": 26,
      "gauge": 50, "rule": 24, "rec": 26}


def _telemetry_items(*, phase_value, target, voice_status, adaptation_active,
                     error, loss, gripper, robot_connected, hand_conf,
                     is_recording, recorded_frames):
    """The card's contents, in order. One source of truth for draw and measure."""
    v_txt, v_col = {
        "LISTENING": ("listening", C["teal"]),
        "TRANSCRIBING": ("transcribing", C["orange"]),
        "FAILED": ("not heard - retry", C["red"]),
    }.get(voice_status, ("push 'v'", C["quaternary"]))

    items = [
        ("head",),
        ("tiles",),
        ("sec", "SESSION"),
        ("chip", "stage", phase_label(phase_value), phase_colour(phase_value)),
        ("row", "target", (target or "standby")[:18], C["orange"]),
        ("row", "voice", v_txt, v_col),
        ("rule",),
        ("sec", "ADAPTATION"),
        ("row", "learning", "online" if adaptation_active else "paused",
         C["green"] if adaptation_active else C["red"]),
        ("row", "discrepancy", f"{error:.3f}", C["label"]),
        ("row", "net loss", f"{loss:.3f}", C["label"]),
        ("gauge", "gripper", f"{gripper * 100:.0f}%", float(gripper), C["teal"]),
        ("rule",),
        ("sec", "SYSTEM"),
        ("row", "robot", "7-DOF ok" if robot_connected else "offline",
         C["green"] if robot_connected else C["red"]),
    ]
    if hand_conf is not None:
        items.append(("row", "hand", f"tracked {hand_conf * 100:.0f}%", C["green"]))
    else:
        items.append(("row", "hand", "searching", C["orange"]))
    if is_recording:
        items.append(("rec", f"{recorded_frames} frames"))
    return items


def telemetry_height(scale: float = 1.0, is_recording: bool = False) -> int:
    """What the card needs, so the layout can allocate it before drawing."""
    items = _telemetry_items(
        phase_value="IDLE", target="x", voice_status="IDLE", adaptation_active=True,
        error=0.0, loss=0.0, gripper=0.0, robot_connected=True, hand_conf=None,
        is_recording=is_recording, recorded_frames=0)
    body = sum(_H[i[0]] for i in items)
    return int(round((body + 34) * scale))


def draw_telemetry_card(stage, rect: Rect, motion, *, fps, latency_ms, phase_value,
                        target, voice_status, adaptation_active, reward, error,
                        loss, gripper, robot_connected, hand_conf: Optional[float],
                        is_recording, recorded_frames, scale: float) -> None:
    """The primary readout: muted labels left, values right-aligned into a rail."""
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=int(24 * s), tint=(22, 26, 34),
                      tint_strength=0.66, blur=1.15, highlight=0.55)

    pad = int(22 * s)
    lx, rx = x1 + pad, x2 - pad
    y = y1 + int(16 * s)

    items = _telemetry_items(
        phase_value=phase_value, target=target, voice_status=voice_status,
        adaptation_active=adaptation_active, error=error, loss=loss, gripper=gripper,
        robot_connected=robot_connected, hand_conf=hand_conf,
        is_recording=is_recording, recorded_frames=recorded_frames)

    for item in items:
        kind = item[0]
        step = int(_H[kind] * s)
        if kind == "head":
            text(canvas, "Precognition", (lx, y + int(26 * s)), 0.62 * s,
                 C["label"], 1, _FONT_HEAVY)
            live_dot(canvas, (rx - int(5 * s), y + int(18 * s)),
                     C["green"] if fps >= 15 else C["orange"], time.time() * 3.0, 3.4 * s)
            text(canvas, "visuomotor hand policy", (lx, y + int(48 * s)),
                 0.36 * s, C["quaternary"])
        elif kind == "tiles":
            tile_h = int(64 * s)
            gap = int(9 * s)
            tw = (rx - lx - 2 * gap) // 3
            e_fps = motion.to("fps", float(fps), 6.0)
            e_lat = motion.to("lat", float(latency_ms), 6.0)
            e_rew = motion.to("reward", float(reward), 7.0)
            for i, (val, cap, col) in enumerate([
                (f"{e_fps:.1f}", "FPS", C["green"] if e_fps >= 15 else C["orange"]),
                (f"{e_lat:.0f}", "MS", C["blue"] if e_lat < 300 else C["orange"]),
                (f"{e_rew:+.2f}", "REWARD",
                 C["green"] if e_rew > 0.5 else C["orange"] if e_rew > 0 else C["red"]),
            ]):
                tx = lx + i * (tw + gap)
                stat_tile(canvas, (tx, y, tx + tw, y + tile_h), val, cap, col, s)
        elif kind == "sec":
            section(canvas, item[1], lx, y + int(16 * s), s)
        elif kind == "chip":
            _, label, value, colour = item
            base = y + int(19 * s)
            text(canvas, label, (lx, base), 0.40 * s, C["tertiary"])
            pw = cv2.getTextSize(value, _FONT, 0.36 * s, 1)[0][0] + int(20 * s)
            ph_h = int(24 * s)
            pill(canvas, (rx - pw, base - int(16 * s), rx, base - int(16 * s) + ph_h), colour)
            text_centre(canvas, value, rx - pw // 2, base, 0.36 * s, colour)
        elif kind == "row":
            _, label, value, colour = item
            row(canvas, label, value, lx, rx, y + int(17 * s), colour, s)
        elif kind == "gauge":
            _, label, value_txt, frac, colour = item
            text(canvas, label, (lx, y + int(14 * s)), 0.40 * s, C["tertiary"])
            text_right(canvas, value_txt, rx, y + int(14 * s), 0.40 * s, colour)
            gy = y + int(24 * s)
            progress_track(canvas, (lx, gy, rx, gy + int(10 * s)),
                           motion.to("grip", float(frac), 10.0), colour)
        elif kind == "rule":
            rule(canvas, lx, rx, y + int(12 * s))
        elif kind == "rec":
            base = y + int(17 * s)
            cv2.circle(canvas, (lx + int(4 * s), base - int(4 * s)), int(4 * s),
                       C["red"], -1, cv2.LINE_AA)
            text(canvas, "recording", (lx + int(16 * s), base), 0.40 * s, C["tertiary"])
            text_right(canvas, item[1], rx, base, 0.40 * s, C["red"])
        y += step


def draw_status_bar(stage, rect: Rect, motion, *, title, body, colour,
                    progress: Optional[float], scale: float) -> None:
    """The instruction bar under the video: what to do, in plain language."""
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=int(22 * s), tint=(22, 26, 34),
                      tint_strength=0.66, blur=1.15, highlight=0.50)
    pad = int(26 * s)
    bar_h = int(4 * s)
    cy1 = y1 + int((y2 - y1) * 0.36)
    cv2.rectangle(canvas, (x1 + pad, y1 + int(16 * s)),
                  (x1 + pad + int(3 * s), y2 - int(16 * s)), colour, -1)
    tx = x1 + pad + int(16 * s)
    tracked(canvas, title, (tx, cy1), 0.40 * s, colour, 2.2 * s)
    text(canvas, body, (tx, y1 + int((y2 - y1) * 0.72)), 0.44 * s, C["secondary"])
    if progress is not None:
        eased = motion.to("phase_progress", float(np.clip(progress, 0, 1)), 7.0)
        track = (tx, y2 - int(11 * s) - bar_h, x2 - pad, y2 - int(11 * s))
        progress_track(canvas, track, eased, colour)


def draw_depth_card(stage, rect: Rect, heatmap: Optional[np.ndarray], scale: float) -> None:
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=int(20 * s), tint=(22, 26, 34),
                      tint_strength=0.60, blur=1.0, highlight=0.45)
    pad = int(14 * s)
    head_h = int(26 * s)
    tracked(canvas, "DEPTH", (x1 + pad, y1 + head_h), 0.34 * s, C["quaternary"], 2.2 * s)
    text_right(canvas, "metres", x2 - pad, y1 + head_h, 0.32 * s, C["quaternary"])
    slot = (x1 + pad, y1 + head_h + int(8 * s), x2 - pad, y2 - pad)
    if heatmap is None:
        text_centre(canvas, "no depth", (x1 + x2) // 2, (slot[1] + slot[3]) // 2,
                    0.40 * s, C["quaternary"])
        return
    # Fit rather than fill: stretching a depth map to the card's aspect
    # misrepresents the geometry it exists to show.
    sw, sh = slot[2] - slot[0], slot[3] - slot[1]
    ih, iw = heatmap.shape[:2]
    k = min(sw / max(iw, 1), sh / max(ih, 1))
    dw, dh = max(2, int(iw * k)), max(2, int(ih * k))
    ox, oy = slot[0] + (sw - dw) // 2, slot[1] + (sh - dh) // 2
    G.rounded_blit(canvas, heatmap, (ox, oy, ox + dw, oy + dh),
                   radius=int(12 * s), shadow=False)


def draw_hotkey_card(stage, rect: Rect, entries: Sequence[Tuple[str, str]],
                     scale: float) -> None:
    """Fixed content on fixed glass, so it is rendered at most once per backdrop."""
    stage.panel_static("hotkeys", rect,
                       lambda cv, r: _draw_hotkeys(cv, r, entries, scale))


def _draw_hotkeys(canvas, rect: Rect, entries: Sequence[Tuple[str, str]],
                  scale: float) -> None:
    x1, y1, x2, y2 = rect
    s = scale
    G.glass(canvas, rect, radius=int(20 * s), tint=(22, 26, 34), tint_strength=0.60,
            blur=1.0, highlight=0.45)
    pad = int(14 * s)
    tracked(canvas, "SHORTCUTS", (x1 + pad, y1 + int(26 * s)), 0.34 * s, C["quaternary"], 2.2 * s)

    # Two columns, and the row count is derived from the height actually
    # available rather than assumed - the card shrinks on short displays, and a
    # fixed row count would run its last entries off the bottom edge.
    top = y1 + int(40 * s)
    avail = max(0, (y2 - int(10 * s)) - top)
    step = int(20 * s)
    rows = max(1, avail // max(step, 1))
    cols = 2
    shown = list(entries)[: rows * cols]
    per_col = max(1, (len(shown) + cols - 1) // cols)
    col_w = (x2 - x1 - 2 * pad) // cols
    for i, (key, desc) in enumerate(shown):
        col, rowi = divmod(i, per_col)
        cx = x1 + pad + col * col_w
        cy = top + rowi * step + int(11 * s)
        kw = cv2.getTextSize(key, _FONT, 0.32 * s, 1)[0][0] + int(12 * s)
        kh = int(17 * s)
        pill(canvas, (cx, cy - kh + int(4 * s), cx + kw, cy + int(4 * s)),
             C["separator"], radius=int(5 * s), opacity=0.55)
        text_centre(canvas, key, cx + kw // 2, cy, 0.32 * s, C["secondary"])
        text(canvas, desc, (cx + kw + int(7 * s), cy), 0.33 * s, C["quaternary"])


def draw_banner(canvas, video_rect: Rect, motion, *, title, colour,
                progress: Optional[float], scale: float) -> None:
    """A floating capsule over the top of the video - the live phase, at a glance."""
    vx1, vy1, vx2, vy2 = video_rect
    s = scale
    txt_w = cv2.getTextSize(title, _FONT, 0.44 * s, 1)[0][0]
    # Room for the dot, the gap after the title, and the percentage - measured,
    # not guessed, or the readout prints over the label.
    pct_w = cv2.getTextSize("100%", _FONT, 0.40 * s, 1)[0][0] if progress is not None else 0
    bw = int(56 * s) + txt_w + (int(18 * s) + pct_w if pct_w else 0) + int(20 * s)
    bh = int(42 * s)
    cx = (vx1 + vx2) // 2
    bx1, by1 = cx - bw // 2, vy1 + int(18 * s)
    rect = (bx1, by1, bx1 + bw, by1 + bh)
    G.glass(canvas, rect, radius=bh // 2, tint=(18, 22, 30), tint_strength=0.70,
            blur=1.3, highlight=0.60)
    live_dot(canvas, (bx1 + int(24 * s), by1 + bh // 2), colour, time.time() * 3.2, 4.0 * s)
    text(canvas, title, (bx1 + int(40 * s), by1 + int(bh * 0.63)), 0.44 * s, C["label"])
    if progress is not None:
        eased = motion.to("banner_progress", float(np.clip(progress, 0, 1)), 7.0)
        text_right(canvas, f"{eased * 100:.0f}%", bx1 + bw - int(18 * s),
                   by1 + int(bh * 0.63), 0.40 * s, colour)


def sparkline(canvas, rect: Rect, series: Sequence[float], colour,
              lo: float = -1.0, hi: float = 1.0, thickness: int = 2) -> None:
    """A trend line for a short history - shape matters here, not precision."""
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 6:
        return
    cv2.line(canvas, (x1, y1 + h // 2), (x2, y1 + h // 2), C["separator"], 1, cv2.LINE_AA)
    if not series:
        return
    vals = list(series)[-48:]
    if len(vals) == 1:
        vals = vals * 2
    span = max(hi - lo, 1e-6)
    pts = []
    for i, v in enumerate(vals):
        px = x1 + int(round(i * (w - 1) / (len(vals) - 1)))
        norm = float(np.clip((v - lo) / span, 0.0, 1.0))
        pts.append([px, y2 - 1 - int(round(norm * (h - 2)))])
    poly = np.array(pts, dtype=np.int32)
    # Stroke weight is set by the design, not by how tall the slot happens to
    # be: deriving it from the height made the line grow into a fat ribbon
    # whenever the card had room to spare.
    cv2.polylines(canvas, [poly], False, colour, thickness, cv2.LINE_AA)
    cv2.circle(canvas, tuple(poly[-1]), thickness + 1, colour, -1, cv2.LINE_AA)


def learning_height(scale: float = 1.0) -> int:
    """What the learning card needs, so the rail can size it like telemetry."""
    return int(round(346 * scale))


def draw_learning_card(stage, rect: Rect, *, trials: int, init_err_mm: float,
                       cur_err_mm: float, rewards: Sequence[float],
                       scale: float) -> None:
    """Co-adaptation across trials: is the system actually getting better?

    This is the question the whole loop exists to answer, so it gets a card of
    its own rather than living behind a toggle.
    """
    canvas = stage.canvas
    x1, y1, x2, y2 = rect
    s = scale
    stage.panel_glass(rect, radius=int(20 * s), tint=(22, 26, 34),
                      tint_strength=0.60, blur=1.0, highlight=0.45)
    pad = int(18 * s)
    lx, rx = x1 + pad, x2 - pad
    tracked(canvas, "LEARNING", (lx, y1 + int(26 * s)), 0.34 * s, C["quaternary"], 2.2 * s)

    if trials <= 0:
        text(canvas, "No trials yet", (lx, y1 + int(58 * s)), 0.42 * s, C["secondary"])
        text(canvas, "Complete a foresee-execute cycle", (lx, y1 + int(80 * s)),
             0.34 * s, C["quaternary"])
        text(canvas, "to start tracking adaptation.", (lx, y1 + int(98 * s)),
             0.34 * s, C["quaternary"])
        return

    reduction = 0.0 if init_err_mm <= 0 else (init_err_mm - cur_err_mm) / init_err_mm * 100.0
    col = C["green"] if reduction > 0 else C["red"]
    y = y1 + int(62 * s)
    text(canvas, f"{reduction:+.1f}%", (lx, y), 0.72 * s, col, 1, _FONT_HEAVY)
    text_right(canvas, f"{trials} trials", rx, y, 0.38 * s, C["tertiary"])
    y += int(18 * s)
    text(canvas, "error reduction vs first attempt", (lx, y), 0.32 * s, C["quaternary"])

    y += int(26 * s)
    row(canvas, "first", f"{init_err_mm:.1f} mm", lx, rx, y, C["tertiary"], s)
    y += int(24 * s)
    row(canvas, "latest", f"{cur_err_mm:.1f} mm", lx, rx, y, C["label"], s)

    y += int(24 * s)
    mean_r = float(np.mean(list(rewards))) if len(rewards) else 0.0
    row(canvas, "mean reward", f"{mean_r:+.3f}", lx, rx, y, C["label"], s)

    # The trend sits directly under its own caption in a band of fixed height,
    # rather than being stretched to whatever is left at the bottom of the card.
    y += int(22 * s)
    if y + int(46 * s) <= y2 - pad:
        text(canvas, "episode reward", (lx, y + int(12 * s)), 0.32 * s, C["quaternary"])
        band_top = y + int(18 * s)
        band_bottom = min(y2 - pad, band_top + int(96 * s))
        sparkline(canvas, (lx, band_top, rx, band_bottom), list(rewards), C["teal"],
                  thickness=max(1, int(round(2 * s))))
