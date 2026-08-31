"""Offline preview of the composed stage (tools/preview_hud.py).

Renders the full display surface to a PNG without a camera, a server or a GPU,
so the layout and typography can be judged - and timed - without standing up the
whole system.

    python tools/preview_hud.py --width 1920 --phase AUTONOMOUS_DEMO
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui import glass as G          # noqa: E402
from src.ui import hud                  # noqa: E402
from src.ui.stage import Stage          # noqa: E402

HOTKEYS = [("v", "talk"), ("a", "auto demo"), ("c", "step"), ("r", "record"),
           ("p", "adapt"), ("f", "ghost"), ("m", "stats"), ("k", "save"),
           ("l", "load"), ("x", "reset"), ("h", "detail"), ("q", "quit")]

MESSAGES = {
    "IDLE": ("STANDBY", "Hold 'v' and say what to pick up, e.g. \"wine glass\""),
    "FORESEEING": ("PREVIEWING", "Watch a replay of your last attempt, or press 'a' for the demo"),
    "WAIT_USER": ("YOUR TURN", "Get in position and press 'c' when you're ready"),
    "USER_EXECUTING": ("GO", "Reach for the coffee cup now - do it your way"),
    "AUTONOMOUS_DEMO": ("AUTONOMOUS DEMO", "Simulating the grasp, using everything learned so far"),
    "ADAPTING": ("REVIEW", "Here's a replay of what you just did"),
}


def fake_camera(w=640, h=480):
    """A room-ish frame: warm window light, a desk, and a mug-shaped blob."""
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y, :] = (58 + y // 9, 62 + y // 11, 70 + y // 14)
    cv2.rectangle(img, (0, int(h * 0.62)), (w, h), (78, 88, 104), -1)
    cv2.rectangle(img, (int(w * 0.05), int(h * 0.06)), (int(w * 0.34), int(h * 0.55)),
                  (120, 168, 214), -1)
    cv2.circle(img, (int(w * 0.66), int(h * 0.30)), 78, (96, 104, 118), -1)
    cv2.ellipse(img, (int(w * 0.52), int(h * 0.66)), (34, 44), 0, 0, 360, (72, 96, 186), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3.0)
    # Signed noise added in int16. Casting it to uint8 first wraps -3 to 253 and
    # turns the whole frame into rainbow static.
    noise = np.random.default_rng(0).normal(0, 3.5, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def depth_map(w=240, h=180):    # the sensor's depth maps are 4:3, like the feed
    g = np.linspace(0, 255, w * h).reshape(h, w).astype(np.uint8)
    g = cv2.GaussianBlur(np.roll(g, 40, axis=1), (0, 0), 6)
    return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)


def render(width: int, height: int, phase: str, motion, stage, cam, depth) -> np.ndarray:
    stage.compose_backdrop(cam)
    L = stage.layout
    s = L.scale
    stage.place_video(cam)

    title, body = MESSAGES.get(phase, MESSAGES["IDLE"])
    col = hud.phase_colour(phase)
    prog = None if phase == "IDLE" else 0.62

    hud.draw_telemetry_card(
        stage, L.telemetry, motion, fps=24.3, latency_ms=88, phase_value=phase,
        target="coffee cup", voice_status="IDLE", adaptation_active=True, reward=0.63,
        error=0.041, loss=0.060, gripper=0.42, robot_connected=True,
        hand_conf=0.92, is_recording=False, recorded_frames=0, scale=s,
        utterance="I'm going to pick up this coffee cup", intent_conditioned=True,
        action="grasp from above, then lift it")
    hud.draw_depth_card(stage, L.depth, depth, s)
    hud.draw_hotkey_card(stage, L.hotkeys, HOTKEYS, s)
    if L.learning:
        rng = np.random.default_rng(3)
        hud.draw_learning_card(stage, L.learning, trials=7, init_err_mm=58.4, updates=57, pulse=0.7,
                               cur_err_mm=31.9,
                               rewards=list(np.cumsum(rng.normal(0.09, 0.3, 26)) * 0.1 - 0.5),
                               scale=s)
    hud.draw_status_bar(stage, L.status, motion, title=title, body=body,
                        colour=col, progress=prog, scale=s)
    hud.draw_banner(stage.canvas, L.video, motion, title=hud.phase_label(phase).title(),
                    colour=col, progress=prog, scale=s)
    return stage.canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--phase", type=str, default="IDLE")
    ap.add_argument("--out", type=str, default="/tmp/hud_preview.png")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    height = a.height or int(round(a.width * 1440 / 3440))

    motion = G.Motion()
    cam, depth = fake_camera(), depth_map()
    probe = Stage(a.width, height)
    rail_w = probe.layout.telemetry[2] - probe.layout.telemetry[0]
    stage = Stage(a.width, height,
                  telemetry_h=hud.telemetry_height(probe.layout.scale),
                  learning_h=hud.learning_height(probe.layout.scale),
                  depth_h=hud.depth_height(probe.layout.scale, rail_w))
    for _ in range(4):
        motion.tick()
        render(a.width, height, a.phase, motion, stage, cam, depth)
    cv2.imwrite(a.out, render(a.width, height, a.phase, motion, stage, cam, depth))
    print(f"wrote {a.out} ({a.width}x{height}, phase {a.phase})")

    if a.bench:
        n = 40
        t = time.perf_counter()
        for _ in range(n):
            motion.tick()
            render(a.width, height, a.phase, motion, stage, cam, depth)
        ms = (time.perf_counter() - t) / n * 1000
        print(f"stage compose: {ms:.1f} ms/frame ({1000 / ms:.0f} fps ceiling)")


if __name__ == "__main__":
    main()
