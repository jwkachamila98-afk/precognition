"""Offline preview of the simulated lab: renders the Autonomous Demo reenactment
to PNG frames (and an optional MP4) without a camera, server, or GUI.

    PYTHONPATH=. python tools/preview_lab.py --sprite path/to/crop.png --out /tmp/lab

Useful for iterating on the renderer, and for checking frame cost on the target
CPU without running the whole client.
"""

import argparse
import os
import time

import cv2
import numpy as np

from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.perception.scene_parser import BoundingBox3D
from src.simulation.lab_sim import LabSimulator


def synthetic_sprite(w=120, h=88):
    """A stand-in object crop when no real photo-crop is supplied."""
    img = np.full((h, w, 3), (48, 44, 42), dtype=np.uint8)
    cv2.rectangle(img, (int(w * 0.16), int(h * 0.10)), (int(w * 0.84), int(h * 0.90)),
                  (46, 40, 150), -1)
    cv2.rectangle(img, (int(w * 0.16), int(h * 0.10)), (int(w * 0.84), int(h * 0.90)),
                  (30, 26, 90), 2)
    for r in range(4):
        for c in range(3):
            cv2.circle(img, (int(w * (0.30 + c * 0.20)), int(h * (0.24 + r * 0.17))),
                       3, (200, 200, 205), -1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprite", default=None, help="PNG/JPG crop of the target object")
    ap.add_argument("--out", default="/tmp/lab_preview")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--video", action="store_true", help="also write reenactment.mp4")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sprite = cv2.imread(args.sprite) if args.sprite else synthetic_sprite()
    if sprite is None:
        raise SystemExit(f"could not read sprite: {args.sprite}")

    bbox = BoundingBox3D(label="remote control",
                         center=np.array([0.02, 0.05, 0.46], dtype=np.float32),
                         size=np.array([0.15, 0.055, 0.03], dtype=np.float32))
    hand = MockHandTracker().estimate(np.zeros((480, 640, 3), np.uint8))
    affordance = MockAffordanceExtractor().extract_affordance(bbox, "pick up the remote")
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=hand[0] if hand else None, target_object=bbox,
        affordance_map=affordance, intent="pick up the remote", num_steps=60)

    sim = LabSimulator(width=args.width, height=args.height)
    t0 = time.perf_counter()
    ok = sim.prepare(traj, bbox, sprite)
    print(f"prepare: {ok}  ({(time.perf_counter()-t0)*1000:.0f} ms)")
    if not ok:
        raise SystemExit("nothing to render")

    stills = np.linspace(0.0, 1.0, args.frames)
    for i, p in enumerate(stills):
        step = sim.step_for_progress(float(p))
        img = sim.render(step, elapsed=float(p) * 6.0, push_in=float(p))
        path = os.path.join(args.out, f"frame_{i:02d}_p{p:.2f}.png")
        cv2.imwrite(path, img)
        print(f"  {path}  step={step:2d}  {sim.last_render_ms:6.1f} ms  "
              f"hand={sim.hand_screen_height(step):3.0f}px/{args.height}  {sim.telemetry(step)}")

    # Framing check: nothing in the plan may leave the viewport at any step.
    worst = 0.0
    for st in range(len(traj.waypoints)):
        scr, _ = sim.camera.project(sim._hand_paths_lab[st])
        worst = max(worst, float(np.max([
            -scr[:, 0].min(), scr[:, 0].max() - (args.width - 1),
            -scr[:, 1].min(), scr[:, 1].max() - (args.height - 1)])))
    print(f"\nframing: worst overshoot {worst:+.0f}px "
          f"({'CLIPPED' if worst > 0 else 'ok, margin ' + str(int(-worst)) + 'px'})")

    n = 40
    t0 = time.perf_counter()
    for k in range(n):
        sim.render(sim.step_for_progress(k / (n - 1.0)), elapsed=k * 0.05, push_in=k / (n - 1.0))
    dt = (time.perf_counter() - t0) / n
    print(f"\nsteady-state: {dt*1000:.1f} ms/frame  ({1.0/dt:.0f} fps render-only) "
          f"at {args.width}x{args.height}")

    if args.video:
        path = os.path.join(args.out, "reenactment.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                             (args.width, args.height))
        for k in range(180):
            p = k / 179.0
            vw.write(sim.render(sim.step_for_progress(p), elapsed=p * 6.0, push_in=p))
        vw.release()
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
