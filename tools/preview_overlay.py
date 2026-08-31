"""Offline preview of the LIVE video overlay (tools/preview_overlay.py).

Renders the annotated camera card - skeleton, object box, ghost-hand replay,
affordance hotspots - composed on the full stage, to a PNG. No camera, no
server: a synthetic room, a mock hand, and a fabricated "previous attempt"
recording that reaches for the object, so the overlay can be judged against
the chrome around it without standing up the whole system.

    PYTHONPATH=. python tools/preview_overlay.py --phase FORESEEING
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_parser import AppConfig                    # noqa: E402
from apps.local_client import LocalVisualizer                 # noqa: E402
from src.perception.hand_tracker import HandPose, HandSide    # noqa: E402
from src.perception.scene_parser import BoundingBox3D         # noqa: E402
from src.simulation.trajectory_generator import AffordanceMap  # noqa: E402
from src.ui import glass as G                                  # noqa: E402
from src.ui import hud                                         # noqa: E402
from src.ui.stage import Stage                                 # noqa: E402
from tools.preview_hud import fake_camera, depth_map, HOTKEYS  # noqa: E402

CAM_W, CAM_H = 640, 480


def _hand_pose(wrist_2d, spread=1.0, curl=0.0, z=0.55) -> HandPose:
    """A plausible 21-joint right hand anchored at `wrist_2d` (camera px).

    `curl` closes the fingers toward the palm - 0 is open, 1 is a grasp.
    """
    wx, wy = wrist_2d
    kpts = np.zeros((21, 2), dtype=np.float32)
    kpts[0] = (wx, wy)
    # Finger base directions fanning up from the wrist (screen y is down).
    fingers = {  # base joint: (angle deg from vertical, length px per segment)
        1: (-58, 26), 5: (-24, 30), 9: (-6, 32), 13: (12, 30), 17: (30, 26),
    }
    for base, (ang, seg) in fingers.items():
        a = np.deg2rad(ang * spread)
        d = np.array([np.sin(a), -np.cos(a)], dtype=np.float32)
        n = np.array([d[1], -d[0]], dtype=np.float32)
        p = kpts[0] + d * seg * 1.7
        kpts[base] = p
        for k in range(1, 4):
            bend = curl * 0.55 * k
            step = d * seg * (1.0 - 0.16 * k)
            step = step * (1.0 - bend) + n * seg * bend
            p = p + step
            kpts[base + k] = p
    k3 = np.zeros((21, 3), dtype=np.float32)
    fx = 0.8 * CAM_W
    k3[:, 0] = (kpts[:, 0] - CAM_W / 2) * z / fx
    k3[:, 1] = (kpts[:, 1] - CAM_H / 2) * z / fx
    k3[:, 2] = z
    return HandPose(hand_id=0, side=HandSide.RIGHT, keypoints_3d=k3,
                    keypoints_2d=kpts, confidence=0.94, timestamp=0.0)


def _bbox(label="coffee cup", z=0.62) -> BoundingBox3D:
    fx = 0.8 * CAM_W
    u, v = CAM_W * 0.66, CAM_H * 0.34
    center = np.array([(u - CAM_W / 2) * z / fx, (v - CAM_H / 2) * z / fx, z],
                      dtype=np.float32)
    return BoundingBox3D(label=label, center=center,
                         size=np.array([0.11, 0.12, 0.11], dtype=np.float32))


def _recording(bbox: BoundingBox3D, n=48):
    """A fabricated 'previous attempt': a sweep that ends grasping the object."""
    fx = 0.8 * CAM_W
    obj_uv = np.array([fx * bbox.center[0] / bbox.center[2] + CAM_W / 2,
                       fx * bbox.center[1] / bbox.center[2] + CAM_H / 2])
    start = np.array([CAM_W * 0.24, CAM_H * 0.82])
    poses = []
    for i in range(n):
        t = i / (n - 1)
        e = 1 - (1 - t) ** 2
        pos = start + (obj_uv + np.array([0, 26]) - start) * e
        pos[0] += 30 * np.sin(t * 3.1)        # a human arc, not a straight rail
        p = _hand_pose(pos, curl=min(1.0, max(0.0, (t - 0.55) / 0.35)))
        # Wrist 3D approaches the object so the grasp frame is detected there.
        p.keypoints_3d[0] = bbox.center + (p.keypoints_3d[0] - bbox.center) * (1 - e * 0.94)
        poses.append(p)
    return poses


def render(stage, motion, vis, phase: str, out: str) -> None:
    cam = fake_camera(CAM_W, CAM_H)
    L = stage.layout
    stage.compose_backdrop(cam)

    vw, vh = L.video[2] - L.video[0], L.video[3] - L.video[1]
    frame = cv2.resize(cam, (vw, vh), interpolation=cv2.INTER_LINEAR)
    vis.draw_scale = ((vw / CAM_W) + (vh / CAM_H)) * 0.5
    k = np.array([vw / CAM_W, vh / CAM_H], dtype=np.float32)

    bbox = _bbox()
    live = _hand_pose((CAM_W * 0.30, CAM_H * 0.74))
    replay = _recording(bbox)

    def scaled(p):
        from dataclasses import replace
        return replace(p, keypoints_2d=(p.keypoints_2d * k).astype(np.float32))

    display_live = [scaled(live)]
    display_replay = [scaled(p) for p in replay]

    residuals = [0.012, -0.008, 0.02, 0.004, -0.01, 0.006, 0.0]
    vis.draw_hand_skeleton(frame, display_live, residuals=residuals,
                           adaptation_active=True)
    grasping = phase in ("FORESEEING", "ADAPTING")
    vis.draw_3d_bounding_boxes(frame, [bbox], simplified=grasping)
    hotspots = np.array([bbox.center + [0.012, -0.03, 0.0],
                         bbox.center + [-0.02, 0.01, 0.0]], dtype=np.float32)
    vis.draw_affordance_hotspots(frame, AffordanceMap(
        object_label=bbox.label, surface_points=hotspots,
        contact_probabilities=np.array([0.9, 0.7], dtype=np.float32),
        hotspots=hotspots, intent="pick up the coffee cup"))
    if grasping:
        for _ in range(3):    # settle the exponential ghost smoothing
            vis.anim_frame_idx = 30
            vis.draw_hand_replay(
                frame, display_replay, real_poses=display_live,
                reanchor=(phase == "FORESEEING"),
                label=("Preview · your last attempt" if phase == "FORESEEING"
                       else "Replay · what you just did"),
                target_bbox=bbox, object_sprite=None)

    stage.place_video(frame)
    s = L.scale
    motion.tick()
    hud.draw_telemetry_card(
        stage, L.telemetry, motion, fps=24.3, latency_ms=88, phase_value=phase,
        target="coffee cup", voice_status="IDLE", adaptation_active=True,
        reward=0.63, error=0.041, loss=0.060, gripper=0.42, robot_connected=True,
        hand_conf=0.92, is_recording=False, recorded_frames=0, scale=s,
        utterance="I'm going to pick up this coffee cup", intent_conditioned=True,
        action="grasp from above, then lift it")
    hud.draw_depth_card(stage, L.depth, depth_map(), s)
    hud.draw_hotkey_card(stage, L.hotkeys, HOTKEYS, s)
    if L.learning:
        rng = np.random.default_rng(3)
        hud.draw_learning_card(stage, L.learning, trials=7, init_err_mm=58.4,
                               cur_err_mm=31.9, updates=57, pulse=0.7,
                               rewards=list(np.cumsum(rng.normal(0.09, 0.3, 26)) * 0.1 - 0.5),
                               scale=s)
    col = hud.phase_colour(phase)
    hud.draw_status_bar(stage, L.status, motion, title=phase.title(), colour=col,
                        body="Watch a replay of your last attempt at the coffee cup",
                        progress=0.62, scale=s)
    hud.draw_banner(stage.canvas, L.video, motion, title=hud.phase_label(phase).title(),
                    colour=col, progress=0.62, scale=s)
    cv2.imwrite(out, stage.canvas)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--phase", type=str, default="FORESEEING")
    ap.add_argument("--out", type=str, default="/tmp/overlay_preview.png")
    a = ap.parse_args()
    height = int(round(a.width * 1440 / 3440))

    cfg = AppConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..",
                                           "config", "system_config.yaml"))
    vis = LocalVisualizer(cfg)
    probe = Stage(a.width, height)
    rail_w = probe.layout.telemetry[2] - probe.layout.telemetry[0]
    stage = Stage(a.width, height,
                  telemetry_h=hud.telemetry_height(probe.layout.scale),
                  learning_h=hud.learning_height(probe.layout.scale),
                  depth_h=hud.depth_height(probe.layout.scale, rail_w))
    render(stage, G.Motion(), vis, a.phase, a.out)


if __name__ == "__main__":
    main()
