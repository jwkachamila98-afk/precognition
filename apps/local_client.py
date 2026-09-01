"""Local OpenCV client application for Intel Mac (apps/local_client.py).

Captures camera frames, executes local or remote perception pipelines, renders 2D/3D skeleton overlays,
3D object bounding primitives, surface affordance hotspots, 60-step 'Foreseen' ghost hand trajectory,
real-time residual adaptation corrections, continuous audio voice transcription (Whisper / Mock),
structured LLM intent reasoning, Staged 'Foresee-then-Execute' State Machine, robot hardware feedback,
policy checkpoint management, multi-trial co-adaptation analytics, and real-time Telemetry HUD.

Supported Modes:
- 'mock_local': Standalone execution on Mac CPU.
- 'mock_remote': Streams compressed frames to ws_server.py via WebSockets.
"""

import argparse
import importlib.util
import asyncio
import collections
from dataclasses import replace
import math
import threading
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse
import cv2
import numpy as np
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_parser import AppConfig
# Must run before torch/ultralytics are imported: they download models over
# HTTPS and cannot be handed a custom SSL context (see src/utils/certs.py).
from src.utils.camera_stream import CameraStream
from src.utils.certs import ensure_ca_bundle
from src.utils.preflight import client_checks, enforce

ensure_ca_bundle()

from src.perception.action_schema import ActionPlan, plan_from_text
from src.perception.gemini_action_parser import GeminiActionParser
from src.perception.hand_tracker import HAND_CONNECTIONS, HandPose, HandTrackerABC
from src.perception.scene_parser import BoundingBox3D, ParsedScene
from src.perception.mediapipe_tracker import MediaPipeHandTracker, MEDIAPIPE_AVAILABLE
from src.perception.intent_parser import IntentParserABC, MockLLMIntentParser, ParsedIntent
from src.audio.notification_sounds import NotificationSounds
from src.audio.speech_to_text import AudioTranscriberABC, GeminiTranscriber, MockTranscriber, WhisperTranscriber
from src.audio.text_to_speech import MockSpeaker, SpeechSynthesizerABC
from src.simulation.trajectory_generator import AffordanceMap, ForeseenTrajectory
from src.simulation.lab_sim import (
    LabSimulator, _MIN_DEMONSTRATION_POSES as LAB_MIN_DEMONSTRATION_POSES)
from src.simulation.render import object_library as OL
from src.perception.gemini_mesh_author import GeminiMeshAuthor
from src.policy.discrepancy import DiscrepancyEngine, DiscrepancyState, EpisodeDiscrepancyReport
from src.policy.workflow_state import ExecutionPhase, WorkflowController
from src.policy.checkpointing import PolicyCheckpointManager
from src.analytics.benchmark import CoAdaptationBenchmark
from src.hardware.robot_interface import MockRobotHardware, RobotHardwareABC, RobotState
from src.safety.safety_monitor import SafetyMonitor, SafetyStatus
from src.mocks.mock_hand_tracker import MockHandTracker
from src.ui import glass as UIG
from src.ui import hud as UIH
from src.ui import typography as T
from src.ui.stage import Stage
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion, minimum_jerk_step
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.mocks.mock_policy import MockResidualPolicy
from src.policy.policy_base import PolicyObservation
from src.transport.ws_client import WSStreamingClient
from src.utils.profiler import LatencyProfiler
from src.utils.recorder import SessionRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LocalClient")

# Preset intent prompts for cycling via keypress 'i'
PRESET_INTENTS = [
    "idle",
    "foresee me picking this remote control",
    "grasp the red coffee cup by the handle",
    "pick up the tall water bottle on the right",
    "grab the stylus pen near the keyboard",
]

# The overlay speaks the same language as the chrome around it: the macOS
# dark-mode system set from src/ui/hud.py, one accent, quiet neutrals. The
# old overlay had its own neon palette, which made the video card read as a
# different app from the panels beside it.
INK = UIH.C
# Drawing ON VIDEO needs two tones the glass chrome doesn't: a bone/line
# colour bright enough to survive any room, and a backing dark enough to
# seat a label on live footage.
BONE = (236, 234, 233)
SCRIM = (14, 13, 12)


_FINGER_BASE_JOINT = {
    1: 1, 2: 1, 3: 1, 4: 1,        # thumb
    5: 5, 6: 5, 7: 5, 8: 5,        # index
    9: 9, 10: 9, 11: 9, 12: 9,     # middle
    13: 13, 14: 13, 15: 13, 16: 13,  # ring
    17: 17, 18: 17, 19: 17, 20: 17,  # pinky
}


def bone_radius(idx1: int, idx2: int) -> float:
    """Approximate anatomical taper: thick at the palm/MCP, narrowing toward
    fingertips, used to give the procedural hand mesh real proportions instead of
    uniform-width bones."""
    lo = min(idx1, idx2)
    if lo == 0:
        return 7.0
    base = _FINGER_BASE_JOINT.get(lo)
    if base is None:
        return 5.0  # palm cross-connections
    tip_dist = lo - base
    return max(3.0, 7.0 - tip_dist * 1.3)


class LocalVisualizer:
    """Renders modern glassmorphism HUD, glowing hand skeletons, holographic ghost trajectories, and sci-fi telemetry."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.show_depth_inset = config.visualization.draw_depth_inset
        self.show_bounding_box = config.visualization.draw_bounding_box
        self.show_foreseen_ghost = True
        self.show_analytics_panel = False
        self.show_telemetry_detail = False
        self.fps_history = collections.deque(maxlen=30)
        self._last_tick = time.perf_counter()
        self.anim_frame_idx = 0
        self._panel_cache: dict = {}
        # Rotation+scale+translation mapping the ghost trajectory's baked-in starting
        # pose onto the user's real, live hand pose. Persists across frames where the
        # real hand is briefly not detected, so the ghost doesn't jump back to its raw
        # stale position.
        self._ghost_transform = (np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
        # Exponentially-smoothed rendered ghost hand pose. The underlying trajectory
        # only refreshes once per server round-trip (~250-400ms), so without smoothing
        # the ghost would visibly teleport to a new pose each update instead of gliding.
        self._smoothed_ghost_kpts: Optional[np.ndarray] = None
        # Annotations used to be drawn on the 640x480 feed and then scaled up
        # roughly threefold into the video card, which left every skeleton line,
        # box and label soft next to the chrome around them. They are now drawn
        # at the card's own resolution; this is the factor between the two, and
        # line weights and type sizes are multiplied by it so the drawing keeps
        # its proportions instead of becoming hairline-thin at the larger size.
        self.draw_scale: float = 1.0

    def _th(self, base: int = 1) -> int:
        """Stroke weight at the current drawing scale, never thinner than 1."""
        return max(1, int(round(base * self.draw_scale)))

    def _px(self, base: float) -> int:
        """A pixel offset or radius at the current drawing scale."""
        return max(1, int(round(base * self.draw_scale)))

    def _rounded_panel_mask(self, w: int, h: int, radius: int):
        """Build (and cache) an anti-aliased rounded-rect alpha mask + border contour."""
        key = (w, h, radius)
        cached = self._panel_cache.get(key)
        if cached is not None:
            return cached
        r = max(0, min(radius, w // 2, h // 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
            cv2.circle(mask, (cx, cy), r, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
        result = (mask_f, contours)
        self._panel_cache[key] = result
        return result

    def update_fps(self) -> float:
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        self.anim_frame_idx += 1
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    def draw_hand_skeleton(
        self,
        frame: np.ndarray,
        poses: List[HandPose],
        residuals: Optional[List[float]] = None,
        adaptation_active: bool = True
    ) -> None:
        """The tracked 21-joint hand, drawn the way a pro tool annotates video:
        thin near-white bones, small joints, the single accent on the
        fingertips. No glow passes, no pulsing halos, no coordinate reticle -
        the hand on screen is the user's own, and decoration on it competes
        with the thing being demonstrated.

        The joints are drawn exactly where the tracker put them. An earlier
        version displaced the fingertips by the policy residuals, which
        rendered the user's REAL hand somewhere it wasn't; corrections belong
        on the plan (see draw_hand_replay), never on the measurement.
        """
        if not self.config.visualization.draw_skeleton:
            return

        h, w = frame.shape[:2]
        for pose in poses:
            kpts_2d = pose.keypoints_2d

            def pt(i: int) -> tuple:
                return (int(np.clip(kpts_2d[i, 0], 0, w - 1)),
                        int(np.clip(kpts_2d[i, 1], 0, h - 1)))

            for u, v in HAND_CONNECTIONS:
                cv2.line(frame, pt(u), pt(v), BONE, self._th(2), cv2.LINE_AA)
            for j_idx in range(21):
                p = pt(j_idx)
                if j_idx in (4, 8, 12, 16, 20):
                    cv2.circle(frame, p, self._px(4), SCRIM, -1, cv2.LINE_AA)
                    cv2.circle(frame, p, self._px(3), INK["blue"], -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, p, self._px(3), SCRIM, -1, cv2.LINE_AA)
                    cv2.circle(frame, p, self._px(2), BONE, -1, cv2.LINE_AA)

    @staticmethod
    def _compute_similarity_transform(
        src_p0: np.ndarray, src_p1: np.ndarray, dst_p0: np.ndarray, dst_p1: np.ndarray,
        scale_limits: tuple = (0.4, 2.5),
    ) -> tuple:
        """2D rotation+scale+translation mapping the (src_p0 -> src_p1) frame onto the
        (dst_p0 -> dst_p1) frame. Used to retarget the ghost hand's baked-in wrist ->
        middle-MCP segment onto the real hand's, so the ghost starts at the same
        position AND orientation as the real hand, not just the same point."""
        v_src = src_p1 - src_p0
        v_dst = dst_p1 - dst_p0
        len_src = float(np.linalg.norm(v_src))
        len_dst = float(np.linalg.norm(v_dst))
        if len_src < 1e-3 or len_dst < 1e-3:
            return np.eye(2, dtype=np.float32), (dst_p0 - src_p0).astype(np.float32)

        scale = float(np.clip(len_dst / len_src, *scale_limits))
        angle = math.atan2(v_dst[1], v_dst[0]) - math.atan2(v_src[1], v_src[0])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32) * scale
        t = dst_p0 - R @ src_p0
        return R.astype(np.float32), t.astype(np.float32)

    def _draw_hand_mesh(
        self,
        frame: np.ndarray,
        kpts_2d: np.ndarray,
        base_color: tuple,
        alpha: float = 0.85,
        light_dir: np.ndarray = np.array([-0.6, -0.8], dtype=np.float32),
    ) -> None:
        """Procedurally-shaded 'solid' hand: capsule bones (tapered per bone_radius)
        plus spherical joints, each with a directional highlight/shadow stripe to fake
        cylindrical/spherical volume. This is real anatomical-ish geometry and shading,
        not a flat wireframe - but it is NOT an anatomically exact mesh. That would be
        MANO, the actual academic hand model already referenced throughout this
        codebase's naming (MANOParameters) - MANO requires registering for a license at
        the MPI project site and isn't something this project can legally ship or fetch
        automatically, so it's out of scope here by design, not oversight."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        light = light_dir / (np.linalg.norm(light_dir) + 1e-6)

        def clip_pt(p: np.ndarray) -> tuple:
            return (int(np.clip(p[0], 0, w - 1)), int(np.clip(p[1], 0, h - 1)))

        darker = tuple(max(0, int(c * 0.45)) for c in base_color)
        lighter = tuple(min(255, int(c * 1.7)) for c in base_color)

        for idx1, idx2 in HAND_CONNECTIONS:
            p1, p2 = kpts_2d[idx1], kpts_2d[idx2]
            radius = bone_radius(idx1, idx2)
            direction = p2 - p1
            length = float(np.linalg.norm(direction))
            p1c, p2c = clip_pt(p1), clip_pt(p2)

            cv2.line(overlay, p1c, p2c, base_color, thickness=int(radius * 2), lineType=cv2.LINE_AA)
            cv2.circle(overlay, p1c, int(radius), base_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, p2c, int(radius), base_color, -1, cv2.LINE_AA)

            if length > 1e-3:
                unit = direction / length
                normal = np.array([-unit[1], unit[0]], dtype=np.float32)
                side = 1.0 if np.dot(normal, light) < 0 else -1.0
                hi_off = normal * side * radius * 0.4
                sh_off = -normal * side * radius * 0.4
                thick = max(1, int(radius * 0.5))
                cv2.line(overlay, clip_pt(p1 + hi_off), clip_pt(p2 + hi_off), lighter, thickness=thick, lineType=cv2.LINE_AA)
                cv2.line(overlay, clip_pt(p1 + sh_off), clip_pt(p2 + sh_off), darker, thickness=thick, lineType=cv2.LINE_AA)

        for j_idx in range(21):
            p = kpts_2d[j_idx]
            pc = clip_pt(p)
            radius = 8 if j_idx == 0 else max(3, int(bone_radius(j_idx, j_idx) * 1.15))
            cv2.circle(overlay, pc, radius, darker, -1, cv2.LINE_AA)
            hi_pc = clip_pt(p - light * radius * 0.4)
            cv2.circle(overlay, hi_pc, max(1, int(radius * 0.5)), lighter, -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    @staticmethod
    def _bbox_screen_rect(target_bbox: BoundingBox3D, w: int, h: int) -> tuple:
        """Project a BoundingBox3D's center + size to a 2D screen-space
        (center_xy, px_w, px_h) rectangle, using the same default pinhole
        convention used throughout this project (BoundingBox3D.project_to_2d)."""
        depth = max(float(target_bbox.center[2]), 0.1)
        fx = fy = 0.8 * w
        cx, cy = w / 2.0, h / 2.0
        center = np.array(
            [fx * target_bbox.center[0] / depth + cx, fy * target_bbox.center[1] / depth + cy],
            dtype=np.float32,
        )
        px_w = max(10, int(fx * float(target_bbox.size[0]) / depth))
        px_h = max(10, int(fy * float(target_bbox.size[1]) / depth))
        return center, px_w, px_h

    def capture_object_sprite(self, frame: np.ndarray, target_bbox: BoundingBox3D) -> Optional[np.ndarray]:
        """Crop a real snapshot of the target object out of the live camera frame.

        This is what makes the ghost afterimage look like the actual object
        instead of an abstract colored blob: rather than synthesizing a shape,
        we grab a real photo of it while it's plainly visible (called only when
        NOT mid-grasp, so the real hand isn't occluding it) and stamp that photo
        into the afterimage later."""
        h, w = frame.shape[:2]
        center, px_w, px_h = self._bbox_screen_rect(target_bbox, w, h)
        x0, y0 = int(center[0] - px_w / 2), int(center[1] - px_h / 2)
        x1, y1 = x0 + px_w, y0 + px_h
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        if x1c - x0c < 6 or y1c - y0c < 6:
            return None
        return frame[y0c:y1c, x0c:x1c].copy()

    def _stamp_object_sprite(
        self,
        frame: np.ndarray,
        sprite: np.ndarray,
        center: tuple,
        px_w: int,
        px_h: int,
        tint: tuple = INK["teal"],
        alpha: float = 0.80,
    ) -> None:
        """Blend a captured real photo of the object into the frame at (center),
        resized to (px_w, px_h), soft-masked to an oval cutout, and lightly
        tinted so it still reads as a holographic afterimage rather than a
        second physical copy of the object sitting in the scene."""
        h, w = frame.shape[:2]
        px_w, px_h = max(6, px_w), max(6, px_h)
        resized = cv2.resize(sprite, (px_w, px_h), interpolation=cv2.INTER_LINEAR)

        tint_layer = np.full_like(resized, tint)
        tinted = cv2.addWeighted(resized, 0.62, tint_layer, 0.38, 0)

        mask = np.zeros((px_h, px_w), dtype=np.uint8)
        cv2.ellipse(mask, (px_w // 2, px_h // 2), (px_w // 2, px_h // 2), 0, 0, 360, 255, -1, cv2.LINE_AA)
        blur_sigma = max(1.0, px_w * 0.05)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
        mask_f = (mask.astype(np.float32) / 255.0) * alpha

        x0, y0 = int(center[0] - px_w / 2), int(center[1] - px_h / 2)
        x1, y1 = x0 + px_w, y0 + px_h
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(w, x1), min(h, y1)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sx0, sy0 = fx0 - x0, fy0 - y0
        sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)

        roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        sprite_roi = tinted[sy0:sy1, sx0:sx1].astype(np.float32)
        mask_roi = mask_f[sy0:sy1, sx0:sx1][..., None]
        frame[fy0:fy1, fx0:fx1] = (roi * (1.0 - mask_roi) + sprite_roi * mask_roi).astype(np.uint8)

        cv2.ellipse(frame, (int(center[0]), int(center[1])), (px_w // 2, px_h // 2),
                    0, 0, 360, tint, self._th(1), cv2.LINE_AA)

    def _draw_object_replay_afterimage(
        self,
        frame: np.ndarray,
        replay_poses: List[HandPose],
        idx: int,
        xf,
        target_bbox: BoundingBox3D,
        sprite: Optional[np.ndarray] = None,
    ) -> None:
        """Ghost afterimage of the object as it would be manipulated by the
        replayed hand motion.

        Anchored to target_bbox - the REAL, currently detected object position
        (ground truth, refreshed every frame) - never a simulated position,
        since a real recorded hand replay carries no object-physics data of its
        own to draw from.

        Contact is approximated as the replay frame where the recorded wrist
        comes closest to the object in 3D (the real hand tracker's own 3D
        estimate). Before that frame the object hasn't moved, so it's drawn
        exactly at the real bbox position. From that frame onward, the object
        ghost follows the SAME relative 2D screen displacement the wrist
        undergoes since the contact moment (in the replay's own re-anchored
        space via xf) - approximating "the object moves rigidly with the hand
        once grasped" without a separate physics simulation.

        The CURRENT step is rendered as a real photo-crop of the object (see
        capture_object_sprite/_stamp_object_sprite) when one is available, so it
        looks like the actual object rather than a flat colored shape. The
        fading trail behind it stays a simple colored blob - stamping a full
        image at every trail step would be visually busy and costly for what's
        meant to be a quick motion cue.
        """
        h, w = frame.shape[:2]
        bbox_2d, px_w, px_h = self._bbox_screen_rect(target_bbox, w, h)

        dists = [float(np.linalg.norm(p.keypoints_3d[0] - target_bbox.center)) for p in replay_poses]
        grasp_idx = int(np.argmin(dists))
        grasp_wrist_2d = replay_poses[grasp_idx].keypoints_2d[0].reshape(1, 2)

        def anchored_2d(step: int) -> np.ndarray:
            if step < grasp_idx:
                return bbox_2d
            wrist_2d = replay_poses[step].keypoints_2d[0].reshape(1, 2)
            delta = xf(wrist_2d)[0] - xf(grasp_wrist_2d)[0]
            return bbox_2d + delta

        trail_span = 24
        trail_steps = sorted(set(range(max(0, idx - trail_span), idx, 4)) | {idx})

        overlay = frame.copy()
        for s in trail_steps:
            if s == idx:
                continue
            center = anchored_2d(s)
            cx_i, cy_i = int(np.clip(center[0], 0, w - 1)), int(np.clip(center[1], 0, h - 1))
            age = (idx - s) / float(trail_span)
            in_contact = s >= grasp_idx
            color = INK["teal"] if in_contact else tuple(int(c * 0.55) for c in INK["teal"])
            cv2.ellipse(overlay, (cx_i, cy_i), (px_w // 2, px_h // 2), 0, 0, 360, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.30 * (1.0 - age), frame, 1.0 - 0.30 * (1.0 - age), 0, dst=frame)
            overlay = frame.copy()

        # Current step: a real photo-crop of the object if we have one, else the
        # same colored-ellipse fallback used before this feature existed.
        center = anchored_2d(idx)
        cx_i, cy_i = int(np.clip(center[0], 0, w - 1)), int(np.clip(center[1], 0, h - 1))
        if sprite is not None and sprite.size > 0:
            self._stamp_object_sprite(frame, sprite, (cx_i, cy_i), px_w, px_h)
        else:
            cv2.ellipse(frame, (cx_i, cy_i), (px_w // 2, px_h // 2), 0, 0, 360,
                        INK["teal"], self._th(1), cv2.LINE_AA)

    def draw_hand_replay(
        self,
        frame: np.ndarray,
        replay_poses: Optional[List[HandPose]],
        real_poses: Optional[List[HandPose]] = None,
        reanchor: bool = False,
        label: str = "",
        color: tuple = INK["teal"],
        target_bbox: Optional[BoundingBox3D] = None,
        object_sprite: Optional[np.ndarray] = None,
    ) -> int:
        """Render an afterimage/replay of a REAL previously-recorded hand motion -
        never a synthetic generated plan. Two uses:

        - reanchor=True: the preview shown during FORESEEING for the 2nd+ attempt at
          the same object. It replays the user's OWN previous attempt, re-anchored
          (2D similarity transform, wrist -> middle-MCP) to start from wherever the
          real hand currently is, so it reads as "here's roughly what you did last
          time, from here." Nothing is drawn on the very first attempt (no prior
          recording yet) - callers pass replay_poses=None in that case.
        - reanchor=False: the post-execution review moment during ADAPTING. It
          replays the attempt that JUST finished at its own real recorded screen
          positions, unmodified - literally "here's what you just did."

        Never called during USER_EXECUTING - the user's real hand is unobstructed
        while actually performing the action."""
        if not self.show_foreseen_ghost or not replay_poses:
            self._smoothed_ghost_kpts = None
            return 0

        h, w = frame.shape[:2]
        num_frames = len(replay_poses)
        idx = self.anim_frame_idx % num_frames
        current_pose = replay_poses[idx]

        if reanchor and real_poses and len(real_poses[0].keypoints_2d) > 9:
            real_kpts = real_poses[0].keypoints_2d
            replay_kpts0 = replay_poses[0].keypoints_2d
            anchored_to_object = False
            if target_bbox is not None and num_frames > 1:
                # Anchor the PATH, not the starting pose: map the recorded
                # (start wrist -> grasp wrist) segment onto (live wrist ->
                # object). The ghost then leaves from wherever the real hand is
                # AND its grasp frame lands on the object where it is NOW - so
                # it visibly closes on the live bounding box instead of
                # grabbing empty air wherever the object used to be. Both ends
                # are re-read every frame, so it follows a moving hand and a
                # moving object alike.
                bbox_2d, _, _ = self._bbox_screen_rect(target_bbox, w, h)
                dists = [float(np.linalg.norm(p.keypoints_3d[0] - target_bbox.center))
                         for p in replay_poses]
                grasp_idx = int(np.argmin(dists))
                grasp_wrist = replay_poses[grasp_idx].keypoints_2d[0]
                # A degenerate recording (grasp at the very first frame, or a
                # reach too short to define a direction) can't anchor a path.
                if grasp_idx > 0 and np.linalg.norm(grasp_wrist - replay_kpts0[0]) > 24.0:
                    self._ghost_transform = self._compute_similarity_transform(
                        replay_kpts0[0], grasp_wrist, real_kpts[0], bbox_2d,
                        # Wider than the pose-matching clamp: reaching the
                        # object is the point, however far away it now is.
                        scale_limits=(0.2, 5.0),
                    )
                    anchored_to_object = True
            if not anchored_to_object:
                self._ghost_transform = self._compute_similarity_transform(
                    replay_kpts0[0], replay_kpts0[9], real_kpts[0], real_kpts[9]
                )
            R, t = self._ghost_transform
        else:
            R, t = np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

        def xf(pts_2d: np.ndarray) -> np.ndarray:
            return pts_2d @ R.T + t

        # Object afterimage (drawn first, underneath the hand)
        if target_bbox is not None:
            self._draw_object_replay_afterimage(frame, replay_poses, idx, xf, target_bbox, sprite=object_sprite)

        # Shimmering trail of the wrist path across the whole recorded motion.
        trail_pts_raw = xf(np.stack([p.keypoints_2d[0] for p in replay_poses]))
        trail_pts = [
            (int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))) for u, v in trail_pts_raw
        ]
        if len(trail_pts) > 1:
            # One colour, brightening toward the destination - the path reads
            # as a single gesture, not a rainbow.
            for i in range(len(trail_pts) - 1):
                along = i / float(len(trail_pts))
                k = 0.30 + 0.70 * along
                trail_color = tuple(int(c * k) for c in color)
                cv2.line(frame, trail_pts[i], trail_pts[i + 1], trail_color,
                         self._th(1 + along), lineType=cv2.LINE_AA)

        # Holographic afterimage hand, smoothed frame-to-frame so the loop glides
        # rather than snapping between recorded samples.
        target_kpts_2d = xf(current_pose.keypoints_2d)
        if self._smoothed_ghost_kpts is None or self._smoothed_ghost_kpts.shape != target_kpts_2d.shape:
            self._smoothed_ghost_kpts = target_kpts_2d.copy()
        else:
            self._smoothed_ghost_kpts = self._smoothed_ghost_kpts + 0.35 * (target_kpts_2d - self._smoothed_ghost_kpts)
        ghost_kpts_2d = self._smoothed_ghost_kpts
        self._draw_hand_mesh(frame, ghost_kpts_2d, color, alpha=0.80)

        if label:
            wrist_pt = (int(np.clip(ghost_kpts_2d[0, 0], 0, w - 1)),
                        int(np.clip(ghost_kpts_2d[0, 1], 0, h - 1)))
            self._video_label(frame, label,
                              (wrist_pt[0] - self._px(60), wrist_pt[1] + self._px(30)),
                              colour=color, px=11)

        return idx + 1

    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return 1.0 - (1.0 - t) ** 3

    def lab_panel_rect(self, frame_shape: tuple, lab_shape: tuple) -> tuple:
        """Where the simulated-lab viewport sits when fully open.

        Sized to leave the top status banner and the bottom instruction bar
        uncovered - those two carry the workflow state, and losing them for the
        six seconds of the demo would be a downgrade.
        """
        h, w = frame_shape[:2]
        lab_h, lab_w = lab_shape[:2]
        top, bottom = 48, 74
        ph = max(80, h - top - bottom)
        pw = int(round(ph * lab_w / max(lab_h, 1)))
        if pw > w - 24:
            pw = w - 24
            ph = int(round(pw * lab_h / max(lab_w, 1)))
        x1 = (w - pw) // 2
        y1 = top + max(0, (h - top - bottom - ph) // 2)
        return x1, y1, x1 + pw, y1 + ph

    def draw_lab_panel(
        self,
        frame: np.ndarray,
        lab_image: np.ndarray,
        open_t: float,
        anchor_rect: Optional[tuple] = None,
        target_label: str = "",
        telemetry: Optional[dict] = None,
        progress: float = 0.0,
    ) -> None:
        """Composite the simulated-lab reenactment as a viewport that irises open.

        The panel grows from the target object's own position in the live frame
        out to its full size, over the dimmed and blurred camera feed - so it
        reads as the system opening a window into its own simulation of THAT
        object, rather than as a separate video cutting in.
        """
        if lab_image is None or open_t <= 0.001:
            return
        h, w = frame.shape[:2]
        tx1, ty1, tx2, ty2 = self.lab_panel_rect(frame.shape, lab_image.shape)
        e = self._ease_out_cubic(open_t)

        if anchor_rect is None:
            cx, cy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
            anchor_rect = (cx - 8, cy - 6, cx + 8, cy + 6)
        ax1, ay1, ax2, ay2 = anchor_rect

        x1 = int(round(ax1 + (tx1 - ax1) * e))
        y1 = int(round(ay1 + (ty1 - ay1) * e))
        x2 = int(round(ax2 + (tx2 - ax2) * e))
        y2 = int(round(ay2 + (ty2 - ay2) * e))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        pw, ph = x2 - x1, y2 - y1
        if pw < 8 or ph < 8:
            return

        # Recede the live feed: darken, and defocus via a downscale/upscale pair
        # (a real Gaussian at full resolution costs more than the 3-D render).
        small = cv2.resize(frame, (max(1, w // 6), max(1, h // 6)), interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(blurred, 0.55 * e, frame, 1.0 - 0.55 * e, 0, dst=frame)
        if e > 0.01:
            dark = np.zeros_like(frame)
            cv2.addWeighted(dark, 0.42 * e, frame, 1.0 - 0.42 * e, 0, dst=frame)

        radius = max(2, int(12 * e))
        mask_f, contours = self._rounded_panel_mask(pw, ph, radius)
        content = cv2.resize(lab_image, (pw, ph), interpolation=cv2.INTER_LINEAR)
        roi = frame[y1:y2, x1:x2]
        roi[:] = (content * mask_f + roi * (1.0 - mask_f)).astype(np.uint8)
        cv2.drawContours(roi, contours, -1, (72, 70, 68), 1, lineType=cv2.LINE_AA)

        if e < 0.985:
            return  # chrome would be unreadable mid-flight

        self._draw_lab_chrome(frame, (x1, y1, x2, y2), target_label, telemetry or {}, progress)

    def _draw_lab_chrome(self, frame: np.ndarray, rect: tuple, target_label: str,
                         telemetry: dict, progress: float) -> None:
        """Header and the live plan telemetry strip, in the app's own voice:
        quiet tracked eyebrows, real type, one accent - no corner brackets."""
        x1, y1, x2, y2 = rect

        # Header strip
        hh = self._px(26)
        head = frame[y1 + 1:y1 + 1 + hh, x1 + 1:x2 - 1]
        if head.size:
            cv2.addWeighted(np.full_like(head, SCRIM), 0.78, head, 0.22, 0, dst=head)
        cv2.circle(frame, (x1 + self._px(15), y1 + hh // 2), self._px(3),
                   INK["teal"], -1, cv2.LINE_AA)
        T.draw_tracked(frame, "SIMULATED LAB · REENACTMENT",
                       (x1 + self._px(26), y1 + self._px(18)), self._px(10),
                       INK["secondary"])
        label = (target_label or "object").replace("_", " ").capitalize()
        T.draw(frame, label, (x2 - self._px(12), y1 + self._px(18)), self._px(11),
               INK["label"], weight="medium", align="right")

        # Phase progress, hairline along the header's lower edge
        bar_w = int((x2 - x1 - 4) * float(np.clip(progress, 0.0, 1.0)))
        if bar_w > 0:
            cv2.line(frame, (x1 + 2, y1 + hh), (x1 + 2 + bar_w, y1 + hh),
                     INK["blue"], self._th(2), cv2.LINE_AA)

        if not telemetry:
            return

        # Footer telemetry: every number here comes from the executed plan.
        fh = self._px(34)
        fy1 = y2 - 1 - fh
        foot = frame[fy1:y2 - 1, x1 + 1:x2 - 1]
        if foot.size:
            cv2.addWeighted(np.full_like(foot, SCRIM), 0.80, foot, 0.20, 0, dst=foot)

        ty = fy1 + self._px(13)
        px = self._px(10)
        T.draw(frame, f"Step {telemetry.get('step', 0):02d}/{telemetry.get('num_steps', 0):02d}",
               (x1 + self._px(12), ty), px, INK["label"], weight="medium")
        T.draw(frame, f"T+{telemetry.get('sim_time', 0.0):.2f}s",
               (x1 + self._px(94), ty), px, INK["tertiary"])
        T.draw(frame, f"Lift {telemetry.get('lift_cm', 0.0):4.1f} cm",
               (x1 + self._px(156), ty), px, INK["orange"], weight="medium")

        # Gripper aperture bar
        gx, gy, gw = x1 + self._px(12), fy1 + self._px(20), self._px(96)
        T.draw_tracked(frame, "GRIP", (gx, gy + self._px(7)), self._px(8), INK["tertiary"])
        bx = gx + self._px(34)
        bh = self._px(5)
        UIH.progress_track(frame, (bx, gy, bx + gw, gy + bh),
                           float(np.clip(telemetry.get("gripper", 0.0), 0.0, 1.0)),
                           INK["teal"])

        # Per-fingertip contact indicators
        contact = float(np.clip(telemetry.get("contact", 0.0), 0.0, 1.0))
        cx0 = bx + gw + self._px(20)
        T.draw_tracked(frame, "CONTACT", (cx0, gy + self._px(7)), self._px(8),
                       INK["tertiary"])
        dot0 = cx0 + self._px(64)
        for i in range(5):
            on = contact > (i + 0.5) / 5.0
            cv2.circle(frame, (dot0 + i * self._px(11), gy + self._px(2)), self._px(3),
                       INK["green"] if on else INK["separator"], -1, cv2.LINE_AA)

    def _video_label(self, frame: np.ndarray, text: str, org: tuple,
                     colour: tuple = BONE, px: float = 12) -> None:
        """A caption seated on live video: SF type on a soft dark capsule.

        Chrome cards get their labels from the glass they sit on; a caption on
        footage has to bring its own backing or vanish into a bright room.
        """
        h, w = frame.shape[:2]
        size = self._px(px)
        tw = T.measure(text, size, "medium")
        pad_x, above, below = self._px(8), self._px(13), self._px(6)
        x, y = int(org[0]), int(org[1])
        x = int(np.clip(x, 2, max(2, w - tw - 2 * pad_x - 2)))
        y = int(np.clip(y, above + 2, h - below - 2))
        x1, y1 = x, y - above
        x2, y2 = x + tw + 2 * pad_x, y + below
        roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size:
            backing = np.full_like(roi, SCRIM)
            cv2.addWeighted(backing, 0.72, roi, 0.28, 0, dst=roi)
        T.draw(frame, text, (x + pad_x, y), size, colour, weight="medium")

    def draw_3d_bounding_boxes(self, frame: np.ndarray, bboxes: List[BoundingBox3D],
                               simplified: bool = False) -> None:
        """The identified object, marked the way a pro tool marks a selection:
        a hairline rectangle around its projected extent, accent ticks at the
        corners, and its name with the measured depth in real type. The old
        12-edge cyan wireframe read as a different app from the chrome.

        `simplified` drops the corner ticks while a ghost hand is grasping on
        the live view - at that moment the hand is the thing to look at.
        """
        if not self.show_bounding_box or not bboxes:
            return

        h, w = frame.shape[:2]
        for bbox in bboxes:
            corners_2d = bbox.project_to_2d(image_shape=(h, w))
            x0 = int(np.clip(corners_2d[:, 0].min(), 0, w - 1))
            x1 = int(np.clip(corners_2d[:, 0].max(), 0, w - 1))
            y0 = int(np.clip(corners_2d[:, 1].min(), 0, h - 1))
            y1 = int(np.clip(corners_2d[:, 1].max(), 0, h - 1))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue

            # Hairline outline, blended so it sits IN the footage rather than
            # on top of it.
            box_layer = frame.copy()
            cv2.rectangle(box_layer, (x0, y0), (x1, y1), BONE, self._th(1), cv2.LINE_AA)
            cv2.addWeighted(box_layer, 0.55, frame, 0.45, 0, dst=frame)

            if not simplified:
                tick = min(self._px(14), (x1 - x0) // 3, (y1 - y0) // 3)
                thick = self._th(2)
                for cx_, cy_, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                         (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    cv2.line(frame, (cx_, cy_), (cx_ + dx * tick, cy_),
                             INK["blue"], thick, cv2.LINE_AA)
                    cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy * tick),
                             INK["blue"], thick, cv2.LINE_AA)

            name = bbox.label.replace("_", " ").capitalize()
            self._video_label(frame, f"{name} · {bbox.center[2]:.2f} m",
                              (x0, y0 - self._px(10)))

    def draw_policy_corrections(self, frame: np.ndarray, foreseen_traj,
                                learned_bias: Optional[np.ndarray]) -> None:
        """The learned residual, made visible ON the plan.

        The planned wrist path is drawn as a quiet dotted line, and at a few
        points along the reach an accent arrow points from where the
        uncorrected plan would have passed to where the learned bias moved it
        - that is, toward how this user actually reaches. The plan already
        CONTAINS the bias (see generate_foreseen_rollout), so the arrows
        depict a real nudge, not an illustration of one.

        Nothing is drawn while the bias is still ~zero: an arrow on screen
        always means the network changed something.
        """
        if foreseen_traj is None or not getattr(foreseen_traj, "waypoints", None):
            return
        if learned_bias is None:
            return
        bias = np.asarray(learned_bias, dtype=np.float32).reshape(-1)[:3]
        if bias.shape[0] < 3 or float(np.linalg.norm(bias)) < 0.004:   # < 4 mm
            return

        h, w = frame.shape[:2]
        fx = 0.8 * w
        cx, cy = w / 2.0, h / 2.0

        def proj(p3: np.ndarray) -> tuple:
            z = max(float(p3[2]), 0.1)
            return (int(fx * float(p3[0]) / z + cx), int(fx * float(p3[1]) / z + cy))

        wrists = [np.asarray(wp.wrist_pose[:3], dtype=np.float32)
                  for wp in foreseen_traj.waypoints]
        pts = [proj(p) for p in wrists]

        # The plan's path: dotted, receding - context for the arrows, not a
        # second ghost.
        for i in range(0, len(pts) - 1, 2):
            cv2.line(frame, pts[i], pts[i + 1], INK["tertiary"], self._th(1),
                     cv2.LINE_AA)

        # The plan carries the full correction from the end of the approach
        # onward, so these three are all full-size arrows, spread along the
        # reach and clear of the endpoint the ghost hand occupies.
        n = len(wrists)
        samples = sorted({(2 * n) // 5, (11 * n) // 20, (7 * n) // 10})
        drawn = None
        for i in samples:
            # How much of the bias is in the plan AT waypoint i. The generator
            # folds the bias into the grasp point and everything after it, and
            # interpolates into that over the approach phase (the first 28% of
            # the rollout, minimum-jerk) - it is NOT a linear ramp to the end.
            # Scaling these arrows linearly drew them at a third of the real
            # correction over most of the path.
            t_frac = i / max(n - 1, 1)
            ramp = minimum_jerk_step(t_frac / 0.28) if t_frac < 0.28 else 1.0
            tail = proj(wrists[i] - bias * ramp)
            head = pts[i]
            if abs(head[0] - tail[0]) + abs(head[1] - tail[1]) < 6:
                continue
            cv2.arrowedLine(frame, tail, head, INK["blue"], self._th(2),
                            line_type=cv2.LINE_AA, tipLength=0.30)
            drawn = head
        if drawn is not None:
            mm = float(np.linalg.norm(bias)) * 1000.0
            # Below the arrow: the object's own name sits above the box, and
            # the two captions were landing on each other.
            self._video_label(frame, f"Learned correction · {mm:.0f} mm",
                              (drawn[0] + self._px(14), drawn[1] + self._px(26)),
                              colour=INK["blue"], px=11)

    def draw_affordance_hotspots(self, frame: np.ndarray, affordance_map: Optional[AffordanceMap]) -> None:
        """Candidate contact points: a quiet accent dot and a thin ring each.
        They support the grasp story - they don't get names shouted at them."""
        if affordance_map is None or not len(affordance_map.hotspots):
            return

        h, w = frame.shape[:2]
        fx = fy = 0.8 * w
        cx = w / 2.0
        cy = h / 2.0

        for hs in affordance_map.hotspots:
            z = max(hs[2], 0.1)
            u = int(fx * (hs[0] / z) + cx)
            v = int(fy * (hs[1] / z) + cy)
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(frame, (u, v), self._px(7), INK["blue"], self._th(1), cv2.LINE_AA)
                cv2.circle(frame, (u, v), self._px(2), INK["blue"], -1, cv2.LINE_AA)

class SyntheticCamera:
    """Generates synthetic animated RGB video frames when webcam is inaccessible."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._start_time = time.time()

    def read(self) -> tuple[bool, np.ndarray]:
        t = time.time() - self._start_time
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for y in range(self.height):
            c = int(35 + 25 * (y / self.height))
            frame[y, :] = (c, c - 5, c - 10)

        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (45, 45, 50), 1)
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (45, 45, 50), 1)

        T.draw(frame, "Synthetic video stream — no camera",
               (self.width // 2, self.height - 20), 13, (120, 120, 120),
               align="center")

        return True, frame

    def release(self) -> None:
        pass


class LocalClientRunner:
    """Orchestrates video capture, mode execution, robot hardware, checkpointing, and co-adaptation analytics."""

    def __init__(
        self,
        config: AppConfig,
        cli_mode: Optional[str] = None,
        tracker_type: Optional[str] = None,
        intent: Optional[str] = None,
        enable_profiling: bool = False,
        enable_recording: bool = False,
        server_url: Optional[str] = None,
        transcriber_type: str = "mock",
        gemini_api_key: Optional[str] = None
    ) -> None:
        self.config = config
        
        self.server_url = server_url
        if server_url:
            parsed_url = urlparse(server_url)
            if parsed_url.hostname:
                config.network.server_host = parsed_url.hostname
            if parsed_url.port:
                config.network.server_port = parsed_url.port
            cli_mode = "mock_remote"

        raw_mode = cli_mode or config.system.mode
        if raw_mode in ("mock", "mock_local"):
            self.mode = "mock_local"
        elif raw_mode in ("remote", "mock_remote"):
            self.mode = "mock_remote"
        else:
            self.mode = raw_mode

        self.intent = intent if intent is not None else config.system.intent
        self.preset_idx = 0
        self.adaptation_active = True
        self.enable_profiling = enable_profiling
        self.auto_record = enable_recording
        
        self.visualizer = LocalVisualizer(config)
        self.profiler = LatencyProfiler(window_size=100)
        self.recorder = SessionRecorder()
        
        # Phase 6, 7 & 8 components. Gemini (transcription + TTS) takes priority when
        # an API key is available, since it's confirmed higher quality than the local
        # tiny.en Whisper model and macOS 'say' - each still falls back automatically
        # to its non-Gemini counterpart internally on any network/API failure.
        self.gemini_api_key = gemini_api_key
        if gemini_api_key:
            self.transcriber: AudioTranscriberABC = GeminiTranscriber(api_key=gemini_api_key)
            logger.info("LocalClient: Using GeminiTranscriber for voice intent capture.")
        elif transcriber_type == "whisper" and importlib.util.find_spec("torch") is not None:
            # Same conflict documented below for NeuralResidualPolicy, but this
            # one is fatal rather than avoidable: faster-whisper (ctranslate2)
            # and torch each bundle an OpenMP runtime, and constructing a
            # WhisperModel in a process where torch is importable aborts the
            # interpreter outright. Degrade to the mock rather than take the
            # whole client down mid-session.
            logger.warning(
                "LocalClient: torch is installed, which crashes faster-whisper on load "
                "(conflicting OpenMP runtimes). Falling back to MockTranscriber - set "
                "GEMINI_API_KEY for real speech-to-text."
            )
            self.transcriber: AudioTranscriberABC = MockTranscriber()
        elif transcriber_type == "whisper":
            self.transcriber: AudioTranscriberABC = WhisperTranscriber()
        else:
            self.transcriber: AudioTranscriberABC = MockTranscriber()

        self.intent_parser: IntentParserABC = MockLLMIntentParser()
        # Kept only so callers with a handle on it (and the shutdown path) still
        # work; nothing in the workflow speaks any more. See below.
        self.speaker: SpeechSynthesizerABC = MockSpeaker()
        self.workflow = WorkflowController(
            foresee_steps=60, wait_user_timeout=2.0, auto_advance=True,
            # NOTHING speaks, in either mode. Spoken guidance was removed for
            # reasons that apply to every backend, not just Gemini's: it
            # arrived a second or two after the moment it described and talked
            # over someone concentrating on a reach. Local mode had kept a
            # macOS `say` speaker, so the two modes disagreed - a live local
            # session narrated itself out loud ("Go. Reach for the remote
            # control now.") while a remote one played a tone. Phase changes
            # are announced by NotificationSounds in both, below.
            speaker=None,
            voice_guidance_enabled=True
        )
        self.robot = MockRobotHardware(dof=7)
        self.checkpoint_manager = PolicyCheckpointManager()
        self.benchmark = CoAdaptationBenchmark()
        self.safety_monitor = SafetyMonitor(dof=7)
        
        self.voice_status = "IDLE"
        self._voice_status_until = 0.0
        self.current_parsed_intent = self.intent_parser.parse_intent(self.intent)
        self.last_episode_report: Optional[EpisodeDiscrepancyReport] = None
        self._control_cmd_to_send: Optional[str] = None

        self.cap = None
        self.is_synthetic_camera = False
        self._audio_stream: Optional["sd.InputStream"] = None
        self._is_fullscreen = False
        self.camera: Optional[CameraStream] = None
        self._screen_w, self._screen_h = self._detect_screen_size()
        # The composition surface. Chrome is drawn on it at its own resolution
        # rather than on the 640x480 feed, so type is rasterised at the size it
        # is shown instead of being scaled up threefold on the way to the window.
        self.motion = UIG.Motion()
        self.stage = self._build_stage()
        # Episode rewards for the learning card's trend line. The benchmark
        # summary reports aggregates only, so the series is kept here.
        self._reward_history: collections.deque = collections.deque(maxlen=48)

        selected_tracker = tracker_type or config.perception.hand_tracker.tracker_type
        self.use_mediapipe = (selected_tracker == "mediapipe") and MEDIAPIPE_AVAILABLE

        if self.use_mediapipe:
            logger.info("Initializing live MediaPipe Hand Tracker (CPU profile)...")
            self.active_tracker: HandTrackerABC = MediaPipeHandTracker(
                max_num_hands=config.perception.hand_tracker.max_hands,
                min_detection_confidence=config.perception.hand_tracker.confidence_threshold,
                model_complexity=0
            )
        else:
            logger.info("Initializing Mock Hand Tracker (Synthetic Kinematics)...")
            self.active_tracker = MockHandTracker()

        self.mock_tracker = MockHandTracker()
        self.mediapipe_tracker = MediaPipeHandTracker() if MEDIAPIPE_AVAILABLE else None

        # Local mock generators
        self.local_depth_estimator = MockDepthEstimator()
        # Real, open-vocabulary detection wherever a key allows it. The mock
        # invents objects from the intent string and never looks at the camera,
        # which is indistinguishable from working until you notice the box is
        # attached to nothing. Gemini names whatever is actually in frame -
        # "utensil holder", "houseplant" - which COCO-80 cannot.
        self.local_scene_parser = self._build_local_scene_parser(gemini_api_key)
        self.local_affordance_extractor = MockAffordanceExtractor()
        self.local_trajectory_diffusion = MockTrajectoryDiffusion()
        self.local_discrepancy_engine = DiscrepancyEngine()
        self.local_physics_engine = MockPhysicsEngine()
        # The real neural policy is used locally whenever it is SAFE to load
        # torch in this process. The hazard is specific and narrow: faster-whisper
        # (ctranslate2) and torch each bundle an OpenMP runtime, and having both
        # in one interpreter aborts it outright. That only arises when
        # WhisperTranscriber is the active backend - with Gemini transcription,
        # or the mock, faster-whisper is never imported and torch is fine.
        #
        # This matters beyond tidiness: it is what makes the laptop-only path a
        # real fallback that actually learns, rather than a demo of a stub, for
        # the times the GPU pod is unavailable.
        self.local_object_detector = getattr(self, "local_object_detector", None)
        self.local_policy = self._build_local_policy()
        # How the spoken verb should be carried out. The rules answer instantly;
        # the language model refines the reading a second or two later, off the
        # render loop, and the refined plan is cached against the utterance.
        self.action_parser = (GeminiActionParser(api_key=gemini_api_key)
                              if gemini_api_key else None)
        self._action_plan = ActionPlan()
        self._last_action = np.zeros(7, dtype=np.float32)
        self._cached_foreseen_traj = None
        self._local_learned_wrist_bias = np.zeros(3, dtype=np.float32)
        self._local_adaptation_computed_this_episode = False

        # Network client
        self.ws_client = WSStreamingClient(
            host=config.network.server_host,
            port=config.network.server_port,
            server_url=server_url,
            compression_quality=config.network.compression_quality,
            timeout=config.network.timeout_seconds
        )

        # Decoupled network state: the render/input loop never blocks on the WS round-trip.
        # A background task streams frames and updates this snapshot whenever a fresh
        # response lands; the render loop always draws the latest known snapshot instead.
        self._network_inflight = False
        self._network_task: Optional[asyncio.Task] = None
        self._network_latency_ms = 0.0
        self._network_got_first_response = False
        # Calming tones in place of spoken guidance. See notification_sounds
        # for why speech was the wrong instrument here.
        self.sounds = NotificationSounds()
        self._last_announced_phase: Optional[ExecutionPhase] = None
        self._training_target_announced: Optional[str] = None

        # Client-side real-motion afterimage recording. The ghost hand is now a
        # replay of the user's OWN recorded hand poses, never a synthetic plan -
        # recorded locally (independent of the server's internal recording, which
        # isn't sent back over the wire) so it works the same in mock_local and
        # mock_remote. `_last_completed_recording` persists across the RESTARTING
        # loop until the NEXT execution finishes, so it can serve both as this
        # attempt's post-execution review AND the next attempt's FORESEEING preview.
        self._local_recorded_poses: List[HandPose] = []
        self._last_completed_recording: List[HandPose] = []
        self._last_recording_phase: Optional[ExecutionPhase] = None
        self._last_recording_target: Optional[str] = None
        # A real photo-crop of the target object, refreshed whenever it's plainly
        # visible (not mid-grasp), so the ghost afterimage can look like the
        # actual object instead of an abstract colored shape.
        self._object_sprite: Optional[np.ndarray] = None
        # A short-lived line on the status bar, for the moments the app has
        # to decline something the user just asked for.
        self._notice: Optional[str] = None
        self._notice_until = 0.0

        # Simulated-lab reenactment for the Autonomous Demo. The plan is staged
        # and rendered in a 3-D lab (src/simulation/lab_sim.py) rather than drawn
        # as a flat overlay on the webcam image; `_lab_open` drives the iris that
        # opens and closes the viewport.
        # Registered: the reenactment is staged in the real camera, over the
        # live frame, at the object's own detected position and extent.
        self.lab_sim = LabSimulator(registered=True)
        self._lab_staged = False
        self._lab_open = 0.0
        self._lab_image: Optional[np.ndarray] = None
        self._lab_anchor_rect: Optional[tuple] = None
        self._lab_last_t = time.perf_counter()
        # Wall-clock start of the running demo. The reenactment is played from
        # this rather than from the server's reported phase progress: that value
        # only refreshes when a response lands (about once a second on a CPU-only
        # host), so animating from it held each pose for a beat and then jumped,
        # however fast this client was actually drawing.
        self._lab_demo_started_at: Optional[float] = None
        # Mesh authoring runs off the render thread: the call takes 1-3 s and the
        # loop cannot stall on it. The canonical class shape is used until an
        # authored profile lands, and the next staging of that object picks it up.
        self._mesh_author = (GeminiMeshAuthor(api_key=gemini_api_key)
                             if gemini_api_key else None)
        self._mesh_author_busy: set = set()
        self._remote_snapshot = {
            "poses": [], "bboxes": [], "affordance_map": None, "foreseen_traj": None,
            "depth_heatmap": None, "gripper_cmd": 0.0, "residuals": None, "reward_score": 0.0,
            "discrepancy_norm": 0.0, "buffer_steps": 0, "parsed_intent": None,
            "workflow_phase": ExecutionPhase.IDLE, "phase_progress": 0.0,
            "benchmark_summary": None, "policy_loss": 0.0, "policy_updates": 0,
            "learned_wrist_bias": None, "depth_raw": None,
        }
        # Metric depth for the scene reconstruction. The HUD's heatmap is a
        # colourmapped picture and cannot be inverted back to metres.
        self._latest_depth_m: Optional[np.ndarray] = None
        # The learning card's training heartbeat. The count is read from
        # whichever policy is really learning (local or the server's); the
        # pulse timestamp marks the instant it last incremented, so each RWR
        # gradient step is a visible flash rather than a silent number change.
        self._policy_updates_seen = 0
        self._policy_update_pulse_at = 0.0

    def toggle_voice_mode(self) -> None:
        """Toggle Push-To-Talk voice listening / transcription."""
        if not self.transcriber.is_listening:
            self.transcriber.start_listening()
            self.voice_status = "LISTENING"
            self.sounds.play("listening")
            logger.info("Voice Mode: LISTENING... Speak your intent.")
        else:
            self.voice_status = "TRANSCRIBING"
            transcript = self.transcriber.stop_listening()
            if not transcript:
                # The transcriber declined to guess. Say so on the HUD rather
                # than silently leaving the old target in place.
                logger.warning("Voice Mode: nothing transcribed - target unchanged.")
                self.sounds.play("attention")
                self.voice_status = "FAILED"
                self._voice_status_until = time.time() + 4.0
                return
            if transcript:
                self.intent = transcript
                self.current_parsed_intent = self.intent_parser.parse_intent(transcript)
                self.workflow.trigger_intent(self.current_parsed_intent.target_object if self.current_parsed_intent.is_active else "none")
                self.sounds.play("heard")
                logger.info(f"Voice Mode Transcribed: '{transcript}' -> Target: {self.current_parsed_intent.target_object}")
            self.voice_status = "IDLE"

    def _build_local_scene_parser(self, gemini_api_key: Optional[str]):
        """Open-vocabulary detection locally, or the mock when there is no key."""
        if not gemini_api_key:
            logger.warning(
                "LocalClient: no GEMINI_API_KEY, so objects are SYNTHETIC - "
                "MockSceneParser invents them from the intent text and never "
                "looks at the camera."
            )
            return MockSceneParser()
        try:
            from src.perception.gemini_scene_detector import GeminiObjectDetector
            from src.perception.live_scene_parser import LiveSceneParser
            self.local_object_detector = GeminiObjectDetector(
                api_key=gemini_api_key, cadence_sec=1.5)
            logger.info(
                "LocalClient: Using Gemini open-vocabulary detection "
                "(refreshed every ~1.5 s, tracked between calls)."
            )
            return LiveSceneParser(object_detector=self.local_object_detector)
        except Exception as e:
            logger.warning(f"LocalClient: Gemini detection unavailable ({e}); "
                           "objects will be SYNTHETIC.")
            return MockSceneParser()

    def _build_local_policy(self):
        """The real residual policy when torch can safely be loaded here."""
        if isinstance(self.transcriber, WhisperTranscriber):
            logger.info(
                "LocalClient: WhisperTranscriber is active, so torch cannot be loaded "
                "in this process (conflicting OpenMP runtimes). Using MockResidualPolicy, "
                "which DOES NOT LEARN - set GEMINI_API_KEY to enable real local adaptation."
            )
            return MockResidualPolicy()
        try:
            from src.policy.neural_policy import NeuralResidualPolicy
            policy = NeuralResidualPolicy()
            logger.info(
                f"LocalClient: Using real NeuralResidualPolicy (device={policy.device}), "
                "trained online via Reward-Weighted Regression."
            )
            return policy
        except Exception as e:
            logger.warning(
                f"LocalClient: NeuralResidualPolicy unavailable ({e}); using "
                "MockResidualPolicy, which DOES NOT LEARN."
            )
            return MockResidualPolicy()

    def _record_trial_if_scorable(self, rep, intent=None) -> bool:
        """Record a completed trial, unless there was nothing to score.

        With no object detected there is no plan, so the comparison comes back
        as zeros. Recording that as a trial reports PERFECT accuracy for an
        attempt that was never measured - it flatters the error curve on screen
        and feeds a fabricated reward into the policy's history.
        """
        if not rep.is_scorable:
            logger.warning(
                "Episode not scored: no foreseen plan to compare against "
                f"({rep.num_steps_real} tracked frames, {rep.num_steps_sim} planned). "
                "Usually means no object was detected - check the target is in view."
            )
            return False
        self.benchmark.record_trial(rep, intent=intent if intent is not None else self.intent)
        return True

    def _refresh_action_plan(self) -> ActionPlan:
        """The current reading of what the user said they would do."""
        said = self._spoken_intent()
        if not said:
            self._action_plan = ActionPlan()
            return self._action_plan
        if self.action_parser is not None:
            self._action_plan = self.action_parser.plan_async(said)
        else:
            self._action_plan = plan_from_text(said)
        return self._action_plan

    def _spoken_intent(self) -> Optional[str]:
        """The last utterance, or None if it is a placeholder rather than speech."""
        said = (self.intent or "").strip()
        if not said or said.lower() in ("idle", "none", "standby"):
            return None
        return said

    def _current_voice_status(self) -> str:
        """Voice status for display, expiring any transient notice."""
        if self._voice_status_until and time.time() > self._voice_status_until:
            self.voice_status = "IDLE"
            self._voice_status_until = 0.0
        return self.voice_status

    # Which cue marks arriving in each phase.
    _PHASE_CUES = {
        ExecutionPhase.FORESEEING: "ready",
        ExecutionPhase.WAIT_USER: "ready",
        ExecutionPhase.USER_EXECUTING: "go",
        ExecutionPhase.ADAPTING: "complete",
        ExecutionPhase.AUTONOMOUS_DEMO: "improved",
    }

    def _sound_phase_change(self, phase, previous) -> None:
        """Sound the cue for arriving in `phase`."""
        if not self.workflow.voice_guidance_enabled:
            return
        if phase == ExecutionPhase.IDLE:
            # Returning to standby is only worth a sound if something was
            # actually completed, not every time the workflow unwinds.
            self._training_target_announced = None
            if previous == ExecutionPhase.ADAPTING:
                self.sounds.play("complete")
            return
        cue = self._PHASE_CUES.get(phase)
        if cue:
            self.sounds.play(cue)

    def toggle_voice_guidance(self) -> None:
        """Toggle the notification cues ('g')."""
        self.workflow.voice_guidance_enabled = not self.workflow.voice_guidance_enabled
        self.sounds.enabled = self.workflow.voice_guidance_enabled
        if not self.workflow.voice_guidance_enabled:
            self.sounds.stop()
        logger.info(f"Notification sounds: {'ON' if self.sounds.enabled else 'MUTED'}")

    def cycle_intent(self) -> None:
        """Cycle through preset natural language intent prompts."""
        self.preset_idx = (self.preset_idx + 1) % len(PRESET_INTENTS)
        self.intent = PRESET_INTENTS[self.preset_idx]
        self.current_parsed_intent = self.intent_parser.parse_intent(self.intent)
        self.workflow.trigger_intent(self.current_parsed_intent.target_object if self.current_parsed_intent.is_active else "none")
        logger.info(f"Switched user intent prompt to: '{self.intent}'")

    def toggle_adaptation(self) -> None:
        """Toggle online residual adaptation loop."""
        self.adaptation_active = not self.adaptation_active
        self.local_policy.adaptation_active = self.adaptation_active
        status = "ACTIVE" if self.adaptation_active else "PAUSED"
        logger.info(f"Online Residual Adaptation: {status}")

    def advance_workflow_phase(self) -> None:
        """Advance to subsequent workflow state."""
        new_phase = self.workflow.advance_phase()
        if self.mode == "mock_remote":
            self._control_cmd_to_send = "ADVANCE_PHASE"
        logger.info(f"Manually advanced workflow to: [{new_phase.value}]")

    def _notify(self, text: str, seconds: float = 4.0) -> None:
        """Say something to the user on the status bar, briefly."""
        self._notice = text
        self._notice_until = time.time() + float(seconds)
        self.sounds.play("attention")

    def _current_notice(self) -> Optional[str]:
        if self._notice and time.time() < self._notice_until:
            return self._notice
        self._notice = None
        return None

    def trigger_autonomous_demo(self) -> None:
        """On-demand reenactment of the user's OWN recorded attempt (the 'a'
        hotkey), staged in the lab and run through the trained residual policy.

        It requires an active intent AND a real recording. It used to fall back
        to a generated plan when there was no recording - silently, and while
        still captioned as a reenactment, so a canned approach that was never
        the user's was presented as a replay of them. Refusing and saying why is
        the only honest option: the demo's whole claim is that it is showing
        THEIR movement.
        """
        if self.workflow._target_label in ("none", "", "idle", "clear"):
            logger.info("Autonomous Demo: no active intent - say what to pick up first.")
            self._notify("Say what to pick up first — hold 'v'")
            return
        have = len(self._last_completed_recording)
        if have < LAB_MIN_DEMONSTRATION_POSES:
            logger.info(
                f"Autonomous Demo: only {have} recorded frames "
                f"(need {LAB_MIN_DEMONSTRATION_POSES}) - take a turn first.")
            self._notify("Take a turn first — press 'c' and reach for it")
            return
        self.lab_sim.invalidate()
        self._lab_staged = False
        self.workflow.handle_control_command("START_AUTONOMOUS_DEMO")
        if self.mode == "mock_remote":
            self._control_cmd_to_send = "START_AUTONOMOUS_DEMO"
        logger.info(f"Autonomous Demo triggered for target: {self.workflow._target_label}")

    def _request_authored_mesh(self, label: str, sprite: Optional[np.ndarray]) -> None:
        """Kick off Gemini profile authoring for `label`, at most once per object."""
        if (self._mesh_author is None or sprite is None or not label
                or label in self._mesh_author_busy
                or OL.authored_profile(label) is not None
                or not OL.is_turned(label)):
            # Flat classes are boxes, not turned shapes - authoring a lathe
            # profile for one buys a rod and an API call.
            return
        self._mesh_author_busy.add(label)
        crop = sprite.copy()

        def _work() -> None:
            try:
                profile = self._mesh_author.author(crop, label)
                if profile:
                    OL.remember_authored_profile(label, profile)
            except Exception as exc:                       # best-effort by design
                logger.warning(f"Mesh authoring failed for '{label}': {exc}")
            finally:
                self._mesh_author_busy.discard(label)

        threading.Thread(target=_work, name=f"mesh-author:{label}", daemon=True).start()

    def _to_display_resolution(self, frame, poses):
        """Enlarge the feed to its card size and rescale 2-D landmarks to match.

        Returns the enlarged frame and a SEPARATE list of display poses. The
        originals are left alone: they are still being recorded, scored and
        replayed in sensor pixels, and rescaling them in place would silently
        corrupt the episode the user is performing - it happens to be safe today
        only because the recorder reads them a few lines earlier.
        """
        L = self.stage.layout
        target_w, target_h = L.video[2] - L.video[0], L.video[3] - L.video[1]
        src_h, src_w = frame.shape[:2]
        if target_w <= 0 or target_h <= 0 or (target_w, target_h) == (src_w, src_h):
            self.visualizer.draw_scale = 1.0
            return frame, list(poses or [])

        sx, sy = target_w / float(src_w), target_h / float(src_h)
        # One factor drives type and stroke weight; the axes differ only by the
        # rounding of an aspect-preserving fit, so their mean is exact enough.
        self.visualizer.draw_scale = float((sx + sy) * 0.5)

        scale = np.array([sx, sy], dtype=np.float32)
        display_poses = [replace(p, keypoints_2d=(p.keypoints_2d * scale).astype(np.float32))
                         for p in (poses or [])]
        return cv2.resize(frame, (target_w, target_h),
                          interpolation=cv2.INTER_LINEAR), display_poses

    def _scale_poses_for_display(self, poses):
        """Display copies of `poses` at the current annotation scale."""
        if not poses:
            return poses
        k = self.visualizer.draw_scale
        if abs(k - 1.0) < 1e-3:
            return poses
        scale = np.array([k, k], dtype=np.float32)
        return [replace(p, keypoints_2d=(p.keypoints_2d * scale).astype(np.float32))
                for p in poses]

    def _update_lab_panel(
        self,
        frame: np.ndarray,
        phase: ExecutionPhase,
        progress: float,
        foreseen_traj,
        bboxes: List[BoundingBox3D],
    ) -> None:
        """Drive and composite the simulated-lab viewport.

        Staging happens once, the first frame of the demo, from the plan the
        server (or the local mock) just generated for the object's CURRENT
        position - so the reenactment is of this attempt, not a canned animation.
        The iris then opens over ~0.45 s, holds for the phase, and closes again,
        which is why the open/close fraction is driven by wall-clock delta rather
        than by frame count: the client's frame rate varies with what the
        perception stack is doing.
        """
        now = time.perf_counter()
        dt = min(max(now - self._lab_last_t, 0.0), 0.25)
        self._lab_last_t = now
        active = phase == ExecutionPhase.AUTONOMOUS_DEMO

        if active and not self._lab_staged:
            target_bbox = bboxes[0] if bboxes else None
            self.lab_sim.demo_duration_sec = self.workflow.autonomous_demo_duration_sec
            if target_bbox is not None:
                self._request_authored_mesh(target_bbox.label, self._object_sprite)

            # Prefer the user's OWN recorded motion. The plan is a fallback for
            # the first attempt at an object, before there is a demonstration to
            # replay - it is a generated approach, not something they did.
            staged = False
            if self._last_completed_recording:
                staged = self.lab_sim.prepare_from_demonstration(
                    self._last_completed_recording, target_bbox, self._object_sprite)
            if not staged:
                staged = self.lab_sim.prepare(foreseen_traj, target_bbox, self._object_sprite)
            if staged:
                self._lab_staged = True
                # Authored ground at the real surface, rather than a mesh
                # reconstructed from depth: a single webcam's depth is too soft
                # to carry a room, and a desk comes back as a relief.
                self.lab_sim.set_stylised_room()
                # Back-date the local clock onto the SERVER's phase clock. Starting
                # it at the moment of staging left the playback trailing the phase
                # by however long detection and staging took, so the panel closed
                # while the animation still had that much left to run.
                span = max(self.workflow.autonomous_demo_duration_sec, 0.1)
                self._lab_demo_started_at = now - float(np.clip(progress, 0.0, 1.0)) * span
                if target_bbox is not None:
                    h, w = frame.shape[:2]
                    centre, px_w, px_h = self.visualizer._bbox_screen_rect(target_bbox, w, h)
                    self._lab_anchor_rect = (
                        int(centre[0] - px_w / 2), int(centre[1] - px_h / 2),
                        int(centre[0] + px_w / 2), int(centre[1] + px_h / 2),
                    )
                else:
                    self._lab_anchor_rect = None

        # Open in ~0.6 s, close in ~0.4 s. Longer than feels necessary on paper:
        # at the frame rates this client actually achieves, a 0.45 s open was
        # only a handful of frames and read as a snap rather than a movement.
        rate = (1.0 / 0.60) if (active and self._lab_staged) else -(1.0 / 0.40)
        self._lab_open = float(np.clip(self._lab_open + rate * dt, 0.0, 1.0))

        if self._lab_open <= 0.001:
            if self._lab_image is not None and not active:
                self._lab_image = None
                self._lab_staged = False
                self._lab_demo_started_at = None
                self.lab_sim.invalidate()
            return

        if self._lab_staged and self.lab_sim.is_ready:
            # Play from the local clock, falling back to the server's progress
            # only if staging somehow happened without a start time. The server's
            # value is the authority on when the phase ENDS; it is a poor source
            # for how far through the animation is, because it arrives in steps.
            if self._lab_demo_started_at is not None:
                span = max(self.workflow.autonomous_demo_duration_sec, 0.1)
                # Purely the local clock. Taking max() with the server's progress
                # to avoid lagging it defeated the entire point: the server starts
                # its phase timer when the plan is generated, which is ~0.8 s
                # before this client finishes staging, so its value is always the
                # larger one and the max() picked the stepped source every time.
                # A small lag behind the server is invisible - the plan finishes
                # at 82% of the phase and holds - whereas the stepping is not.
                play = float(np.clip((now - self._lab_demo_started_at) / span, 0.0, 1.0))
            else:
                play = progress
            step = self.lab_sim.step_for_progress(play)
            rendered = self.lab_sim.render(step, elapsed=now, push_in=play,
                                           background=frame)
            if rendered is not None:
                self._lab_image = rendered
            telemetry = self.lab_sim.telemetry(step)
            label = self.lab_sim.target_label
        else:
            telemetry, label = {}, ""

        if self.lab_sim.registered:
            # Registered: the render already IS this frame, with the object and
            # hand standing in it. So it cross-fades in place rather than
            # opening as a panel - a panel would rescale the image into a
            # smaller rect and slide everything off the pixels it was
            # registered to, which is the one thing this mode exists to avoid.
            self._blend_registered_lab(frame, label, telemetry, progress)
            return

        self.visualizer.draw_lab_panel(
            frame, self._lab_image, self._lab_open,
            anchor_rect=self._lab_anchor_rect, target_label=label,
            telemetry=telemetry, progress=progress,
        )

    def _blend_registered_lab(self, frame, label, telemetry, progress) -> None:
        """Dissolve the registered reenactment into the live frame, in place."""
        if self._lab_image is None or self._lab_open <= 0.001:
            return
        img = self._lab_image
        if img.shape[:2] != frame.shape[:2]:
            img = cv2.resize(img, (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        a = float(np.clip(self._lab_open, 0.0, 1.0))
        cv2.addWeighted(img, a, frame, 1.0 - a, 0.0, dst=frame)
        if self._lab_open > 0.985:
            h, w = frame.shape[:2]
            self.visualizer._draw_lab_chrome(frame, (0, 0, w - 1, h - 1),
                                             label, telemetry or {}, progress)

    def save_checkpoint(self) -> None:
        """Save learned residual policy checkpoint."""
        if self.mode == "mock_local":
            self.checkpoint_manager.save_checkpoint(self.local_policy)
        else:
            self._control_cmd_to_send = "SAVE_CHECKPOINT"
        logger.info("Saved policy adaptation checkpoint.")

    def load_checkpoint(self) -> None:
        """Load saved residual policy checkpoint."""
        if self.mode == "mock_local":
            self.checkpoint_manager.load_checkpoint(self.local_policy)
        else:
            self._control_cmd_to_send = "LOAD_CHECKPOINT"
        logger.info("Loaded policy adaptation checkpoint.")

    def reset_baseline(self) -> None:
        """Reset policy adaptation weights to zero baseline."""
        if self.mode == "mock_local":
            self.checkpoint_manager.reset_to_baseline(self.local_policy)
        else:
            self._control_cmd_to_send = "RESET_BASELINE"
        logger.info("Reset residual policy weights to baseline zero.")

    def toggle_recording(self, width: int = 640, height: int = 480) -> None:
        """Toggle session recording on / off."""
        if self.recorder.is_recording:
            saved_path = self.recorder.stop_recording()
            logger.info(f"Session recording stopped. Saved to {saved_path}")
        else:
            saved_path = self.recorder.start_recording(width=width, height=height, fps=self.config.camera.fps)
            logger.info(f"Session recording started at: {saved_path}")

    @staticmethod
    def _detect_screen_size() -> tuple:
        """Best-effort physical display resolution for fake-fullscreen mode, via
        tkinter (bundled with the python.org macOS installer this project already
        requires). Falls back to a common 1080p size if unavailable."""
        try:
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            return 1920, 1080

    @staticmethod
    def _solve_axis(p1: int, r1: int, p2: int, r2: int, target: int):
        """The request that reports `target`, from two (request, reported)
        probes. None when the axis does not respond, so the caller leaves it
        alone rather than moving the window somewhere arbitrary."""
        if p2 == p1 or r2 == r1:
            return None
        slope = (r2 - r1) / (p2 - p1)
        if abs(slope) < 1e-6:
            return None
        return int(round(p1 + (target - r1) / slope))

    def _solve_window_position(self, window_name, x, y):
        """Move the window so its CONTENT lands at (x, y). Returns the final
        rect, or None if the backend would not cooperate."""
        probes = []
        for delta in (0, 100):
            try:
                cv2.moveWindow(window_name, x + delta, y + delta)
                cv2.waitKey(1)
                probes.append((x + delta, y + delta,
                               *cv2.getWindowImageRect(window_name)[:2]))
            except (cv2.error, AttributeError):
                return None
        (px1, py1, rx1, ry1), (px2, py2, rx2, ry2) = probes
        want_x = self._solve_axis(px1, rx1, px2, rx2, x)
        want_y = self._solve_axis(py1, ry1, py2, ry2, y)
        if want_x is None or want_y is None:
            return None
        try:
            cv2.moveWindow(window_name, want_x, want_y)
            cv2.waitKey(1)
            return cv2.getWindowImageRect(window_name)
        except (cv2.error, AttributeError):
            return None

    def _place_window_on_screen(self) -> None:
        """Move the window to a known-visible origin and verify it landed.

        The check is not decoration: the failure this guards against is
        invisible from inside the process - the app runs perfectly, and the
        user sees nothing at all - so the one cheap signal that it happened
        belongs in the log rather than in someone's bug report.
        """
        window_name = self.config.visualization.window_name
        x = max(0, (self._screen_w - self.stage.width) // 2)
        # Below the menu bar, not centred vertically: the title bar is drawn
        # ABOVE the content origin, so a centred window sits low.
        y = 40
        try:
            # A window that has never been shown has no geometry: moveWindow
            # does nothing and getWindowImageRect reports zeros. Realising it
            # with one frame first is what makes the placement take effect -
            # and it puts the stage on screen immediately rather than after
            # the first camera frame arrives.
            cv2.imshow(window_name, self.stage.canvas)
            cv2.waitKey(1)
            cv2.moveWindow(window_name, x, y)
            cv2.waitKey(1)
        except cv2.error as exc:
            logger.warning(f"Could not position the window: {exc}")
            return
        try:
            rx, ry, rw, rh = cv2.getWindowImageRect(window_name)
        except (cv2.error, AttributeError):
            return                       # not all backends report geometry
        if rw <= 0 or rh <= 0:
            return

        # The coordinate moveWindow accepts is not the one getWindowImageRect
        # reports. On this Cocoa backend the two are related by y_reported =
        # 32 - 2*y_requested: a flipped origin and a Retina factor. Feeding
        # back the raw error therefore DOUBLES it - asking for 40 gave -48,
        # and "correcting" to 128 gave -224.
        #
        # So calibrate instead of guessing: two probes give the line, and the
        # request that lands the window where we want it follows directly.
        # Nothing here is display-specific; a backend where the mapping is the
        # identity solves to the identity.
        if (rx, ry) != (x, y):
            placed = self._solve_window_position(window_name, x, y)
            if placed is not None:
                rx, ry, rw, rh = placed

        visible_h = min(ry + rh, self._screen_h) - max(ry, 0)
        if visible_h < 0.6 * rh:
            logger.warning(
                f"The window is mostly off-screen (at y={ry}, {rh} px tall, on "
                f"a {self._screen_h} px display). Move it into view, or run "
                f"with a smaller window.")
        else:
            logger.info(f"Window placed at ({rx}, {ry}) at {rw}x{rh} on a "
                        f"{self._screen_w}x{self._screen_h} display.")

    def toggle_fullscreen(self) -> None:
        """Toggle the visualizer window between its native size and a maximized
        'fake fullscreen' that fills the screen.

        Deliberately NOT using cv2.WND_PROP_FULLSCREEN: on macOS's Cocoa HighGUI
        backend that property transition (a) stretches the 640x480 4:3 camera
        frame to the display's own aspect ratio using HighGUI's flat grey
        letterbox fill - not this app's theme, and easily a third of a widescreen
        display - and (b) has been observed to drop keyboard focus, so
        cv2.waitKey() stops receiving most hotkeys until the window is clicked
        back into focus. Resizing/moving a WINDOW_NORMAL window to the screen's
        dimensions gets the same "fills the screen" effect while keeping
        keyboard capture reliable; letterboxing is instead done ourselves (see
        the stage composition) with the backdrop rather than a flat fill."""
        window_name = self.config.visualization.window_name
        self._is_fullscreen = not self._is_fullscreen
        if self._is_fullscreen:
            cv2.resizeWindow(window_name, self._screen_w, self._screen_h)
            cv2.moveWindow(window_name, 0, 0)
        else:
            cv2.resizeWindow(window_name, self.stage.width, self.stage.height)
            cv2.moveWindow(window_name, 60, 60)
        logger.info(f"Visualizer window: {'FULLSCREEN' if self._is_fullscreen else 'windowed'}")

    STAGE_HOTKEYS = [
        ("v", "talk"), ("a", "auto demo"), ("c", "step"), ("r", "record"),
        ("p", "adapt"), ("f", "ghost"), ("m", "stats"), ("g", "sound"),
        ("l", "load"), ("x", "reset"), ("k", "save"), ("q", "quit"),
    ]

    # Composing at the full width of a 3440-pixel display costs more than the
    # visible gain: the stage is built at most this wide and the window scales
    # it, which still rasterises text far larger than the old 640-wide path did.
    _MAX_STAGE_W = 1920

    # A window is not the same size as the screen it has to fit on. The menu
    # bar, the title bar and the Dock all take vertical space, and a window
    # sized to the full screen height does not fit: macOS pushed it down until
    # only a ~100 px sliver of a 900 px window was on a 900 px display, which
    # reads exactly like the app never opened. Measured on a 1440x900 Mac:
    # the window's content origin came back at y=794.
    _SCREEN_CHROME_H = 130
    _SCREEN_CHROME_W = 80

    def _build_stage(self) -> Stage:
        """Size the stage to fit the USABLE screen area, capped for cost."""
        sw, sh = max(640, self._screen_w), max(480, self._screen_h)
        avail_w = max(640, sw - self._SCREEN_CHROME_W)
        avail_h = max(400, sh - self._SCREEN_CHROME_H)
        width = min(avail_w, self._MAX_STAGE_W)
        height = max(360, int(round(width * avail_h / avail_w)))
        probe = Stage(width, height)
        rail_w = probe.layout.telemetry[2] - probe.layout.telemetry[0]
        return Stage(width, height,
                     telemetry_h=UIH.telemetry_height(probe.layout.scale),
                     learning_h=UIH.learning_height(probe.layout.scale),
                     depth_h=UIH.depth_height(probe.layout.scale, rail_w))

    def _training_pulse(self) -> tuple:
        """The policy's cumulative RWR gradient-step count, and a 1 -> 0 flash
        that starts the instant the count last rose.

        The count comes from whichever policy is actually training: the local
        one in mock_local, the server's (carried on the wire) in remote mode.
        """
        if self.mode == "mock_local":
            updates = int(getattr(self.local_policy, "cumulative_adaptations", 0))
        else:
            updates = int(self._remote_snapshot.get("policy_updates", 0))
        if updates > self._policy_updates_seen:
            self._policy_updates_seen = updates
            self._policy_update_pulse_at = time.time()
        elif updates < self._policy_updates_seen:   # policy reset ('x')
            self._policy_updates_seen = updates
        pulse = max(0.0, 1.0 - (time.time() - self._policy_update_pulse_at) / 0.9) \
            if self._policy_update_pulse_at else 0.0
        return updates, pulse

    def _compose_stage(self, frame, *, fps, latency_ms, workflow_phase, phase_progress,
                       parsed_intent, depth_heatmap, poses, gripper_cmd, reward_score,
                       discrepancy_norm, policy_loss, benchmark_summary,
                       target_label, has_replay) -> np.ndarray:
        """Lay the annotated feed on the stage and draw the chrome around it."""
        stage, L = self.stage, self.stage.layout
        s = L.scale
        self.motion.tick()

        stage.compose_backdrop(frame)
        stage.place_video(frame)

        phase_value = workflow_phase.value
        colour = UIH.phase_colour(phase_value)
        title, body = self._status_message(workflow_phase, target_label, has_replay,
                                           progress=phase_progress, tracking=bool(poses))
        notice = self._current_notice()
        if notice:
            # A declined request outranks the standing instruction: the user
            # just pressed a key and needs to know why nothing happened.
            title, body = "CAN'T DO THAT YET", notice
            colour = UIH.C["orange"]
        progress = None if workflow_phase == ExecutionPhase.IDLE else phase_progress

        UIH.draw_telemetry_card(
            stage, L.telemetry, self.motion, fps=fps, latency_ms=latency_ms,
            phase_value=phase_value,
            target=(parsed_intent.target_object if parsed_intent and parsed_intent.is_active
                    else None),
            voice_status=self._current_voice_status(),
            adaptation_active=self.adaptation_active, reward=reward_score,
            error=discrepancy_norm, loss=policy_loss, gripper=gripper_cmd,
            robot_connected=self.robot.is_connected,
            hand_conf=(poses[0].confidence if poses else None),
            is_recording=self.recorder.is_recording,
            recorded_frames=self.recorder.frame_count, scale=s,
            # What the user actually said, and whether it is reaching the policy
            # as an embedding rather than merely having been heard.
            utterance=self._spoken_intent(),
            intent_conditioned=bool(self.gemini_api_key),
            action=self._action_plan.summary if self._spoken_intent() else None)

        UIH.draw_depth_card(stage, L.depth, depth_heatmap, s)
        UIH.draw_hotkey_card(stage, L.hotkeys, self.STAGE_HOTKEYS, s)
        if L.learning:
            summary = benchmark_summary or {}
            updates, pulse = self._training_pulse()
            UIH.draw_learning_card(
                stage, L.learning, trials=int(summary.get("total_trials", 0)),
                init_err_mm=float(summary.get("initial_error_mm", 0.0)),
                cur_err_mm=float(summary.get("latest_error_mm", 0.0)),
                rewards=list(self._reward_history), scale=s,
                updates=updates, pulse=pulse)

        UIH.draw_status_bar(stage, L.status, self.motion, title=title, body=body,
                            colour=colour, progress=progress, scale=s)
        UIH.draw_banner(stage.canvas, L.video, self.motion,
                        title=UIH.phase_label(phase_value).title(),
                        colour=colour, progress=progress, scale=s)
        return stage.canvas

    # Tracked frames the execution phase needs before it will advance. Mirrors
    # WorkflowController.foresee_steps; shown to the user because the phase ends
    # on frames CAPTURED, not on elapsed time.
    _EXECUTION_FRAMES = 60

    @staticmethod
    def _status_message(phase: ExecutionPhase, target_label: str, has_replay: bool,
                        progress: float = 0.0, tracking: bool = True):
        """Plain language for what to do right now.

        The execution phase advances on the number of frames in which a hand was
        actually DETECTED, not on elapsed time, so it stalls indefinitely and
        silently if tracking drops. That has ended three separate live sessions
        before an episode ever completed. The count and the reason for a stall
        are now on screen.
        """
        target = (target_label.replace("_", " ")
                  if target_label and target_label.lower() not in ("none", "") else "an object")
        foreseeing = (f"Watch a replay of your last attempt at the {target}, or press 'a'"
                      if has_replay else f"First try - go with your best guess for the {target}")
        return {
            ExecutionPhase.IDLE: ("STANDBY", "Hold 'v' and say what to pick up, e.g. \"wine glass\""),
            ExecutionPhase.FORESEEING: ("PREVIEWING", foreseeing),
            ExecutionPhase.WAIT_USER: ("YOUR TURN", "Get in position and press 'c' - or 'a' for the demo"),
            ExecutionPhase.USER_EXECUTING: (
                "GO",
                (f"Reach for the {target} - keep your hand in view "
                 f"({int(progress * LocalClientRunner._EXECUTION_FRAMES)}"
                 f"/{LocalClientRunner._EXECUTION_FRAMES} frames)")
                if tracking else
                "Raise your hand into view - nothing is being captured"),
            ExecutionPhase.ADAPTING: ("REVIEW", "Here's a replay of what you just did"),
            ExecutionPhase.RESTARTING: ("TRY AGAIN", f"Restarting with an improved plan for the {target}"),
            ExecutionPhase.AUTONOMOUS_DEMO: ("AUTONOMOUS DEMO",
                                             f"Simulating the grasp on the {target}"),
        }.get(phase, ("STANDBY", "Hold 'v' and say what to pick up"))

    def toggle_tracker(self) -> None:
        """Toggle between MediaPipe live tracker and synthetic mock."""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe is not installed; cannot toggle to live tracker.")
            return

        if isinstance(self.active_tracker, MediaPipeHandTracker):
            self.active_tracker = self.mock_tracker
            logger.info("Switched hand tracker to: MOCK (SYNTHETIC)")
        else:
            if not self.mediapipe_tracker:
                self.mediapipe_tracker = MediaPipeHandTracker()
            self.active_tracker = self.mediapipe_tracker
            logger.info("Switched hand tracker to: MEDIAPIPE (LIVE CAMERA)")

    @property
    def tracker_name(self) -> str:
        if isinstance(self.active_tracker, MediaPipeHandTracker):
            return "MEDIAPIPE (LIVE)"
        return "MOCK (SYNTHETIC)"

    async def _network_step(self, frame: np.ndarray, frame_id: int, intent: str, control_command: Optional[str]) -> None:
        """Background WS round-trip. Never awaited by the render loop directly - it only
        ever reads the latest completed snapshot, so a slow/laggy server cannot stall FPS
        or starve keyboard input polling."""
        t0 = time.perf_counter()
        try:
            response = await self.ws_client.send_frame(
                frame, frame_id, intent=intent, control_command=control_command
            )
        except Exception as e:
            logger.warning(f"Network step failed: {e}")
            response = None
        finally:
            self._network_inflight = False

        if response is None:
            return

        self._network_got_first_response = True
        self._network_latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            workflow_phase = ExecutionPhase(response.workflow_phase)
        except ValueError:
            workflow_phase = ExecutionPhase.IDLE

        parsed_scene = response.get_parsed_scene()
        rep = response.get_episode_report()
        if rep:
            self.last_episode_report = rep
            self._reward_history.append(float(rep.episode_reward))

        self._remote_snapshot.update({
            "poses": response.get_hand_poses(),
            "bboxes": parsed_scene.bounding_boxes if parsed_scene else [],
            "affordance_map": response.get_affordance_map(),
            "foreseen_traj": response.get_foreseen_trajectory(),
            "depth_heatmap": response.decode_depth_heatmap(),
            "gripper_cmd": response.gripper_action,
            "residuals": response.policy_residuals,
            "reward_score": response.reward_score,
            "discrepancy_norm": response.discrepancy_norm,
            "buffer_steps": response.buffer_step_count,
            "parsed_intent": response.get_parsed_intent(),
            "workflow_phase": workflow_phase,
            "phase_progress": response.phase_progress,
            "benchmark_summary": response.benchmark_summary or self._remote_snapshot["benchmark_summary"],
            "policy_loss": response.policy_loss,
            "policy_updates": response.policy_updates,
            "learned_wrist_bias": response.learned_wrist_bias,
            "depth_raw": response.decode_depth_raw(),
        })

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """sounddevice callback (runs on its own thread). Forwards raw PCM only while
        the transcriber is actively listening; WhisperTranscriber buffers it under a lock."""
        if status:
            logger.debug(f"Audio input status: {status}")
        if self.transcriber.is_listening:
            self.transcriber.transcribe_stream(indata.copy().tobytes())

    def setup_microphone(self) -> None:
        """Open a persistent microphone input stream feeding the transcriber, so
        push-to-talk voice intent capture actually has real audio to transcribe.

        Opening a CoreAudio input stream can block indefinitely on macOS while an
        unresolved microphone-permission prompt sits behind the process (e.g. no
        foreground window to attach it to) - so this runs in a worker thread with a
        hard timeout instead of calling sd.InputStream() directly on the main thread.
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice is not installed; voice intent capture will use canned fallback text.")
            return

        result: dict = {}

        def _open_stream() -> None:
            try:
                sample_rate = getattr(self.transcriber, "sample_rate", 16000)
                stream = sd.InputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                    callback=self._audio_callback,
                )
                stream.start()
                result["stream"] = stream
                result["sample_rate"] = sample_rate
            except Exception as e:
                result["error"] = e

        opener = threading.Thread(target=_open_stream, daemon=True)
        opener.start()
        opener.join(timeout=5.0)

        if opener.is_alive():
            logger.warning(
                "Microphone input stream did not open within 5s (likely blocked on an "
                "unresolved permission prompt - check System Settings > Privacy & Security > "
                "Microphone). Continuing without live voice capture; grant access and restart "
                "to enable it."
            )
            self._audio_stream = None
        elif "stream" in result:
            self._audio_stream = result["stream"]
            logger.info(f"Microphone input stream opened at {result['sample_rate']} Hz for voice intent capture.")
        else:
            logger.warning(f"Could not open microphone input stream ({result.get('error')}). Voice intent capture will use canned fallback text.")
            self._audio_stream = None

    def _open_capture(self, device_id: int):
        """Open a capture device, preferring AVFoundation on macOS."""
        if sys.platform == "darwin":
            cap = cv2.VideoCapture(device_id, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(device_id)

    @staticmethod
    def _delivers_frames(cap, attempts: int = 8) -> bool:
        """Whether a capture actually yields frames, not merely opens.

        isOpened() is not the same question. A camera asked for a mode it does
        not support opens happily and then fails every read - on macOS each one
        blocks for a full second first, so the client appears to hang rather
        than to fail.
        """
        for _ in range(attempts):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.02)
        return False

    def setup_camera(self) -> None:
        """Initialize physical camera with AVFoundation on macOS or fallback to synthetic."""
        device_id = self.config.camera.device_id
        logger.info(f"Opening camera device {device_id}...")

        self.cap = self._open_capture(device_id)

        if not self.cap.isOpened():
            logger.warning(f"Could not open physical camera (device_id={device_id}).")
            if self.config.camera.use_synthetic_if_unavailable:
                logger.info("Initializing Synthetic Camera fallback generator...")
                self.cap = SyntheticCamera(
                    width=self.config.camera.width,
                    height=self.config.camera.height,
                    fps=self.config.camera.fps
                )
                self.is_synthetic_camera = True
            else:
                raise RuntimeError(f"Failed to open video capture device {device_id}")
        else:
            want_w, want_h = self.config.camera.width, self.config.camera.height
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._delivers_frames(self.cap):
                # Requesting an unsupported mode is not a soft failure: the
                # device stops delivering entirely. Reopen and take whatever it
                # natively offers, resizing in software instead.
                logger.warning(
                    f"Camera {device_id} opened but delivers no frames at "
                    f"{want_w}x{want_h} - that mode is unsupported. Reopening at "
                    f"the camera's native resolution."
                )
                self.cap.release()
                self.cap = self._open_capture(device_id)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self._delivers_frames(self.cap):
                    logger.warning(f"Camera {device_id} delivers no frames in any mode.")
                    self.cap.release()
                    if self.config.camera.use_synthetic_if_unavailable:
                        logger.info("Initializing Synthetic Camera fallback generator...")
                        self.cap = SyntheticCamera(width=want_w, height=want_h,
                                                   fps=self.config.camera.fps)
                        self.is_synthetic_camera = True
                    else:
                        raise RuntimeError(f"Camera {device_id} delivers no frames")

            if not self.is_synthetic_camera:
                got_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                note = "" if (got_w, got_h) == (want_w, want_h) else \
                    f" - resizing to {want_w}x{want_h} in software"
                logger.info(
                    f"Camera opened successfully ({got_w}x{got_h} @ "
                    f"{self.config.camera.fps} FPS){note}."
                )

    async def run(self) -> None:
        """Main application execution loop."""
        self.setup_camera()
        # Take the sensor's cadence off the render loop. VideoCapture.read()
        # blocks until the device has a frame, which measured ~33 ms - about a
        # third of the whole frame budget - spent waiting on hardware. The
        # reader thread also drops frames the loop was too slow to take, so what
        # is displayed is the present rather than a backlog.
        self.camera = CameraStream(self.cap)
        self.setup_microphone()
        cv2.namedWindow(self.config.visualization.window_name, cv2.WINDOW_NORMAL)
        # Open at the stage's own size. Sizing the window to the CAMERA's
        # resolution left the composed widescreen stage squeezed into a 640x480
        # box the moment the app started.
        cv2.resizeWindow(self.config.visualization.window_name,
                         self.stage.width, self.stage.height)
        # And put it somewhere visible. Left to place the window itself, macOS
        # dropped a full-height window below the bottom edge of the display -
        # the app was running, drawing, and responding to keys, with a ~100 px
        # sliver on screen. It looked exactly like nothing had opened.
        self._place_window_on_screen()
        self.robot.connect()
        if self.server_url:
            mode_str = f"REMOTE CLOUD GPU ({self.server_url})"
        elif self.mode == "mock_local":
            mode_str = "MOCK LOCAL"
        else:
            mode_str = f"MOCK REMOTE (ws://{self.config.network.server_host}:{self.config.network.server_port})"
        logger.info(f"Starting LocalClient in [{mode_str}]. Tracker: {self.tracker_name}. Press 'q' or ESC to exit.")

        if self.auto_record:
            self.toggle_recording(self.config.camera.width, self.config.camera.height)

        frame_id = 0
        consecutive_failures = 0

        try:
            while True:
                t_frame_start = time.perf_counter()

                # Stage 1: Frame Ingestion & Preprocessing
                with self.profiler.profile("1. Frame Ingestion"):
                    ret, frame = self.camera.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures % 30 == 1:
                        logger.warning(f"Camera frame drop detected ({consecutive_failures} frames). Retrying...")

                    # Only switch to synthetic camera if 150 consecutive frame drops occur (>5 seconds of failure)
                    if consecutive_failures >= 150 and not self.is_synthetic_camera and self.config.camera.use_synthetic_if_unavailable:
                        logger.warning("Camera unresponsive for 150 frames. Automatically switching to Synthetic Camera generator.")
                        if self.cap:
                            self.cap.release()
                        # Hand the reader thread the new device rather than
                        # leaving it pumping a released one.
                        self.cap = SyntheticCamera(
                            width=self.config.camera.width,
                            height=self.config.camera.height,
                            fps=self.config.camera.fps
                        )
                        self.is_synthetic_camera = True
                        if self.camera is not None:
                            self.camera.replace_capture(self.cap)
                        consecutive_failures = 0

                    await asyncio.sleep(0.005)
                    continue

                consecutive_failures = 0
                frame_id += 1

                # The camera may be running at its native resolution because the
                # configured one was unsupported; everything downstream (network
                # payload size, recorder, HUD layout) assumes the configured one.
                if frame.shape[1] != self.config.camera.width or frame.shape[0] != self.config.camera.height:
                    frame = cv2.resize(frame, (self.config.camera.width, self.config.camera.height),
                                       interpolation=cv2.INTER_AREA)

                # Mirror real webcam feed horizontally for natural selfie view
                if not self.is_synthetic_camera:
                    frame = cv2.flip(frame, 1)

                clean_frame_for_rec = frame.copy()

                poses: List[HandPose] = []
                bboxes: List[BoundingBox3D] = []
                affordance_map: Optional[AffordanceMap] = None
                foreseen_traj: Optional[ForeseenTrajectory] = None
                depth_heatmap: Optional[np.ndarray] = None
                gripper_cmd = 0.0
                residuals: Optional[List[float]] = None
                reward_score = 0.0
                discrepancy_norm = 0.0
                buffer_steps = 0
                policy_loss = 0.0
                latency_ms = 0.0
                state_vec: Optional[np.ndarray] = None
                parsed_intent_resp: Optional[ParsedIntent] = self.current_parsed_intent
                workflow_phase = self.workflow.current_phase
                phase_progress = self.workflow.phase_progress
                benchmark_summary = self.benchmark.get_summary()

                if self.mode == "mock_local":
                    # Stage 2: Hand Tracking & Scene Parsing
                    with self.profiler.profile("2. Perception & Scene Parsing"):
                        t0 = time.perf_counter()
                        poses = self.active_tracker.estimate(frame)
                        depth_map = self.local_depth_estimator.estimate_depth(frame)
                        depth_heatmap = depth_map.to_colored_heatmap()
                        self._latest_depth_m = depth_map.depth
                        
                        prompt_for_scene = self.current_parsed_intent.target_object if self.current_parsed_intent.is_active else self.intent
                        detector = getattr(self, "local_object_detector", None)
                        if detector is not None:
                            detector.set_hint(prompt_for_scene)
                        parsed_scene = self.local_scene_parser.parse_scene(
                            image=frame,
                            depth=depth_map,
                            intent=prompt_for_scene
                        )
                        bboxes = parsed_scene.bounding_boxes

                    current_foreseen_step = None
                    target_box = None

                    # Stage 3: Foreseen Trajectory Generation
                    with self.profiler.profile("3. Foreseen Trajectory"):
                        if bboxes:
                            target_box = bboxes[0]
                            affordance_map = self.local_affordance_extractor.extract_affordance(
                                bounding_box=target_box,
                                intent=self.intent
                            )
                            start_h = poses[0] if poses else None
                            local_live_guidance = self.workflow.current_phase in (
                                ExecutionPhase.FORESEEING, ExecutionPhase.USER_EXECUTING
                            )
                            if self._cached_foreseen_traj is None or local_live_guidance:
                                self._cached_foreseen_traj = self.local_trajectory_diffusion.generate_foreseen_rollout(
                                    start_hand_pose=start_h,
                                    target_object=target_box,
                                    affordance_map=affordance_map,
                                    intent=self.intent,
                                    num_steps=60,
                                    learned_bias=self._local_learned_wrist_bias,
                                    action=self._refresh_action_plan()
                                )
                                self.workflow.stored_foreseen_trajectory = self._cached_foreseen_traj

                            foreseen_traj = self._cached_foreseen_traj
                            step_idx = self.workflow.step_index % len(self._cached_foreseen_traj.waypoints)
                            current_foreseen_step = self._cached_foreseen_traj.waypoints[step_idx]

                    # Stage 4: Workflow Lifecycle Stepping
                    if self.workflow.current_phase == ExecutionPhase.FORESEEING:
                        self.workflow.step_foresee()
                    elif self.workflow.current_phase == ExecutionPhase.WAIT_USER:
                        self.workflow.step_wait_user()
                    elif self.workflow.current_phase == ExecutionPhase.USER_EXECUTING:
                        real_h = poses[0] if poses else None
                        self.workflow.record_execution_step(real_h)
                    elif self.workflow.current_phase == ExecutionPhase.ADAPTING:
                        if not self._local_adaptation_computed_this_episode:
                            rep = self.local_discrepancy_engine.compile_episode_discrepancy(
                                foreseen_traj=self.workflow.stored_foreseen_trajectory,
                                recorded_poses=self.workflow.recorded_physical_poses,
                                policy=self.local_policy
                            )
                            self.last_episode_report = rep
                            self._reward_history.append(float(rep.episode_reward))
                            self._record_trial_if_scorable(rep)
                            episode_offset = np.clip(
                                np.array(rep.grasp_wrist_offset, dtype=np.float32), -0.05, 0.05
                            )
                            self._local_learned_wrist_bias = np.clip(
                                # Accumulate: episode_offset is the residual left
                                # after the current bias, not the total. See the
                                # note in ws_server on why averaging toward it
                                # stalls at half the user's real offset.
                                self._local_learned_wrist_bias + 0.4 * episode_offset,
                                -0.05, 0.05
                            )
                            self._cached_foreseen_traj = None
                            self._local_adaptation_computed_this_episode = True
                        # Holds here for workflow.adapting_duration_sec instead of advancing
                        # immediately, so the post-execution replay review has screen time.
                        self.workflow.step_adapting()
                    elif self.workflow.current_phase == ExecutionPhase.RESTARTING:
                        self._local_adaptation_computed_this_episode = False
                        self.workflow.step_restarting()

                    workflow_phase = self.workflow.current_phase
                    phase_progress = self.workflow.phase_progress
                    benchmark_summary = self.benchmark.get_summary()

                    # Stage 5: Discrepancy Calculation
                    with self.profiler.profile("4. Discrepancy Engine"):
                        real_h = poses[0] if poses else None
                        disc_state = self.local_discrepancy_engine.evaluate(
                            real_hand=real_h,
                            foreseen_step=current_foreseen_step,
                            target_object=target_box,
                            last_action=self._last_action
                        )
                        reward_score = disc_state.reward
                        discrepancy_norm = disc_state.discrepancy_norm
                        state_vec = disc_state.state_vector

                    # Stage 6: Residual Policy Forward & Update Step
                    with self.profiler.profile("5. Residual Policy"):
                        action = self.local_policy.evaluate(disc_state.state_vector)
                        self._last_action = action.joint_residuals.copy()
                        residuals = action.joint_residuals.tolist()
                        gripper_cmd = action.gripper_action

                        if self.adaptation_active and workflow_phase == ExecutionPhase.USER_EXECUTING:
                            self.local_policy.record_transition(
                                state=disc_state.state_vector,
                                action=action.joint_residuals,
                                reward=disc_state.reward
                            )

                        # Robot command step with safety monitor filtering
                        target_joints = np.zeros(7, dtype=np.float32)
                        if current_foreseen_step is not None:
                            target_joints[:min(6, len(target_joints))] = current_foreseen_step.wrist_pose[:min(6, len(target_joints))]
                            target_joints[:len(action.joint_residuals)] += action.joint_residuals
                        
                        cur_robot_state = self.robot.read_joint_states()
                        cart_pos = poses[0].keypoints_3d[0] if poses else np.array([0.05, 0.10, 0.50], dtype=np.float32)
                        safety = self.safety_monitor.evaluate_safety(
                            target_q=target_joints,
                            current_q=cur_robot_state.joint_positions,
                            cartesian_pos=cart_pos,
                            last_packet_time=time.time(),
                            obstacles=bboxes,
                            dt=0.033
                        )
                        self.robot.send_joint_commands(safety.clamped_joint_positions, gripper_command=action.gripper_action)

                        buffer_steps = self.local_policy.step_count
                        policy_loss = getattr(self.local_policy, "loss_history", [0.0])[-1]
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                elif self.mode == "mock_remote":
                    # Stage 7: Network transport runs in the background so a slow/laggy
                    # server round-trip never blocks rendering or keyboard polling.
                    # The live local tracker still runs every frame for a responsive
                    # skeleton overlay, independent of network latency.
                    if isinstance(self.active_tracker, MediaPipeHandTracker):
                        local_poses = self.active_tracker.estimate(frame)
                        poses = local_poses if local_poses else self._remote_snapshot["poses"]
                    else:
                        poses = self._remote_snapshot["poses"]

                    if not self._network_inflight:
                        self._network_inflight = True
                        self._network_task = asyncio.create_task(
                            self._network_step(frame.copy(), frame_id, self.intent, self._control_cmd_to_send)
                        )
                        self._control_cmd_to_send = None

                    latency_ms = self._network_latency_ms
                    snap = self._remote_snapshot
                    bboxes = snap["bboxes"]
                    affordance_map = snap["affordance_map"]
                    foreseen_traj = snap["foreseen_traj"]
                    depth_heatmap = snap["depth_heatmap"]
                    if snap["depth_raw"] is not None:
                        self._latest_depth_m = snap["depth_raw"]
                    gripper_cmd = snap["gripper_cmd"]
                    residuals = snap["residuals"]
                    reward_score = snap["reward_score"]
                    discrepancy_norm = snap["discrepancy_norm"]
                    buffer_steps = snap["buffer_steps"]
                    policy_loss = snap["policy_loss"]
                    parsed_intent_resp = snap["parsed_intent"] or self.current_parsed_intent
                    workflow_phase = snap["workflow_phase"]
                    phase_progress = snap["phase_progress"]
                    if snap["benchmark_summary"]:
                        benchmark_summary = snap["benchmark_summary"]

                    if not self._network_got_first_response:
                        T.draw(frame, f"Connecting to {self.server_url or self.config.network.server_host}…",
                               (30, frame.shape[0] // 2), 15, UIH.C["orange"],
                               weight="medium")

                # A short tone on each phase change, rather than a spoken
                # sentence. Speech arrived a second or two after the moment it
                # described, talked over someone concentrating on a reach, and
                # consumed Gemini quota that transcription and detection need
                # more. A tone says "something changed" without asking to be
                # listened to. This applies to BOTH modes: it used to sit
                # inside the remote branch, so a local session got macOS `say`
                # narrating the workflow instead.
                if workflow_phase != self._last_announced_phase:
                    previous = self._last_announced_phase
                    self._last_announced_phase = workflow_phase
                    self._sound_phase_change(workflow_phase, previous)

                # Client-side afterimage recording: capture the user's OWN real hand
                # poses (already tracked locally every frame in both modes) while they
                # execute, so the ghost hand can later replay their ACTUAL motion
                # instead of a synthetic plan. Snapshot the finished recording the
                # moment execution ends, before a future RESTARTING loop starts a
                # fresh one. A genuinely NEW target (as opposed to the RESTARTING loop
                # re-attempting the SAME target) drops any stale recording from a
                # different object.
                if self.workflow._target_label != self._last_recording_target:
                    self._last_recording_target = self.workflow._target_label
                    self._last_completed_recording = []
                    self._local_recorded_poses = []
                if workflow_phase != self._last_recording_phase:
                    if workflow_phase == ExecutionPhase.USER_EXECUTING:
                        self._local_recorded_poses = []
                    elif self._last_recording_phase == ExecutionPhase.USER_EXECUTING and self._local_recorded_poses:
                        self._last_completed_recording = list(self._local_recorded_poses)
                    self._last_recording_phase = workflow_phase
                if workflow_phase == ExecutionPhase.USER_EXECUTING and poses:
                    self._local_recorded_poses.append(poses[0])

                # Refresh the real object photo-crop whenever the object is plainly
                # visible and not mid-grasp (a real hand about to close on it would
                # otherwise get baked into the snapshot).
                if bboxes and workflow_phase not in (ExecutionPhase.USER_EXECUTING, ExecutionPhase.ADAPTING):
                    sprite = self.visualizer.capture_object_sprite(frame, bboxes[0])
                    if sprite is not None:
                        self._object_sprite = sprite

                # Session Recording
                if self.recorder.is_recording:
                    mano_dict = poses[0].mano_params.to_dict() if (poses and poses[0].mano_params) else None
                    kpts_3d_list = poses[0].keypoints_3d.tolist() if poses else None
                    kpts_2d_list = poses[0].keypoints_2d.tolist() if poses else None
                    
                    tel_record = {
                        "frame_id": frame_id,
                        "timestamp": time.time(),
                        "intent": self.intent,
                        "parsed_intent": parsed_intent_resp.to_dict() if parsed_intent_resp else None,
                        "workflow_phase": workflow_phase.value,
                        "phase_progress": phase_progress,
                        "mano_params": mano_dict,
                        "keypoints_3d": kpts_3d_list,
                        "keypoints_2d": kpts_2d_list,
                        "state_vector": state_vec.tolist() if state_vec is not None else None,
                        "residual_action": residuals,
                        "reward_score": reward_score,
                        "discrepancy_norm": discrepancy_norm,
                        "gripper_command": gripper_cmd
                    }
                    self.recorder.record_frame(clean_frame_for_rec, tel_record)

                # Console Profiling Table
                if self.enable_profiling and frame_id % 90 == 0:
                    fps_val = self.visualizer.update_fps()
                    print("\n" + self.profiler.format_table(fps=fps_val) + "\n")

                # Render Visualizations.
                #
                # Annotations are drawn at the VIDEO CARD's resolution, not the
                # sensor's. Drawing them on the 640x480 feed and letting the
                # stage scale it up left every line and label soft next to the
                # crisp chrome around it - and the upscale happens either way,
                # so doing it first costs nothing. Line drawing is vector work,
                # so a larger canvas is essentially free; only the pixel-for-
                # pixel lab render still pays for its own resolution.
                fps = self.visualizer.update_fps()
                frame, display_poses = self._to_display_resolution(frame, poses)
                self.visualizer.draw_hand_skeleton(
                    frame, display_poses, residuals=residuals,
                    adaptation_active=self.adaptation_active)
                # While a ghost hand is grasping on the live view, drop the 3-D
                # wireframe to a plain rectangle - its pillars cut through the
                # fingers exactly where the grasp needs to be legible.
                ghost_is_grasping = workflow_phase in (
                    ExecutionPhase.FORESEEING, ExecutionPhase.ADAPTING
                ) and bool(self._last_completed_recording)
                self.visualizer.draw_3d_bounding_boxes(frame, bboxes,
                                                       simplified=ghost_is_grasping)
                self.visualizer.draw_affordance_hotspots(frame, affordance_map)

                # What learning has changed about the plan, drawn on the plan.
                # Shown while the system is previewing or waiting - never while
                # the user executes, when their hand needs a clear stage.
                if workflow_phase in (ExecutionPhase.FORESEEING, ExecutionPhase.WAIT_USER):
                    display_bias = (self._local_learned_wrist_bias
                                    if self.mode == "mock_local"
                                    else self._remote_snapshot["learned_wrist_bias"])
                    self.visualizer.draw_policy_corrections(frame, foreseen_traj,
                                                            display_bias)
                
                # Ghost hand is an afterimage/replay of the user's OWN real recorded
                # motion, never a synthetic generated plan:
                #  - FORESEEING: replay of the PREVIOUS attempt, re-anchored to start
                #    from wherever the real hand currently is. Nothing is shown on the
                #    very first attempt at an object (no prior recording exists yet).
                #  - ADAPTING: a review moment replaying the attempt that JUST
                #    finished, at its own real recorded positions (no re-anchoring).
                #  - Never drawn during USER_EXECUTING - the user's real hand is
                #    unobstructed while actually performing the action.
                replay_poses: Optional[List[HandPose]] = None
                replay_reanchor = False
                ghost_label = ""
                if workflow_phase == ExecutionPhase.FORESEEING and self._last_completed_recording:
                    replay_poses = self._last_completed_recording
                    replay_reanchor = True
                    ghost_label = "Preview · your last attempt"
                elif workflow_phase == ExecutionPhase.ADAPTING and self._last_completed_recording:
                    replay_poses = self._last_completed_recording
                    replay_reanchor = False
                    ghost_label = "Replay · what you just did"
                # The Autonomous Demo is deliberately absent from this list: it is
                # not drawn as an overlay at all any more. It is staged and
                # rendered as a 3-D reenactment inside the simulated lab, which is
                # composited over the whole frame further down (_update_lab_panel).

                # The ghost is replayed from recordings held in SENSOR pixels;
                # on the enlarged card they would be drawn at a fraction of the
                # right position. Scale display copies, never the recordings -
                # they are the episode being scored.
                replay_poses = self._scale_poses_for_display(replay_poses)

                foreseen_step = self.visualizer.draw_hand_replay(
                    frame, replay_poses, real_poses=display_poses, reanchor=replay_reanchor, label=ghost_label,
                    target_bbox=bboxes[0] if bboxes else None, object_sprite=self._object_sprite
                )
                # The simulated-lab reenactment irises open over the camera
                # image itself, so it is composited before the frame is placed
                # on the stage and appears inside the video card.
                self._update_lab_panel(frame, workflow_phase, phase_progress,
                                       foreseen_traj, bboxes)

                # Chrome is drawn on the stage, not on the feed. The feed keeps
                # only what is anchored to image content - the skeleton, the
                # boxes, the ghost hand and the lab viewport.
                display = self._compose_stage(
                    frame,
                    fps=fps, latency_ms=latency_ms, workflow_phase=workflow_phase,
                    phase_progress=phase_progress, parsed_intent=parsed_intent_resp,
                    depth_heatmap=depth_heatmap, poses=poses, gripper_cmd=gripper_cmd,
                    reward_score=reward_score, discrepancy_norm=discrepancy_norm,
                    policy_loss=policy_loss, benchmark_summary=benchmark_summary,
                    target_label=(parsed_intent_resp.target_object
                                  if parsed_intent_resp else self.intent),
                    has_replay=bool(self._last_completed_recording),
                )
                cv2.imshow(self.config.visualization.window_name, display)

                # Handle keyboard inputs
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")): # ESC or 'q'
                    logger.info("Exit signal received. Terminating client.")
                    break
                elif key in (13, ord("c")): # ENTER or 'c' to advance workflow state
                    self.advance_workflow_phase()
                elif key in (ord("v"), 32): # 'v' or SPACEBAR for Voice Push-To-Talk
                    self.toggle_voice_mode()
                elif key == ord("m"): # Toggle Co-Adaptation Benchmark Panel
                    self.visualizer.show_analytics_panel = not self.visualizer.show_analytics_panel
                elif key == ord("h"): # Toggle expanded telemetry detail
                    self.visualizer.show_telemetry_detail = not self.visualizer.show_telemetry_detail
                elif key == ord("g"): # Toggle notification cues
                    self.toggle_voice_guidance()
                elif key == ord("e"): # Export co-adaptation benchmark trials to JSON + CSV
                    if self.benchmark.total_trials == 0:
                        logger.info("Benchmark Export: No trials recorded yet.")
                    else:
                        json_path = self.benchmark.export_summary_json()
                        csv_path = self.benchmark.export_csv()
                        logger.info(f"Benchmark Export: Wrote {json_path} and {csv_path}")
                elif key in (ord("k"), 19): # 'k' or Ctrl+S: Save checkpoint
                    self.save_checkpoint()
                elif key in (ord("l"), 12): # 'l' or Ctrl+L: Load checkpoint
                    self.load_checkpoint()
                elif key in (ord("x"), 18): # 'x' or Ctrl+R: Reset baseline
                    self.reset_baseline()
                elif key == ord("r"): # Toggle Session Recording
                    self.toggle_recording(self.config.camera.width, self.config.camera.height)
                elif key == ord("p"): # Toggle Online Residual Adaptation
                    self.toggle_adaptation()
                elif key == ord("i"): # Cycle intent prompt
                    self.cycle_intent()
                elif key == ord("f"): # Toggle Foreseen Ghost Hand Trajectory
                    self.visualizer.show_foreseen_ghost = not self.visualizer.show_foreseen_ghost
                elif key == ord("t"): # Toggle tracker
                    self.toggle_tracker()
                elif key == ord("d"): # Toggle depth PIP
                    self.visualizer.show_depth_inset = not self.visualizer.show_depth_inset
                elif key == ord("b"): # Toggle 3D bounding box
                    self.visualizer.show_bounding_box = not self.visualizer.show_bounding_box
                elif key == ord("s"): # Screenshot
                    screenshot_name = f"snapshot_frame_{frame_id}_{int(time.time())}.png"
                    cv2.imwrite(screenshot_name, frame)
                    logger.info(f"Saved screenshot to {screenshot_name}")
                elif key == ord("z"): # Toggle fullscreen
                    self.toggle_fullscreen()
                elif key == ord("a"): # Autonomous Demo: hands-off simulated pick
                    self.trigger_autonomous_demo()

                # Maintain 30 FPS yielding to asyncio
                elapsed = time.perf_counter() - t_frame_start
                target_dt = 1.0 / self.config.camera.fps
                sleep_time = max(0.001, target_dt - elapsed)
                await asyncio.sleep(sleep_time)

        finally:
            if self._network_task is not None and not self._network_task.done():
                self._network_task.cancel()
            if self._audio_stream is not None:
                self._audio_stream.stop()
                self._audio_stream.close()
            if self.recorder.is_recording:
                self.recorder.stop_recording()
            # The reader thread owns the device once started; stopping it
            # releases the capture, so do not release it twice.
            if self.camera is not None:
                self.camera.release()
                self.cap = None
            elif self.cap:
                self.cap.release()
            self.robot.disconnect()
            self.speaker.stop()
            self.sounds.close()
            cv2.destroyAllWindows()
            await self.ws_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visuomotor Hand Policy Local Client (Intel Mac CPU Friendly)")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Path to system_config.yaml")
    parser.add_argument("--mode", type=str, choices=["mock_local", "mock_remote", "mock", "remote"], default=None, help="Execution mode (mock_local | mock_remote)")
    parser.add_argument("--tracker", type=str, choices=["mediapipe", "mock"], default=None, help="Hand tracker type")
    parser.add_argument("--transcriber", type=str, choices=["mock", "whisper"], default="whisper", help="Audio transcriber engine")
    parser.add_argument("--intent", type=str, default=None, help="Natural language intent string")
    parser.add_argument("--device", type=int, default=None, help="Camera device index")
    parser.add_argument("--profile", action="store_true", help="Enable detailed component latency breakdown profiling to console")
    parser.add_argument("--record", action="store_true", help="Automatically begin session recording on launch")
    parser.add_argument("--server-url", type=str, default=None, help="Custom WebSocket server URL (e.g. ws://<CLOUD_GPU_IP>:8765)")
    parser.add_argument("--gemini-key", type=str, default=os.environ.get("GEMINI_API_KEY"), help="Gemini API key for voice transcription + TTS (defaults to $GEMINI_API_KEY)")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="Start even if the camera, microphone or server are unavailable. "
                             "Output may be SYNTHETIC - never present it as real.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip startup verification.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    app_config = AppConfig.from_yaml(config_path)
    if args.device is not None:
        app_config.camera.device_id = args.device

    # Verify before opening a window. A camera that opens but delivers nothing,
    # or a server that is still installing, both otherwise present as a blank
    # or frozen display with the reason buried in the log.
    if not args.skip_preflight:
        if args.gemini_key:
            os.environ.setdefault("GEMINI_API_KEY", args.gemini_key)
        remote = (args.mode or app_config.system.mode or "").startswith(("remote", "mock_remote"))
        url = args.server_url or (
            f"ws://{app_config.network.server_host}:{app_config.network.server_port}"
            if remote else None)
        checks = client_checks(server_url=url, device_id=app_config.camera.device_id)
        if not enforce(checks, "PRECOGNITION - local client preflight",
                       allow_degraded=args.allow_degraded):
            sys.exit(1)

    runner = LocalClientRunner(
        config=app_config,
        cli_mode=args.mode,
        tracker_type=args.tracker,
        intent=args.intent,
        enable_profiling=args.profile,
        enable_recording=args.record,
        server_url=args.server_url,
        transcriber_type=args.transcriber,
        gemini_api_key=args.gemini_key
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
