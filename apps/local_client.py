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
import asyncio
import collections
import math
import threading
import logging
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
from src.perception.hand_tracker import HAND_CONNECTIONS, HandPose, HandSide, HandTrackerABC
from src.perception.scene_parser import BoundingBox3D, ParsedScene
from src.perception.mediapipe_tracker import MediaPipeHandTracker, MEDIAPIPE_AVAILABLE
from src.perception.intent_parser import IntentParserABC, MockLLMIntentParser, ParsedIntent
from src.audio.speech_to_text import AudioTranscriberABC, MockTranscriber, WhisperTranscriber
from src.audio.text_to_speech import SpeechSynthesizerABC, SystemSpeaker
from src.simulation.trajectory_generator import AffordanceMap, ForeseenTrajectory, ForeseenWaypoint
from src.policy.discrepancy import DiscrepancyEngine, DiscrepancyState, EpisodeDiscrepancyReport
from src.policy.workflow_state import ExecutionPhase, WorkflowController
from src.policy.checkpointing import PolicyCheckpointManager
from src.analytics.benchmark import CoAdaptationBenchmark
from src.hardware.robot_interface import MockRobotHardware, RobotHardwareABC, RobotState
from src.safety.safety_monitor import SafetyMonitor, SafetyStatus
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
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

# Short lookahead into the continuously-replanned foreseen trajectory used to render
# the "current" ghost hand pose - roughly 12/60 * 2.0s duration ~= 0.4s ahead of the
# real hand's current position, rather than playing the full trajectory as an animation.
GHOST_LOOKAHEAD_STEPS = 12

# Preset intent prompts for cycling via keypress 'i'
PRESET_INTENTS = [
    "idle",
    "foresee me picking this remote control",
    "grasp the red coffee cup by the handle",
    "pick up the tall water bottle on the right",
    "grab the stylus pen near the keyboard",
]

# Modern Cyber-Sleek Color Palette (BGR)
PALETTE = {
    "cyan_electric": (255, 240, 0),      # #00F0FF Electric Cyan
    "neon_green": (80, 255, 120),        # #78FF50 Neon Emerald
    "neon_violet": (255, 60, 180),       # #B43CFF Neon Purple
    "amber_gold": (0, 205, 255),         # #FFCD00 Amber Gold
    "laser_red": (60, 60, 255),          # #FF3C3C Warning Red
    "dark_glass_bg": (12, 16, 22),       # Dark Slate Glass
    "glass_border": (70, 85, 110),       # High-tech Border
    "text_white": (245, 248, 252),       # Clean Crisp White
    "text_dim": (140, 155, 175),         # Muted Blue-Grey
}

# Anatomical finger bone colors (BGR)
FINGER_COLORS = {
    "thumb": (0, 215, 255),    # Gold
    "index": (80, 255, 100),   # Neon Green
    "middle": (255, 230, 40),  # Bright Cyan
    "ring": (255, 130, 20),    # Electric Azure
    "pinky": (240, 50, 240),   # Neon Magenta
    "palm": (190, 210, 225),   # Cool Slate
}

# 3D bounding box wireframe edges connecting 8 corner vertices
BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4), # Top face
    (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
]


def get_bone_color(idx1: int, idx2: int) -> tuple:
    """Select color based on anatomical finger group."""
    joints = {idx1, idx2}
    if 1 in joints or 2 in joints or 3 in joints or 4 in joints:
        return FINGER_COLORS["thumb"]
    elif 5 in joints or 6 in joints or 7 in joints or 8 in joints:
        return FINGER_COLORS["index"]
    elif 9 in joints or 10 in joints or 11 in joints or 12 in joints:
        return FINGER_COLORS["middle"]
    elif 13 in joints or 14 in joints or 15 in joints or 16 in joints:
        return FINGER_COLORS["ring"]
    elif 17 in joints or 18 in joints or 19 in joints or 20 in joints:
        return FINGER_COLORS["pinky"]
    return FINGER_COLORS["palm"]


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

    def _glass_panel(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        alpha: float = 0.80,
        radius: int = 14,
        border_color: Optional[tuple] = None,
    ) -> None:
        """Blend a soft rounded-corner frosted glass card onto the frame in place."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            return
        roi = frame[y1:y2, x1:x2]
        mask_f, contours = self._rounded_panel_mask(pw, ph, radius)
        glass = np.full_like(roi, PALETTE["dark_glass_bg"])
        blended = cv2.addWeighted(glass, alpha, roi, 1 - alpha, 0)
        roi[:] = (blended * mask_f + roi * (1 - mask_f)).astype(np.uint8)
        cv2.drawContours(roi, contours, -1, border_color or PALETTE["glass_border"], 1, lineType=cv2.LINE_AA)

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
        """Draw tracked 21-joint hand skeleton with glowing bones and pulsing nodes."""
        if not self.config.visualization.draw_skeleton:
            return

        h, w = frame.shape[:2]
        t = time.time()
        pulse = 0.5 + 0.5 * np.sin(t * 6.0)

        for pose in poses:
            kpts_2d = pose.keypoints_2d.copy()
            kpts_3d = pose.keypoints_3d.copy()

            if adaptation_active and residuals is not None and len(residuals) >= 7:
                res_arr = np.array(residuals[:7], dtype=np.float32)
                tip_indices = [4, 8, 12, 16, 20]
                for i, tip in enumerate(tip_indices):
                    delta_px = res_arr[i % len(res_arr)] * 80.0
                    kpts_2d[tip, 0] += delta_px
                    kpts_2d[tip, 1] -= delta_px * 0.5

            # 1. Outer Glow Pass for bones
            glow_layer = frame.copy()
            for u, v in HAND_CONNECTIONS:
                pt1 = (int(np.clip(kpts_2d[u, 0], 0, w - 1)), int(np.clip(kpts_2d[u, 1], 0, h - 1)))
                pt2 = (int(np.clip(kpts_2d[v, 0], 0, w - 1)), int(np.clip(kpts_2d[v, 1], 0, h - 1)))
                color = get_bone_color(u, v)
                cv2.line(glow_layer, pt1, pt2, color, thickness=6, lineType=cv2.LINE_AA)
            cv2.addWeighted(glow_layer, 0.40, frame, 0.60, 0, frame)

            # 2. Core Sharp Bone Lines
            for u, v in HAND_CONNECTIONS:
                pt1 = (int(np.clip(kpts_2d[u, 0], 0, w - 1)), int(np.clip(kpts_2d[u, 1], 0, h - 1)))
                pt2 = (int(np.clip(kpts_2d[v, 0], 0, w - 1)), int(np.clip(kpts_2d[v, 1], 0, h - 1)))
                color = get_bone_color(u, v)
                cv2.line(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

            # 3. High-Tech Joint Nodes & Fingertip Pulsing Halos
            for j_idx in range(21):
                pt = (int(np.clip(kpts_2d[j_idx, 0], 0, w - 1)), int(np.clip(kpts_2d[j_idx, 1], 0, h - 1)))
                is_tip = j_idx in (4, 8, 12, 16, 20)
                
                if is_tip:
                    # Pulsing outer ring
                    halo_r = int(7 + 3 * pulse)
                    ring_col = (100, 255, 180) if adaptation_active else (255, 230, 100)
                    cv2.circle(frame, pt, halo_r, ring_col, 1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, pt, 5, (10, 20, 25), -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, pt, 4, ring_col, -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, pt, 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 4, (15, 20, 30), -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, pt, 3, (220, 235, 250), -1, lineType=cv2.LINE_AA)

            # 4. Stylized 3D Coordinate Reticle on Wrist
            wrist_2d = (int(np.clip(kpts_2d[0, 0], 0, w - 1)), int(np.clip(kpts_2d[0, 1], 0, h - 1)))
            wrist_z = max(kpts_3d[0, 2], 0.1)
            axis_len = int(45 / wrist_z)
            cv2.arrowedLine(frame, wrist_2d, (wrist_2d[0] + axis_len, wrist_2d[1]), (60, 60, 255), 2, tipLength=0.2)
            cv2.arrowedLine(frame, wrist_2d, (wrist_2d[0], wrist_2d[1] - axis_len), (80, 255, 120), 2, tipLength=0.2)
            
            # Wrist badge pill
            badge_w, badge_h = 100, 20
            bx, by = wrist_2d[0] - 50, wrist_2d[1] + 20
            sub_rect = frame[max(0, by):min(h, by+badge_h), max(0, bx):min(w, bx+badge_w)]
            if sub_rect.size > 0:
                dark_badge = np.full_like(sub_rect, (15, 20, 30))
                cv2.addWeighted(dark_badge, 0.75, sub_rect, 0.25, 0, sub_rect)
            cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), PALETTE["cyan_electric"], 1)
            cv2.putText(frame, f"MANO {pose.side.value[:1].upper()}:{pose.confidence*100:.0f}%", 
                        (bx + 8, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _compute_similarity_transform(
        src_p0: np.ndarray, src_p1: np.ndarray, dst_p0: np.ndarray, dst_p1: np.ndarray
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

        scale = float(np.clip(len_dst / len_src, 0.4, 2.5))
        angle = math.atan2(v_dst[1], v_dst[0]) - math.atan2(v_src[1], v_src[0])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32) * scale
        t = dst_p0 - R @ src_p0
        return R.astype(np.float32), t.astype(np.float32)

    @staticmethod
    def _project_3d(p3d: np.ndarray, w: int, h: int) -> np.ndarray:
        """Project a 3D camera-frame point to 2D pixels using the same default pinhole
        intrinsics convention used throughout this project (BoundingBox3D.project_to_2d,
        MockTrajectoryDiffusion._project_2d) when no calibrated intrinsics are available."""
        fx = fy = 0.8 * w
        cx, cy = w / 2.0, h / 2.0
        z = max(float(p3d[2]), 0.1)
        return np.array([fx * p3d[0] / z + cx, fy * p3d[1] / z + cy], dtype=np.float32)

    def _draw_ghost_object_afterimage(
        self,
        frame: np.ndarray,
        trajectory: ForeseenTrajectory,
        step_idx: int,
        xf,
        target_bbox: BoundingBox3D,
    ) -> None:
        """Ghost afterimage of the object being picked up, following the trajectory's
        own object_pose per waypoint - which is already kinematically consistent (static
        until contact, then rigidly attached to the hand through the lift), not a
        separate free-floating animation."""
        h, w = frame.shape[:2]
        depth = max(float(target_bbox.center[2]), 0.1)
        px_w = max(10, int(0.8 * w * float(target_bbox.size[0]) / depth))
        px_h = max(10, int(0.8 * w * float(target_bbox.size[1]) / depth))

        trail_span = 24
        trail_steps = sorted(set(range(max(0, step_idx - trail_span), step_idx, 4)) | {step_idx})

        overlay = frame.copy()
        for s in trail_steps:
            wp = trajectory.waypoints[s]
            center = xf(self._project_3d(wp.object_pose[:3], w, h)[None, :])[0]
            cx_i, cy_i = int(np.clip(center[0], 0, w - 1)), int(np.clip(center[1], 0, h - 1))
            age = (step_idx - s) / float(trail_span)
            in_contact = float(np.max(wp.contact_state)) > 0.5
            color = (0, 210, 255) if in_contact else (150, 210, 255)
            cv2.ellipse(overlay, (cx_i, cy_i), (px_w // 2, px_h // 2), 0, 0, 360, color, -1, cv2.LINE_AA)
            if s != step_idx:
                cv2.addWeighted(overlay, 0.30 * (1.0 - age), frame, 1.0 - 0.30 * (1.0 - age), 0, dst=frame)
                overlay = frame.copy()

        # Crisp outline + label on the current step
        wp = trajectory.waypoints[step_idx]
        center = xf(self._project_3d(wp.object_pose[:3], w, h)[None, :])[0]
        cx_i, cy_i = int(np.clip(center[0], 0, w - 1)), int(np.clip(center[1], 0, h - 1))
        cv2.ellipse(frame, (cx_i, cy_i), (px_w // 2, px_h // 2), 0, 0, 360, (0, 235, 255), 2, cv2.LINE_AA)
        label = target_bbox.label.replace("_", " ")
        cv2.putText(frame, label, (cx_i - px_w // 2, cy_i - px_h // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 235, 255), 1, cv2.LINE_AA)

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

    def draw_foreseen_ghost_trajectory(
        self,
        frame: np.ndarray,
        trajectory: Optional[ForeseenTrajectory],
        step_override: Optional[int] = None,
        real_poses: Optional[List[HandPose]] = None,
        target_bbox: Optional[BoundingBox3D] = None,
        label: str = "NEXT-STEP GUIDE",
    ) -> int:
        """Render animated holographic Foreseen Ghost Hand rollout with particle comets
        and a ghost afterimage of the target object.

        Re-anchored every frame to the real, live hand pose (position AND orientation,
        via a 2D similarity transform from wrist -> middle-MCP) so the ghost visibly
        starts at the same spot and angle as the real hand and tracks it continuously,
        rather than a fixed pose baked in whenever the trajectory was generated
        server-side (which may be stale or from a frame where the hand wasn't visible)."""
        if not self.show_foreseen_ghost or trajectory is None or not trajectory.waypoints:
            self._smoothed_ghost_kpts = None
            return 0

        h, w = frame.shape[:2]
        num_steps = len(trajectory.waypoints)
        step_idx = (step_override if step_override is not None else self.anim_frame_idx) % num_steps
        current_wp: ForeseenWaypoint = trajectory.waypoints[step_idx]

        if real_poses and len(real_poses[0].keypoints_2d) > 9:
            real_kpts = real_poses[0].keypoints_2d
            ghost_kpts0 = trajectory.waypoints[0].hand_keypoints_2d
            self._ghost_transform = self._compute_similarity_transform(
                ghost_kpts0[0], ghost_kpts0[9], real_kpts[0], real_kpts[9]
            )
        R, t = self._ghost_transform

        def xf(pts_2d: np.ndarray) -> np.ndarray:
            return pts_2d @ R.T + t

        # 0. Ghost object afterimage (drawn first, underneath the hand)
        if target_bbox is not None:
            self._draw_ghost_object_afterimage(frame, trajectory, step_idx, xf, target_bbox)

        # 1. Shimmering Trajectory Comet Ribbon
        trail_pts_raw = xf(np.stack([wp.hand_keypoints_2d[0] for wp in trajectory.waypoints]))
        trail_pts = [
            (int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))) for u, v in trail_pts_raw
        ]

        if len(trail_pts) > 1:
            for i in range(len(trail_pts) - 1):
                alpha_frac = i / float(len(trail_pts))
                trail_color = (
                    int(255 * (1.0 - alpha_frac * 0.5)),
                    int(240 * alpha_frac),
                    int(80 * (1.0 - alpha_frac))
                )
                cv2.line(frame, trail_pts[i], trail_pts[i + 1], trail_color, 1 + int(alpha_frac * 2), lineType=cv2.LINE_AA)

        # 2. Holographic Dotted Ghost Hand Pose, smoothed toward the latest replanned
        # target so it glides continuously rather than snapping on each server update.
        target_kpts_2d = xf(current_wp.hand_keypoints_2d)
        if self._smoothed_ghost_kpts is None or self._smoothed_ghost_kpts.shape != target_kpts_2d.shape:
            self._smoothed_ghost_kpts = target_kpts_2d.copy()
        else:
            self._smoothed_ghost_kpts = self._smoothed_ghost_kpts + 0.35 * (target_kpts_2d - self._smoothed_ghost_kpts)
        ghost_kpts_2d = self._smoothed_ghost_kpts
        ghost_color = (255, 235, 100) # Bright Ice Cyan
        self._draw_hand_mesh(frame, ghost_kpts_2d, ghost_color, alpha=0.80)

        wrist_pt = (int(np.clip(ghost_kpts_2d[0, 0], 0, w - 1)), int(np.clip(ghost_kpts_2d[0, 1], 0, h - 1)))
        cv2.putText(frame, label, (wrist_pt[0] - 60, wrist_pt[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ghost_color, 1, cv2.LINE_AA)

        return step_idx + 1

    def draw_workflow_banner(
        self,
        frame: np.ndarray,
        phase: ExecutionPhase,
        progress: float,
        step_idx: int,
        discrepancy_norm: float = 0.0,
        episode_report: Optional[EpisodeDiscrepancyReport] = None
    ) -> None:
        """Render modern, compact Floating Dynamic Island banner at top center."""
        h, w = frame.shape[:2]
        banner_w = 360
        banner_h = 32
        bx1 = (w - banner_w - 240) // 2  # Center relative to demonstrator view area
        if bx1 < 10:
            bx1 = 15
        by1 = 10
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        self._glass_panel(frame, bx1, by1, bx2, by2, alpha=0.82, radius=16)

        t = time.time()
        pulse = 0.5 + 0.5 * np.sin(t * 8.0)

        if phase == ExecutionPhase.FORESEEING:
            cv2.circle(frame, (bx1 + 18, by1 + 16), int(4 + 2 * pulse), PALETTE["amber_gold"], -1, cv2.LINE_AA)
            msg = f"Foreseeing rollout  -  {int(progress*100)}%"
            cv2.putText(frame, msg, (bx1 + 30, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PALETTE["amber_gold"], 1, cv2.LINE_AA)

        elif phase == ExecutionPhase.WAIT_USER:
            cv2.circle(frame, (bx1 + 18, by1 + 16), 5, PALETTE["cyan_electric"], -1, cv2.LINE_AA)
            msg = "Ready - press 'c' to execute"
            cv2.putText(frame, msg, (bx1 + 30, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["text_white"], 1, cv2.LINE_AA)

        elif phase == ExecutionPhase.USER_EXECUTING:
            cv2.circle(frame, (bx1 + 18, by1 + 16), int(4 + 2 * pulse), PALETTE["neon_green"], -1, cv2.LINE_AA)
            msg = f"Tracking execution  -  {int(progress*100)}%"
            cv2.putText(frame, msg, (bx1 + 30, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["neon_green"], 1, cv2.LINE_AA)

            # High precision alignment bar
            bar_x = bx1 + 30
            bar_y = by1 + 24
            bar_w = banner_w - 46
            bar_h = 3
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 50, 60), -1)
            fill_w = int(bar_w * np.clip(1.0 - (discrepancy_norm / 0.10), 0.0, 1.0))
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), PALETTE["neon_green"], -1)

        elif phase == ExecutionPhase.RESTARTING:
            cv2.circle(frame, (bx1 + 18, by1 + 16), int(4 + 2 * pulse), PALETTE["neon_violet"], -1, cv2.LINE_AA)
            msg = f"Restarting with improved plan  -  {int(progress*100)}%"
            cv2.putText(frame, msg, (bx1 + 30, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["neon_violet"], 1, cv2.LINE_AA)
        elif phase == ExecutionPhase.ADAPTING or episode_report is not None:
            cv2.circle(frame, (bx1 + 18, by1 + 16), 5, PALETTE["neon_violet"], -1, cv2.LINE_AA)
            rew = episode_report.episode_reward if episode_report else 0.0
            msg = f"Adapted  -  {rew:+.2f} reward"
            cv2.putText(frame, msg, (bx1 + 30, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["neon_violet"], 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (bx1 + 18, by1 + 16), 3, PALETTE["text_dim"], -1, cv2.LINE_AA)
            msg = "Standby - press 'i' or talk ('v')"
            cv2.putText(frame, msg, (bx1 + 30, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["text_dim"], 1, cv2.LINE_AA)

    def draw_instruction_bar(self, frame: np.ndarray, phase: ExecutionPhase, target_label: str = "") -> None:
        """Large, unmissable bottom-center bar stating exactly what to do right now,
        in plain language - separate from the compact top-of-frame status banner."""
        h, w = frame.shape[:2]
        target = target_label.replace("_", " ") if target_label and target_label.lower() not in ("none", "") else "an object"

        messages = {
            ExecutionPhase.IDLE: ("STANDBY", "Hold 'v' or SPACE and say what to pick up, e.g. \"wine glass\"", PALETTE["text_dim"]),
            ExecutionPhase.FORESEEING: ("PREVIEWING", f"Watch the ghost hand plan how to grab the {target}...", PALETTE["amber_gold"]),
            ExecutionPhase.WAIT_USER: ("YOUR TURN", "Move your hand to match the ghost, then press 'c' when ready", PALETTE["cyan_electric"]),
            ExecutionPhase.USER_EXECUTING: ("GO", f"Reach for the {target} now - follow the ghost hand", PALETTE["neon_green"]),
            ExecutionPhase.ADAPTING: ("LEARNING", "Comparing your motion to the plan...", PALETTE["neon_violet"]),
            ExecutionPhase.RESTARTING: ("TRY AGAIN", f"Restarting with an improved plan for the {target}...", PALETTE["neon_violet"]),
        }
        title, body, color = messages.get(phase, messages[ExecutionPhase.IDLE])

        bar_w = min(600, w - 32)
        bar_h = 56
        bx1 = (w - bar_w) // 2
        by1 = h - bar_h - 16
        bx2, by2 = bx1 + bar_w, by1 + bar_h

        self._glass_panel(frame, bx1, by1, bx2, by2, alpha=0.85, radius=16, border_color=color)
        cv2.putText(frame, title, (bx1 + 18, by1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        cv2.putText(frame, body, (bx1 + 18, by1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PALETTE["text_white"], 1, cv2.LINE_AA)

    def draw_coadaptation_panel(
        self,
        frame: np.ndarray,
        benchmark_summary: Optional[dict] = None
    ) -> None:
        """Render modern Co-Adaptation Performance Analytics Glass Panel."""
        if not self.show_analytics_panel:
            return

        h, w = frame.shape[:2]
        panel_w = 260
        panel_h = 220
        px1 = w - panel_w - 250  # Dock neatly to the left of the right sidebar
        if px1 < 10:
            px1 = 10
        py1 = 10
        px2 = px1 + panel_w
        py2 = py1 + panel_h

        self._glass_panel(frame, px1, py1, px2, py2, alpha=0.86, radius=16, border_color=PALETTE["cyan_electric"])

        cv2.putText(frame, "Co-Adaptation", (px1 + 14, py1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["cyan_electric"], 1, cv2.LINE_AA)

        if not benchmark_summary or benchmark_summary.get("total_trials", 0) == 0:
            cv2.putText(frame, "No trials recorded yet.", (px1 + 12, py1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["text_dim"], 1, cv2.LINE_AA)
            cv2.putText(frame, "Complete Foresee-Execute cycles", (px1 + 12, py1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["text_dim"], 1, cv2.LINE_AA)
            cv2.putText(frame, "to track multi-trial learning.", (px1 + 12, py1 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["text_dim"], 1, cv2.LINE_AA)
        else:
            trials_cnt = benchmark_summary.get("total_trials", 0)
            reduction_pct = benchmark_summary.get("error_reduction_pct", 0.0)
            mean_r = benchmark_summary.get("mean_reward", 0.0)
            init_err = benchmark_summary.get("initial_error_mm", 0.0)
            cur_err = benchmark_summary.get("latest_error_mm", 0.0)

            cv2.putText(frame, f"TOTAL TRIALS: {trials_cnt}", (px1 + 12, py1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"INIT D_traj: {init_err:.1f} mm", (px1 + 12, py1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["amber_gold"], 1, cv2.LINE_AA)
            cv2.putText(frame, f"CURR D_traj: {cur_err:.1f} mm", (px1 + 12, py1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["neon_green"], 1, cv2.LINE_AA)
            
            # Error Reduction Badge
            red_color = PALETTE["neon_green"] if reduction_pct >= 0 else PALETTE["laser_red"]
            cv2.putText(frame, f"ERROR REDUCTION: {reduction_pct:+5.1f}%", (px1 + 12, py1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, red_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"MEAN REWARD: {mean_r:+0.3f}", (px1 + 12, py1 + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 240, 255), 1, cv2.LINE_AA)

        # Checkpoint hotkey instructions
        cv2.line(frame, (px1 + 12, py1 + 155), (px2 - 12, py1 + 155), (50, 65, 80), 1)
        cv2.putText(frame, "HOTKEYS: 'k':Save | 'l':Load | 'x':Reset", (px1 + 12, py1 + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.32, PALETTE["text_white"], 1, cv2.LINE_AA)
        cv2.putText(frame, "SAVED IN: config/profiles/default_user", (px1 + 12, py1 + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.28, PALETTE["text_dim"], 1, cv2.LINE_AA)

    def draw_3d_bounding_boxes(self, frame: np.ndarray, bboxes: List[BoundingBox3D]) -> None:
        """Render futuristic 3D bounding wireframes with corner brackets and scanning reticle."""
        if not self.show_bounding_box or not bboxes:
            return

        h, w = frame.shape[:2]

        for bbox in bboxes:
            corners_2d = bbox.project_to_2d(image_shape=(h, w))

            # Draw translucent base wireframe
            for u, v in BBOX_EDGES:
                pt1 = (int(np.clip(corners_2d[u, 0], -100, w + 100)), int(np.clip(corners_2d[u, 1], -100, h + 100)))
                pt2 = (int(np.clip(corners_2d[v, 0], -100, w + 100)), int(np.clip(corners_2d[v, 1], -100, h + 100)))
                cv2.line(frame, pt1, pt2, (0, 215, 255), thickness=1, lineType=cv2.LINE_AA)

            # High-tech L-brackets on the 8 corners
            for c_idx in range(8):
                pt = (int(np.clip(corners_2d[c_idx, 0], 0, w - 1)), int(np.clip(corners_2d[c_idx, 1], 0, h - 1)))
                cv2.circle(frame, pt, 4, (10, 20, 30), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, pt, 3, PALETTE["cyan_electric"], -1, lineType=cv2.LINE_AA)

            # Target Lock Badge
            top_y_idx = np.argmin(corners_2d[:, 1])
            lx = int(np.clip(corners_2d[top_y_idx, 0] - 40, 10, w - 280))
            ly = int(np.clip(corners_2d[top_y_idx, 1] - 16, 20, h - 10))
            
            label_text = f"TARGET: {bbox.label.upper()} [{bbox.center[2]:.2f}m]"
            lw = len(label_text) * 8 + 12
            cv2.rectangle(frame, (lx, ly - 14), (lx + lw, ly + 4), (15, 20, 30), -1)
            cv2.rectangle(frame, (lx, ly - 14), (lx + lw, ly + 4), PALETTE["cyan_electric"], 1)
            cv2.putText(frame, label_text, (lx + 6, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["cyan_electric"], 1, cv2.LINE_AA)

    def draw_affordance_hotspots(self, frame: np.ndarray, affordance_map: Optional[AffordanceMap]) -> None:
        """Project and draw glowing surface contact hotspots."""
        if affordance_map is None or not len(affordance_map.hotspots):
            return

        h, w = frame.shape[:2]
        fx = fy = 0.8 * w
        cx = w / 2.0
        cy = h / 2.0
        t = time.time()

        for h_idx, hs in enumerate(affordance_map.hotspots):
            z = max(hs[2], 0.1)
            u = int(fx * (hs[0] / z) + cx)
            v = int(fy * (hs[1] / z) + cy)
            if 0 <= u < w and 0 <= v < h:
                r = int(9 + 2 * np.sin(t * 6.0))
                cv2.circle(frame, (u, v), r, PALETTE["neon_green"], 1, cv2.LINE_AA)
                cv2.drawMarker(frame, (u, v), PALETTE["neon_green"], cv2.MARKER_CROSS, 8, 1)
                cv2.putText(frame, f"HOTSPOT #{h_idx+1}", (u + 12, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["neon_green"], 1, cv2.LINE_AA)

    def draw_depth_pip(self, frame: np.ndarray, depth_heatmap: Optional[np.ndarray]) -> None:
        """Draw Picture-in-Picture metric depth inset tucked on bottom-left."""
        if not self.show_depth_inset or depth_heatmap is None:
            return

        h, w = frame.shape[:2]
        pip_w = 120
        pip_h = 90

        pip_resized = cv2.resize(depth_heatmap, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
        
        x1 = 12
        y1 = h - pip_h - 12
        x2 = x1 + pip_w
        y2 = y1 + pip_h

        frame[y1:y2, x1:x2] = pip_resized
        cv2.rectangle(frame, (x1, y1), (x2, y2), PALETTE["glass_border"], 1)
        
        # Header tag
        cv2.rectangle(frame, (x1, y1), (x1 + 90, y1 + 16), (15, 20, 30), -1)
        cv2.putText(frame, "DEPTH [m]", (x1 + 4, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_telemetry_hud(
        self,
        frame: np.ndarray,
        fps: float,
        mode_str: str,
        tracker_name: str,
        intent: str,
        workflow_phase: ExecutionPhase,
        phase_progress: float = 0.0,
        parsed_intent: Optional[ParsedIntent] = None,
        voice_status: str = "IDLE",
        foreseen_step: int = 0,
        latency_ms: float = 0.0,
        poses: Optional[List[HandPose]] = None,
        bboxes: Optional[List[BoundingBox3D]] = None,
        gripper_cmd: float = 0.0,
        residuals: Optional[List[float]] = None,
        reward_score: float = 0.0,
        discrepancy_norm: float = 0.0,
        adaptation_active: bool = True,
        buffer_steps: int = 0,
        is_recording: bool = False,
        recorded_frames: int = 0,
        robot_connected: bool = True
    ) -> None:
        """Render the status HUD: a minimal glance card by default, or the full
        telemetry dock when expanded via 'h'."""
        if self.show_telemetry_detail:
            self._draw_telemetry_expanded(
                frame=frame, fps=fps, tracker_name=tracker_name, workflow_phase=workflow_phase,
                parsed_intent=parsed_intent, voice_status=voice_status, latency_ms=latency_ms,
                poses=poses, gripper_cmd=gripper_cmd, reward_score=reward_score,
                discrepancy_norm=discrepancy_norm, adaptation_active=adaptation_active,
                is_recording=is_recording, recorded_frames=recorded_frames, robot_connected=robot_connected,
            )
        else:
            self._draw_telemetry_compact(
                frame=frame, fps=fps, latency_ms=latency_ms, workflow_phase=workflow_phase,
                parsed_intent=parsed_intent, poses=poses, is_recording=is_recording,
                recorded_frames=recorded_frames,
            )

    def _draw_telemetry_compact(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        workflow_phase: ExecutionPhase,
        parsed_intent: Optional[ParsedIntent],
        poses: Optional[List[HandPose]],
        is_recording: bool,
        recorded_frames: int,
    ) -> None:
        """Minimal glance card - just health, stage, and target. No walls of text."""
        h, w = frame.shape[:2]
        card_w, card_h = 208, 80
        x1 = w - card_w - 10
        y1 = 10
        x2, y2 = x1 + card_w, y1 + card_h

        self._glass_panel(frame, x1, y1, x2, y2, alpha=0.80, radius=14)

        phase_colors = {
            ExecutionPhase.IDLE: PALETTE["text_dim"],
            ExecutionPhase.FORESEEING: PALETTE["amber_gold"],
            ExecutionPhase.WAIT_USER: PALETTE["text_white"],
            ExecutionPhase.USER_EXECUTING: PALETTE["neon_green"],
            ExecutionPhase.ADAPTING: PALETTE["neon_violet"],
        }
        p_col = phase_colors.get(workflow_phase, PALETTE["text_dim"])

        cv2.circle(frame, (x1 + 16, y1 + 22), 4, p_col, -1, cv2.LINE_AA)
        stage_label = workflow_phase.value.replace("_", " ").title()
        cv2.putText(frame, stage_label, (x1 + 28, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, PALETTE["text_white"], 1, cv2.LINE_AA)

        fps_col = PALETTE["neon_green"] if fps >= 20 else PALETTE["amber_gold"]
        hand_pct = f"{poses[0].confidence*100:.0f}%" if poses else "-"
        cv2.circle(frame, (x1 + 16, y1 + 44), 3, fps_col, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:4.1f} fps  -  {latency_ms:3.0f} ms  -  hand {hand_pct}",
                    (x1 + 26, y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.31, PALETTE["text_dim"], 1, cv2.LINE_AA)

        target_obj = parsed_intent.target_object if parsed_intent and parsed_intent.is_active else None
        if target_obj:
            cv2.putText(frame, f"target - {target_obj[:18]}", (x1 + 16, y1 + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, PALETTE["amber_gold"], 1, cv2.LINE_AA)
        elif is_recording:
            cv2.circle(frame, (x1 + 18, y1 + 62), 3, PALETTE["laser_red"], -1, cv2.LINE_AA)
            cv2.putText(frame, f"rec - {recorded_frames} frames", (x1 + 26, y1 + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.31, PALETTE["laser_red"], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "'h' details  -  'i' intent", (x1 + 16, y1 + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, PALETTE["text_dim"], 1, cv2.LINE_AA)

    def _draw_telemetry_expanded(
        self,
        frame: np.ndarray,
        fps: float,
        tracker_name: str,
        workflow_phase: ExecutionPhase,
        parsed_intent: Optional[ParsedIntent],
        voice_status: str,
        latency_ms: float,
        poses: Optional[List[HandPose]],
        gripper_cmd: float,
        reward_score: float,
        discrepancy_norm: float,
        adaptation_active: bool,
        is_recording: bool,
        recorded_frames: int,
        robot_connected: bool,
    ) -> None:
        """Full telemetry dock, sized to its content - opt-in detail view (press 'h')."""
        h, w = frame.shape[:2]

        dock_w = 225
        dock_h = 372
        x1 = w - dock_w - 10
        y1 = 10
        x2 = w - 10
        y2 = y1 + dock_h

        self._glass_panel(frame, x1, y1, x2, y2, alpha=0.84, radius=16)

        # 1. Title & Status
        cv2.putText(frame, "PRECOGNITION", (x1 + 12, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["cyan_electric"], 1, cv2.LINE_AA)

        # Status Pill: FPS & Latency
        fps_col = PALETTE["neon_green"] if fps >= 20 else PALETTE["amber_gold"]
        cv2.circle(frame, (x1 + 16, y1 + 42), 4, fps_col, -1)
        cv2.putText(frame, f"{fps:4.1f} fps  -  {latency_ms:3.0f} ms", (x1 + 26, y1 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["text_white"], 1, cv2.LINE_AA)

        # Divider 1
        cv2.line(frame, (x1 + 12, y1 + 58), (x2 - 12, y1 + 58), (40, 50, 65), 1)

        # 2. Stage & Intent
        phase_colors = {
            ExecutionPhase.IDLE: PALETTE["text_dim"],
            ExecutionPhase.FORESEEING: PALETTE["amber_gold"],
            ExecutionPhase.WAIT_USER: PALETTE["text_white"],
            ExecutionPhase.USER_EXECUTING: PALETTE["neon_green"],
            ExecutionPhase.ADAPTING: PALETTE["neon_violet"]
        }
        p_col = phase_colors.get(workflow_phase, PALETTE["text_dim"])
        cv2.putText(frame, f"stage - {workflow_phase.value}", (x1 + 12, y1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.35, p_col, 1, cv2.LINE_AA)

        target_obj = parsed_intent.target_object if parsed_intent and parsed_intent.is_active else "standby"
        cv2.putText(frame, f"target - {target_obj[:16]}", (x1 + 12, y1 + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.35, PALETTE["amber_gold"], 1, cv2.LINE_AA)

        v_color = PALETTE["cyan_electric"] if voice_status == "LISTENING" else PALETTE["text_dim"]
        v_label = "voice - listening..." if voice_status == "LISTENING" else "voice - talk ['v']"
        cv2.putText(frame, v_label, (x1 + 12, y1 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.33, v_color, 1, cv2.LINE_AA)

        # Divider 2
        cv2.line(frame, (x1 + 12, y1 + 124), (x2 - 12, y1 + 124), (40, 50, 65), 1)

        # 3. Telemetry & Policy Residuals
        cv2.putText(frame, "Residual Adaptation", (x1 + 12, y1 + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.35, PALETTE["text_dim"], 1, cv2.LINE_AA)

        adapt_str = "online ['p']" if adaptation_active else "paused ['p']"
        adapt_col = PALETTE["neon_green"] if adaptation_active else PALETTE["laser_red"]
        cv2.putText(frame, f"learning - {adapt_str}", (x1 + 12, y1 + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.32, adapt_col, 1, cv2.LINE_AA)

        r_col = PALETTE["neon_green"] if reward_score > 0.5 else (PALETTE["amber_gold"] if reward_score > 0.0 else PALETTE["laser_red"])
        cv2.putText(frame, f"reward - {reward_score:+0.2f}", (x1 + 12, y1 + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.34, r_col, 1, cv2.LINE_AA)

        cv2.putText(frame, f"error - {discrepancy_norm:.4f}", (x1 + 12, y1 + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["text_white"], 1, cv2.LINE_AA)

        # Gripper Actuator Progress Bar
        cv2.putText(frame, "gripper", (x1 + 12, y1 + 222), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["text_white"], 1, cv2.LINE_AA)
        bar_x = x1 + 70
        bar_y = y1 + 213
        bar_w = dock_w - 92
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 45, 55), -1)
        fill_w = int(bar_w * np.clip(gripper_cmd, 0.0, 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), PALETTE["cyan_electric"], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), PALETTE["glass_border"], 1)

        # Divider 3
        cv2.line(frame, (x1 + 12, y1 + 234), (x2 - 12, y1 + 234), (40, 50, 65), 1)

        # 4. Hardware & Vision Sensor Status
        cv2.putText(frame, "Hardware & Sensors", (x1 + 12, y1 + 252), cv2.FONT_HERSHEY_SIMPLEX, 0.35, PALETTE["text_dim"], 1, cv2.LINE_AA)

        rob_str = "robot - 7-DOF ok" if robot_connected else "robot - offline"
        cv2.putText(frame, rob_str, (x1 + 12, y1 + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.33, PALETTE["neon_green"] if robot_connected else PALETTE["laser_red"], 1, cv2.LINE_AA)

        track_disp = "mediapipe (live)" if "MEDIAPIPE" in tracker_name else "mock synthetic"
        cv2.putText(frame, f"track - {track_disp}", (x1 + 12, y1 + 288), cv2.FONT_HERSHEY_SIMPLEX, 0.32, PALETTE["cyan_electric"], 1, cv2.LINE_AA)

        if poses and len(poses) > 0:
            p = poses[0]
            cv2.putText(frame, f"hand - tracked ({p.confidence*100:.0f}%)", (x1 + 12, y1 + 308), cv2.FONT_HERSHEY_SIMPLEX, 0.33, PALETTE["neon_green"], 1, cv2.LINE_AA)
            cv2.putText(frame, f"xyz - [{p.keypoints_3d[0,0]:+.2f},{p.keypoints_3d[0,1]:+.2f},{p.keypoints_3d[0,2]:+.2f}]",
                        (x1 + 12, y1 + 326), cv2.FONT_HERSHEY_SIMPLEX, 0.31, PALETTE["text_white"], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "hand - searching...", (x1 + 12, y1 + 308), cv2.FONT_HERSHEY_SIMPLEX, 0.33, PALETTE["amber_gold"], 1, cv2.LINE_AA)
            cv2.putText(frame, "raise hand to camera", (x1 + 12, y1 + 326), cv2.FONT_HERSHEY_SIMPLEX, 0.30, PALETTE["text_dim"], 1, cv2.LINE_AA)

        if is_recording:
            cv2.circle(frame, (x1 + 15, y1 + 342), 3, PALETTE["laser_red"], -1, cv2.LINE_AA)
            cv2.putText(frame, f"rec [{recorded_frames}f]", (x1 + 24, y1 + 346), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["laser_red"], 1, cv2.LINE_AA)

        # Bottom Shortcut Hints
        cv2.line(frame, (x1 + 12, y1 + 350), (x2 - 12, y1 + 350), (40, 50, 65), 1)
        cv2.putText(frame, "'h':Collapse | 'c':Step | 'm':Stats", (x1 + 10, y1 + 366), cv2.FONT_HERSHEY_SIMPLEX, 0.29, PALETTE["text_dim"], 1, cv2.LINE_AA)

    # Ordered to match the README hotkey cheat sheet.
    HOTKEY_LEGEND = [
        ("ENTER/c", "Step Phase"),
        ("h", "Telemetry"),
        ("m", "Co-Adapt"),
        ("k / ^S", "Save Ckpt"),
        ("l / ^L", "Load Ckpt"),
        ("x / ^R", "Reset"),
        ("v/SPACE", "Voice PTT"),
        ("g", "Voice Guide"),
        ("i", "Cycle Intent"),
        ("p", "Toggle Adapt"),
        ("r", "Record"),
        ("f", "Ghost Hand"),
        ("t", "Toggle Tracker"),
        ("b", "3D Box"),
        ("d", "Depth PIP"),
        ("s", "Screenshot"),
        ("z", "Fullscreen"),
        ("q/ESC", "Quit"),
    ]

    def draw_hotkey_panel(self, frame: np.ndarray, top_y: int) -> None:
        """Render the always-on hotkey cheat sheet docked below the glance card on the right side."""
        h, w = frame.shape[:2]
        panel_w = 218
        line_h = 14
        panel_h = 26 + line_h * len(self.HOTKEY_LEGEND) + 8
        x1 = w - panel_w - 10
        y1 = top_y
        x2 = w - 10
        y2 = y1 + panel_h

        self._glass_panel(frame, x1, y1, x2, y2, alpha=0.80, radius=14)
        cv2.putText(frame, "HOTKEYS", (x1 + 12, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["cyan_electric"], 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 12, y1 + 28), (x2 - 12, y1 + 28), (40, 50, 65), 1)

        for idx, (key, action) in enumerate(self.HOTKEY_LEGEND):
            row_y = y1 + 44 + idx * line_h
            cv2.putText(frame, key, (x1 + 12, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, PALETTE["amber_gold"], 1, cv2.LINE_AA)
            cv2.putText(frame, action, (x1 + 84, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, PALETTE["text_white"], 1, cv2.LINE_AA)


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

        cv2.putText(frame, "SYNTHETIC VIDEO STREAM (Mac CPU Mode)", (self.width // 2 - 160, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

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
        transcriber_type: str = "mock"
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
        
        # Phase 6, 7 & 8 components
        if transcriber_type == "whisper":
            self.transcriber: AudioTranscriberABC = WhisperTranscriber()
        else:
            self.transcriber: AudioTranscriberABC = MockTranscriber()
        
        self.intent_parser: IntentParserABC = MockLLMIntentParser()
        self.speaker: SpeechSynthesizerABC = SystemSpeaker()
        self.workflow = WorkflowController(
            foresee_steps=60, wait_user_timeout=2.0, auto_advance=True,
            # In remote mode the server owns the authoritative phase; the client announces
            # phase changes itself from the server's snapshot (see _network_step), so the
            # local WorkflowController must stay silent to avoid double/conflicting speech.
            speaker=(self.speaker if self.mode != "mock_remote" else None),
            voice_guidance_enabled=True
        )
        self.robot = MockRobotHardware(dof=7)
        self.checkpoint_manager = PolicyCheckpointManager()
        self.benchmark = CoAdaptationBenchmark()
        self.safety_monitor = SafetyMonitor(dof=7)
        
        self.voice_status = "IDLE"
        self.current_parsed_intent = self.intent_parser.parse_intent(self.intent)
        self.last_episode_report: Optional[EpisodeDiscrepancyReport] = None
        self._control_cmd_to_send: Optional[str] = None

        self.cap = None
        self.is_synthetic_camera = False
        self._audio_stream: Optional["sd.InputStream"] = None
        self._is_fullscreen = False

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
        self.local_scene_parser = MockSceneParser()
        self.local_affordance_extractor = MockAffordanceExtractor()
        self.local_trajectory_diffusion = MockTrajectoryDiffusion()
        self.local_discrepancy_engine = DiscrepancyEngine()
        self.local_physics_engine = MockPhysicsEngine()
        self.local_policy = MockResidualPolicy()
        self._last_action = np.zeros(7, dtype=np.float32)
        self._cached_foreseen_traj = None
        self._local_learned_wrist_bias = np.zeros(3, dtype=np.float32)

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
        self._last_announced_phase: Optional[ExecutionPhase] = None
        self._remote_snapshot = {
            "poses": [], "bboxes": [], "affordance_map": None, "foreseen_traj": None,
            "depth_heatmap": None, "gripper_cmd": 0.0, "residuals": None, "reward_score": 0.0,
            "discrepancy_norm": 0.0, "buffer_steps": 0, "parsed_intent": None,
            "workflow_phase": ExecutionPhase.IDLE, "phase_progress": 0.0,
            "benchmark_summary": None,
        }

    def toggle_voice_mode(self) -> None:
        """Toggle Push-To-Talk voice listening / transcription."""
        if not self.transcriber.is_listening:
            self.transcriber.start_listening()
            self.voice_status = "LISTENING"
            logger.info("Voice Mode: LISTENING... Speak your intent.")
        else:
            self.voice_status = "TRANSCRIBING"
            transcript = self.transcriber.stop_listening()
            if transcript:
                self.intent = transcript
                self.current_parsed_intent = self.intent_parser.parse_intent(transcript)
                self.workflow.trigger_intent(self.current_parsed_intent.target_object if self.current_parsed_intent.is_active else "none")
                logger.info(f"Voice Mode Transcribed: '{transcript}' -> Target: {self.current_parsed_intent.target_object}")
            self.voice_status = "IDLE"

    def toggle_voice_guidance(self) -> None:
        """Toggle spoken workflow guidance (announcements on each phase transition)."""
        self.workflow.voice_guidance_enabled = not self.workflow.voice_guidance_enabled
        if not self.workflow.voice_guidance_enabled:
            self.speaker.stop()
        logger.info(f"Voice Guidance: {'ON' if self.workflow.voice_guidance_enabled else 'MUTED'}")

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

    def toggle_fullscreen(self) -> None:
        """Toggle the visualizer window between windowed (resizable) and true fullscreen."""
        window_name = self.config.visualization.window_name
        self._is_fullscreen = not self._is_fullscreen
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self._is_fullscreen else cv2.WINDOW_NORMAL,
        )
        logger.info(f"Visualizer window: {'FULLSCREEN' if self._is_fullscreen else 'windowed'}")

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

    def setup_camera(self) -> None:
        """Initialize physical camera with AVFoundation on macOS or fallback to synthetic."""
        device_id = self.config.camera.device_id
        logger.info(f"Opening camera device {device_id}...")

        if sys.platform == "darwin":
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(device_id)
        else:
            self.cap = cv2.VideoCapture(device_id)

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(5):
                self.cap.read()
                time.sleep(0.02)

            logger.info(f"Camera opened successfully ({self.config.camera.width}x{self.config.camera.height} @ {self.config.camera.fps} FPS).")

    async def run(self) -> None:
        """Main application execution loop."""
        self.setup_camera()
        self.setup_microphone()
        cv2.namedWindow(self.config.visualization.window_name, cv2.WINDOW_NORMAL)
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
                    ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures % 30 == 1:
                        logger.warning(f"Camera frame drop detected ({consecutive_failures} frames). Retrying...")

                    # Only switch to synthetic camera if 150 consecutive frame drops occur (>5 seconds of failure)
                    if consecutive_failures >= 150 and not self.is_synthetic_camera and self.config.camera.use_synthetic_if_unavailable:
                        logger.warning("Camera unresponsive for 150 frames. Automatically switching to Synthetic Camera generator.")
                        if self.cap:
                            self.cap.release()
                        self.cap = SyntheticCamera(
                            width=self.config.camera.width,
                            height=self.config.camera.height,
                            fps=self.config.camera.fps
                        )
                        self.is_synthetic_camera = True
                        consecutive_failures = 0

                    await asyncio.sleep(0.005)
                    continue

                consecutive_failures = 0
                frame_id += 1

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
                        
                        prompt_for_scene = self.current_parsed_intent.target_object if self.current_parsed_intent.is_active else self.intent
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
                                    learned_bias=self._local_learned_wrist_bias
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
                        rep = self.local_discrepancy_engine.compile_episode_discrepancy(
                            foreseen_traj=self.workflow.stored_foreseen_trajectory,
                            recorded_poses=self.workflow.recorded_physical_poses,
                            policy=self.local_policy
                        )
                        self.last_episode_report = rep
                        self.benchmark.record_trial(rep, intent=self.intent)
                        episode_offset = np.clip(
                            np.array(rep.mean_wrist_offset, dtype=np.float32), -0.05, 0.05
                        )
                        self._local_learned_wrist_bias = np.clip(
                            0.6 * self._local_learned_wrist_bias + 0.4 * episode_offset, -0.05, 0.05
                        )
                        self.workflow.transition_to(ExecutionPhase.RESTARTING)
                        self._cached_foreseen_traj = None
                    elif self.workflow.current_phase == ExecutionPhase.RESTARTING:
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
                    gripper_cmd = snap["gripper_cmd"]
                    residuals = snap["residuals"]
                    reward_score = snap["reward_score"]
                    discrepancy_norm = snap["discrepancy_norm"]
                    buffer_steps = snap["buffer_steps"]
                    parsed_intent_resp = snap["parsed_intent"] or self.current_parsed_intent
                    workflow_phase = snap["workflow_phase"]
                    phase_progress = snap["phase_progress"]
                    if snap["benchmark_summary"]:
                        benchmark_summary = snap["benchmark_summary"]

                    # The server's workflow controller is headless (no speakers on the GPU
                    # pod), so the client announces phase changes locally instead.
                    if workflow_phase != self._last_announced_phase:
                        self._last_announced_phase = workflow_phase
                        if self.workflow.voice_guidance_enabled:
                            instruction = self.workflow._phase_instruction(workflow_phase)
                            if instruction:
                                self.speaker.speak(instruction)

                    if not self._network_got_first_response:
                        cv2.putText(frame, f"CONNECTING TO {self.server_url or self.config.network.server_host}...",
                                    (30, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                # Reset one-shot command (only relevant for mock_local, which sends none)
                self._control_cmd_to_send = None

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

                # Render Visualizations
                fps = self.visualizer.update_fps()
                self.visualizer.draw_hand_skeleton(frame, poses, residuals=residuals, adaptation_active=self.adaptation_active)
                self.visualizer.draw_3d_bounding_boxes(frame, bboxes)
                self.visualizer.draw_affordance_hotspots(frame, affordance_map)
                
                # The server continuously replans the trajectory from wherever the real
                # hand currently is while actively guiding a grasp (see ws_server.py) -
                # the ghost follows the hand, not the other way around. So render a
                # short near-term lookahead into whatever's the LATEST plan (a "here's
                # where to move next" nudge), not a long fixed animation played back
                # over elapsed time - that would fight the server's own continuous
                # replanning and drift away from the real hand.
                step_idx_to_draw: Optional[int] = None
                ghost_label = "NEXT-STEP GUIDE"
                if foreseen_traj is not None and foreseen_traj.waypoints:
                    num_wp = len(foreseen_traj.waypoints)
                    if workflow_phase in (ExecutionPhase.FORESEEING, ExecutionPhase.WAIT_USER, ExecutionPhase.USER_EXECUTING):
                        step_idx_to_draw = min(GHOST_LOOKAHEAD_STEPS, num_wp - 1)
                    elif workflow_phase == ExecutionPhase.ADAPTING:
                        step_idx_to_draw = num_wp - 1  # episode just ended - show the final grasp/lift pose as reference
                        ghost_label = "FINAL POSE REFERENCE"

                foreseen_step = self.visualizer.draw_foreseen_ghost_trajectory(
                    frame, foreseen_traj, step_override=step_idx_to_draw, real_poses=poses,
                    target_bbox=bboxes[0] if bboxes else None, label=ghost_label
                )
                self.visualizer.draw_depth_pip(frame, depth_heatmap)
                
                # Render Stage Banner
                self.visualizer.draw_workflow_banner(
                    frame=frame,
                    phase=workflow_phase,
                    progress=phase_progress,
                    step_idx=self.workflow.step_index,
                    discrepancy_norm=discrepancy_norm,
                    episode_report=self.last_episode_report
                )
                self.visualizer.draw_instruction_bar(
                    frame=frame,
                    phase=workflow_phase,
                    target_label=(parsed_intent_resp.target_object if parsed_intent_resp else self.intent)
                )

                # Render Co-Adaptation Benchmark Panel
                self.visualizer.draw_coadaptation_panel(frame, benchmark_summary)

                label = "MOCK LOCAL" if self.mode == "mock_local" else f"MOCK REMOTE ({self.config.network.server_host}:{self.config.network.server_port})"
                self.visualizer.draw_telemetry_hud(
                    frame=frame,
                    fps=fps,
                    mode_str=label,
                    tracker_name=self.tracker_name,
                    intent=self.intent,
                    workflow_phase=workflow_phase,
                    phase_progress=phase_progress,
                    parsed_intent=parsed_intent_resp,
                    voice_status=self.voice_status,
                    foreseen_step=foreseen_step,
                    latency_ms=latency_ms,
                    poses=poses,
                    bboxes=bboxes,
                    gripper_cmd=gripper_cmd,
                    residuals=residuals,
                    reward_score=reward_score,
                    discrepancy_norm=discrepancy_norm,
                    adaptation_active=self.adaptation_active,
                    buffer_steps=buffer_steps,
                    is_recording=self.recorder.is_recording,
                    recorded_frames=self.recorder.frame_count,
                    robot_connected=self.robot.is_connected
                )

                # Always-on hotkey cheat sheet, docked below the glance card (skipped when the
                # full telemetry dock is expanded, since it already fills the right column).
                if not self.visualizer.show_telemetry_detail:
                    self.visualizer.draw_hotkey_panel(frame, top_y=100)

                # Display frame window
                cv2.imshow(self.config.visualization.window_name, frame)

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
                elif key == ord("g"): # Toggle spoken workflow guidance
                    self.toggle_voice_guidance()
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
            if self.cap:
                self.cap.release()
            self.robot.disconnect()
            self.speaker.stop()
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
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    app_config = AppConfig.from_yaml(config_path)
    if args.device is not None:
        app_config.camera.device_id = args.device

    runner = LocalClientRunner(
        config=app_config,
        cli_mode=args.mode,
        tracker_type=args.tracker,
        intent=args.intent,
        enable_profiling=args.profile,
        enable_recording=args.record,
        server_url=args.server_url,
        transcriber_type=args.transcriber
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
