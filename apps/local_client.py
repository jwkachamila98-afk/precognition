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
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import cv2
import numpy as np

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


class LocalVisualizer:
    """Renders modern glassmorphism HUD, glowing hand skeletons, holographic ghost trajectories, and sci-fi telemetry."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.show_depth_inset = config.visualization.draw_depth_inset
        self.show_bounding_box = config.visualization.draw_bounding_box
        self.show_foreseen_ghost = True
        self.show_analytics_panel = False
        self.fps_history = collections.deque(maxlen=30)
        self._last_tick = time.perf_counter()
        self.anim_frame_idx = 0

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

    def draw_foreseen_ghost_trajectory(
        self,
        frame: np.ndarray,
        trajectory: Optional[ForeseenTrajectory],
        step_override: Optional[int] = None
    ) -> int:
        """Render animated holographic Foreseen Ghost Hand rollout with particle comets."""
        if not self.show_foreseen_ghost or trajectory is None or not trajectory.waypoints:
            return 0

        h, w = frame.shape[:2]
        num_steps = len(trajectory.waypoints)
        step_idx = (step_override if step_override is not None else self.anim_frame_idx) % num_steps
        current_wp: ForeseenWaypoint = trajectory.waypoints[step_idx]

        # 1. Shimmering Trajectory Comet Ribbon
        trail_pts = []
        for wp in trajectory.waypoints:
            u, v = wp.hand_keypoints_2d[0]
            trail_pts.append((int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))))

        if len(trail_pts) > 1:
            for i in range(len(trail_pts) - 1):
                alpha_frac = i / float(len(trail_pts))
                trail_color = (
                    int(255 * (1.0 - alpha_frac * 0.5)),
                    int(240 * alpha_frac),
                    int(80 * (1.0 - alpha_frac))
                )
                cv2.line(frame, trail_pts[i], trail_pts[i + 1], trail_color, 1 + int(alpha_frac * 2), lineType=cv2.LINE_AA)

        # 2. Holographic Dotted Ghost Hand Pose
        ghost_kpts_2d = current_wp.hand_keypoints_2d
        ghost_color = (255, 235, 100) # Bright Ice Cyan

        for u, v in HAND_CONNECTIONS:
            pt1 = (int(np.clip(ghost_kpts_2d[u, 0], 0, w - 1)), int(np.clip(ghost_kpts_2d[u, 1], 0, h - 1)))
            pt2 = (int(np.clip(ghost_kpts_2d[v, 0], 0, w - 1)), int(np.clip(ghost_kpts_2d[v, 1], 0, h - 1)))
            cv2.line(frame, pt1, pt2, ghost_color, thickness=2, lineType=cv2.LINE_AA)

        for j_idx in range(21):
            pt = (int(np.clip(ghost_kpts_2d[j_idx, 0], 0, w - 1)), int(np.clip(ghost_kpts_2d[j_idx, 1], 0, h - 1)))
            cv2.circle(frame, pt, 4, (10, 30, 45), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        wrist_pt = (int(np.clip(ghost_kpts_2d[0, 0], 0, w - 1)), int(np.clip(ghost_kpts_2d[0, 1], 0, h - 1)))
        cv2.putText(frame, f"HOLOGRAM tau_ref [{step_idx + 1}/60]", (wrist_pt[0] - 60, wrist_pt[1] - 14),
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

        # Fast ROI frosted glass blending (avoids full-frame memory copies)
        sub_overlay = frame[by1:by2, bx1:bx2].copy()
        cv2.rectangle(sub_overlay, (0, 0), (banner_w, banner_h), PALETTE["dark_glass_bg"], -1)
        cv2.addWeighted(sub_overlay, 0.85, frame[by1:by2, bx1:bx2], 0.15, 0, frame[by1:by2, bx1:bx2])

        t = time.time()
        pulse = 0.5 + 0.5 * np.sin(t * 8.0)

        if phase == ExecutionPhase.FORESEEING:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), PALETTE["amber_gold"], 1)
            cv2.circle(frame, (bx1 + 16, by1 + 16), int(4 + 2 * pulse), PALETTE["amber_gold"], -1, cv2.LINE_AA)
            msg = f"[1/3] FORESEEING ROLLOUT  •  {int(progress*100)}%"
            cv2.putText(frame, msg, (bx1 + 28, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PALETTE["amber_gold"], 1, cv2.LINE_AA)

        elif phase == ExecutionPhase.WAIT_USER:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), PALETTE["cyan_electric"], 1)
            cv2.circle(frame, (bx1 + 16, by1 + 16), 5, PALETTE["cyan_electric"], -1, cv2.LINE_AA)
            msg = "READY: PRESS 'c' TO EXECUTE MOTION"
            cv2.putText(frame, msg, (bx1 + 28, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        elif phase == ExecutionPhase.USER_EXECUTING:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), PALETTE["neon_green"], 1)
            cv2.circle(frame, (bx1 + 16, by1 + 16), int(4 + 2 * pulse), PALETTE["neon_green"], -1, cv2.LINE_AA)
            msg = f"[2/3] TRACKING EXECUTION  •  {int(progress*100)}%"
            cv2.putText(frame, msg, (bx1 + 28, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["neon_green"], 1, cv2.LINE_AA)

            # High precision alignment bar
            bar_x = bx1 + 28
            bar_y = by1 + 24
            bar_w = banner_w - 42
            bar_h = 3
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 50, 60), -1)
            fill_w = int(bar_w * np.clip(1.0 - (discrepancy_norm / 0.10), 0.0, 1.0))
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), PALETTE["neon_green"], -1)

        elif phase == ExecutionPhase.ADAPTING or episode_report is not None:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), PALETTE["neon_violet"], 1)
            cv2.circle(frame, (bx1 + 16, by1 + 16), 5, PALETTE["neon_violet"], -1, cv2.LINE_AA)
            rew = episode_report.episode_reward if episode_report else 0.0
            msg = f"[3/3] RESIDUAL ADAPTED ({rew:+.2f} R)"
            cv2.putText(frame, msg, (bx1 + 28, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["neon_violet"], 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), PALETTE["glass_border"], 1)
            cv2.circle(frame, (bx1 + 16, by1 + 16), 3, (120, 130, 145), -1, cv2.LINE_AA)
            msg = "STANDBY  •  PRESS 'i' OR TALK ('v')"
            cv2.putText(frame, msg, (bx1 + 28, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, PALETTE["text_dim"], 1, cv2.LINE_AA)

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

        # Fast ROI frosted glass blending
        sub_overlay = frame[py1:py2, px1:px2].copy()
        cv2.rectangle(sub_overlay, (0, 0), (panel_w, panel_h), PALETTE["dark_glass_bg"], -1)
        cv2.addWeighted(sub_overlay, 0.90, frame[py1:py2, px1:px2], 0.10, 0, frame[py1:py2, px1:px2])
        cv2.rectangle(frame, (px1, py1), (px2, py2), PALETTE["cyan_electric"], 1)

        cv2.putText(frame, "CO-ADAPTATION BENCHMARK", (px1 + 12, py1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, PALETTE["cyan_electric"], 1, cv2.LINE_AA)

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
        """Render cyber-sleek frosted glass telemetry dock tucked strictly along the right edge."""
        h, w = frame.shape[:2]

        dock_w = 225
        dock_h = h - 20
        x1 = w - dock_w - 10
        y1 = 10
        x2 = w - 10
        y2 = y1 + dock_h

        # Fast ROI frosted glass blending
        sub_overlay = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(sub_overlay, (0, 0), (dock_w, dock_h), PALETTE["dark_glass_bg"], -1)
        cv2.addWeighted(sub_overlay, 0.84, frame[y1:y2, x1:x2], 0.16, 0, frame[y1:y2, x1:x2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), PALETTE["glass_border"], 1)
        
        # Header Top Accent
        cv2.line(frame, (x1, y1), (x2, y1), PALETTE["cyan_electric"], 2)

        # 1. Title & Status
        cv2.putText(frame, "PRECOGNITION", (x1 + 10, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["cyan_electric"], 1, cv2.LINE_AA)
        
        # Status Pill: FPS & Latency
        fps_col = PALETTE["neon_green"] if fps >= 20 else PALETTE["amber_gold"]
        cv2.circle(frame, (x1 + 14, y1 + 40), 4, fps_col, -1)
        cv2.putText(frame, f"LIVE {fps:4.1f} FPS | {latency_ms:3.0f}ms", (x1 + 24, y1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 245, 255), 1, cv2.LINE_AA)

        # Divider 1
        cv2.line(frame, (x1 + 10, y1 + 54), (x2 - 10, y1 + 54), (40, 50, 65), 1)

        # 2. Stage & Intent
        phase_colors = {
            ExecutionPhase.IDLE: PALETTE["text_dim"],
            ExecutionPhase.FORESEEING: PALETTE["amber_gold"],
            ExecutionPhase.WAIT_USER: (255, 255, 255),
            ExecutionPhase.USER_EXECUTING: PALETTE["neon_green"],
            ExecutionPhase.ADAPTING: PALETTE["neon_violet"]
        }
        p_col = phase_colors.get(workflow_phase, PALETTE["text_dim"])
        cv2.putText(frame, f"STAGE: {workflow_phase.value}", (x1 + 10, y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.35, p_col, 1, cv2.LINE_AA)

        # Intent
        target_obj = parsed_intent.target_object if parsed_intent and parsed_intent.is_active else "standby"
        cv2.putText(frame, f"TARGET: {target_obj.upper()[:14]}", (x1 + 10, y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, PALETTE["amber_gold"], 1, cv2.LINE_AA)

        # Voice Push-To-Talk
        v_color = PALETTE["cyan_electric"] if voice_status == "LISTENING" else PALETTE["text_dim"]
        v_label = "VOICE: LISTENING..." if voice_status == "LISTENING" else "VOICE: TALK ['v']"
        cv2.putText(frame, v_label, (x1 + 10, y1 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.33, v_color, 1, cv2.LINE_AA)

        # Divider 2
        cv2.line(frame, (x1 + 10, y1 + 120), (x2 - 10, y1 + 120), (40, 50, 65), 1)

        # 3. Telemetry & Policy Residuals
        cv2.putText(frame, "◈ RESIDUAL ADAPTATION", (x1 + 10, y1 + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 220), 1, cv2.LINE_AA)
        
        adapt_str = "ONLINE PPO ['p']" if adaptation_active else "PAUSED ['p']"
        adapt_col = PALETTE["neon_green"] if adaptation_active else PALETTE["laser_red"]
        cv2.putText(frame, f"LEARNING: {adapt_str}", (x1 + 10, y1 + 156), cv2.FONT_HERSHEY_SIMPLEX, 0.32, adapt_col, 1, cv2.LINE_AA)

        # Step Reward
        r_col = PALETTE["neon_green"] if reward_score > 0.5 else (PALETTE["amber_gold"] if reward_score > 0.0 else PALETTE["laser_red"])
        cv2.putText(frame, f"REWARD R_t: {reward_score:+0.2f}", (x1 + 10, y1 + 176), cv2.FONT_HERSHEY_SIMPLEX, 0.34, r_col, 1, cv2.LINE_AA)

        # Trajectory Discrepancy
        cv2.putText(frame, f"ERROR D_traj: {discrepancy_norm:.4f}", (x1 + 10, y1 + 196), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 245, 255), 1, cv2.LINE_AA)

        # Gripper Actuator Progress Bar
        cv2.putText(frame, "GRIPPER:", (x1 + 10, y1 + 218), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["text_white"], 1, cv2.LINE_AA)
        bar_x = x1 + 75
        bar_y = y1 + 209
        bar_w = dock_w - 95
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 45, 55), -1)
        fill_w = int(bar_w * np.clip(gripper_cmd, 0.0, 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), PALETTE["cyan_electric"], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), PALETTE["glass_border"], 1)

        # Divider 3
        cv2.line(frame, (x1 + 10, y1 + 230), (x2 - 10, y1 + 230), (40, 50, 65), 1)

        # 4. Hardware & Vision Sensor Status
        cv2.putText(frame, "◈ HARDWARE & SENSORS", (x1 + 10, y1 + 248), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 220), 1, cv2.LINE_AA)

        rob_str = "ROBOT: 7-DOF [OK]" if robot_connected else "ROBOT: OFFLINE"
        cv2.putText(frame, rob_str, (x1 + 10, y1 + 266), cv2.FONT_HERSHEY_SIMPLEX, 0.33, PALETTE["neon_green"] if robot_connected else PALETTE["laser_red"], 1, cv2.LINE_AA)

        # Tracker Name
        track_disp = "MEDIAPIPE (LIVE)" if "MEDIAPIPE" in tracker_name else "MOCK SYNTHETIC"
        cv2.putText(frame, f"TRACK: {track_disp}", (x1 + 10, y1 + 284), cv2.FONT_HERSHEY_SIMPLEX, 0.32, PALETTE["cyan_electric"], 1, cv2.LINE_AA)

        # Hand Pose Status
        if poses and len(poses) > 0:
            p = poses[0]
            cv2.putText(frame, f"HAND: TRACKED ({p.confidence*100:.0f}%)", (x1 + 10, y1 + 304), cv2.FONT_HERSHEY_SIMPLEX, 0.33, PALETTE["neon_green"], 1, cv2.LINE_AA)
            cv2.putText(frame, f"XYZ: [{p.keypoints_3d[0,0]:+.2f},{p.keypoints_3d[0,1]:+.2f},{p.keypoints_3d[0,2]:+.2f}]", 
                        (x1 + 10, y1 + 322), cv2.FONT_HERSHEY_SIMPLEX, 0.31, PALETTE["text_white"], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "HAND: SEARCHING...", (x1 + 10, y1 + 304), cv2.FONT_HERSHEY_SIMPLEX, 0.33, PALETTE["amber_gold"], 1, cv2.LINE_AA)
            cv2.putText(frame, "Raise hand to camera", (x1 + 10, y1 + 322), cv2.FONT_HERSHEY_SIMPLEX, 0.30, PALETTE["text_dim"], 1, cv2.LINE_AA)

        # Recording Status
        if is_recording:
            cv2.putText(frame, f"● REC [{recorded_frames}f]", (x1 + 10, y1 + 346), cv2.FONT_HERSHEY_SIMPLEX, 0.34, PALETTE["laser_red"], 1, cv2.LINE_AA)

        # Bottom Shortcut Hints
        cv2.line(frame, (x1 + 10, y2 - 32), (x2 - 10, y2 - 32), (40, 50, 65), 1)
        cv2.putText(frame, "'c':Step | 'i':Intent | 'm':Stats", (x1 + 8, y2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, PALETTE["text_dim"], 1, cv2.LINE_AA)


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
        self.workflow = WorkflowController(foresee_steps=60, wait_user_timeout=2.0, auto_advance=True)
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

        # Network client
        self.ws_client = WSStreamingClient(
            host=config.network.server_host,
            port=config.network.server_port,
            server_url=server_url,
            compression_quality=config.network.compression_quality,
            timeout=config.network.timeout_seconds
        )

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
                            if self._cached_foreseen_traj is None or self.workflow.current_phase == ExecutionPhase.FORESEEING:
                                self._cached_foreseen_traj = self.local_trajectory_diffusion.generate_foreseen_rollout(
                                    start_hand_pose=start_h,
                                    target_object=target_box,
                                    affordance_map=affordance_map,
                                    intent=self.intent,
                                    num_steps=60
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
                        self.workflow.transition_to(ExecutionPhase.IDLE)
                        self._cached_foreseen_traj = None

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
                    # Stage 7: WebSocket Serialization & Network Transport
                    with self.profiler.profile("6. WebSocket Transport"):
                        t0 = time.perf_counter()
                        response = await self.ws_client.send_frame(frame, frame_id, intent=self.intent)
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                    if response:
                        srv_poses = response.get_hand_poses()
                        if isinstance(self.active_tracker, MediaPipeHandTracker):
                            local_poses = self.active_tracker.estimate(frame)
                            poses = local_poses if local_poses else srv_poses
                        else:
                            poses = srv_poses
                        depth_heatmap = response.decode_depth_heatmap()
                        gripper_cmd = response.gripper_action
                        residuals = response.policy_residuals
                        reward_score = response.reward_score
                        discrepancy_norm = response.discrepancy_norm
                        buffer_steps = response.buffer_step_count
                        parsed_scene = response.get_parsed_scene()
                        if parsed_scene:
                            bboxes = parsed_scene.bounding_boxes
                        parsed_intent_resp = response.get_parsed_intent() or self.current_parsed_intent
                        affordance_map = response.get_affordance_map()
                        foreseen_traj = response.get_foreseen_trajectory()
                        try:
                            workflow_phase = ExecutionPhase(response.workflow_phase)
                        except ValueError:
                            workflow_phase = ExecutionPhase.IDLE
                        phase_progress = response.phase_progress
                        rep = response.get_episode_report()
                        if rep:
                            self.last_episode_report = rep
                        if response.benchmark_summary:
                            benchmark_summary = response.benchmark_summary
                    else:
                        cv2.putText(frame, f"AWAITING WS SERVER ({self.config.network.server_host}:{self.config.network.server_port})...", 
                                    (30, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                # Reset one-shot command
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
                
                step_idx_to_draw = self.workflow.step_index if workflow_phase == ExecutionPhase.FORESEEING else None
                foreseen_step = self.visualizer.draw_foreseen_ghost_trajectory(frame, foreseen_traj, step_override=step_idx_to_draw)
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

                # Maintain 30 FPS yielding to asyncio
                elapsed = time.perf_counter() - t_frame_start
                target_dt = 1.0 / self.config.camera.fps
                sleep_time = max(0.001, target_dt - elapsed)
                await asyncio.sleep(sleep_time)

        finally:
            if self.recorder.is_recording:
                self.recorder.stop_recording()
            if self.cap:
                self.cap.release()
            self.robot.disconnect()
            cv2.destroyAllWindows()
            await self.ws_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visuomotor Hand Policy Local Client (Intel Mac CPU Friendly)")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Path to system_config.yaml")
    parser.add_argument("--mode", type=str, choices=["mock_local", "mock_remote", "mock", "remote"], default=None, help="Execution mode (mock_local | mock_remote)")
    parser.add_argument("--tracker", type=str, choices=["mediapipe", "mock"], default=None, help="Hand tracker type")
    parser.add_argument("--transcriber", type=str, choices=["mock", "whisper"], default="mock", help="Audio transcriber engine")
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
