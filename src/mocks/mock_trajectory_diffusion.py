"""Synthetic Diffusion Policy reference rollout generator mock."""

import math
from typing import List, Optional
import numpy as np

from src.perception.hand_tracker import HandPose
from src.perception.scene_parser import BoundingBox3D
from src.simulation.simulator import SimState
from src.simulation.trajectory_generator import (
    AffordanceMap,
    ForeseenTrajectory,
    ForeseenWaypoint,
    Trajectory,
    TrajectoryGeneratorABC,
    Waypoint,
)


def minimum_jerk_step(t: float) -> float:
    """Standard 5th-order minimum jerk polynomial s(t) in [0, 1]."""
    t_clamped = np.clip(t, 0.0, 1.0)
    return float(10.0 * (t_clamped ** 3) - 15.0 * (t_clamped ** 4) + 6.0 * (t_clamped ** 5))


class MockTrajectoryDiffusion(TrajectoryGeneratorABC):
    """
    Simulates a Visuomotor Diffusion Policy (e.g. Octo / 3D Diffusion Policy / DP3).
    Generates a 60-step kinematically stable 'foreseen' reference trajectory
    tau_ref = {q_t^hand, q_t^obj}_{t=1}^60 moving the hand from initial position
    toward target contact affordance hotspots, grasping, and lifting the object.
    """

    def __init__(self, camera_shape: tuple = (480, 640)) -> None:
        self.cam_h, self.cam_w = camera_shape
        self.fx = self.fy = 0.8 * self.cam_w
        self.cx = self.cam_w / 2.0
        self.cy = self.cam_h / 2.0

        # Canonical relative 21-joint finger offsets
        self._finger_roots = np.array([
            [-0.030, -0.020, 0.020], # Thumb
            [-0.025, -0.085, 0.010], # Index
            [-0.005, -0.090, 0.010], # Middle
            [ 0.018, -0.085, 0.010], # Ring
            [ 0.040, -0.075, 0.010], # Pinky
        ], dtype=np.float32)

        self._seg_lens = [0.035, 0.026, 0.020] # 3 phalanges

    def _generate_hand_keypoints_3d(
        self,
        wrist_pos: np.ndarray,
        wrist_rot: np.ndarray,
        finger_flex: float
    ) -> np.ndarray:
        """Synthesize 21 3D joint locations given wrist pose and finger flexion factor."""
        kpts = np.zeros((21, 3), dtype=np.float32)
        kpts[0] = wrist_pos

        # Rotation matrix from Euler
        rx, ry, rz = wrist_rot
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        joint_idx = 1
        for f_idx in range(5):
            root = wrist_pos + R @ self._finger_roots[f_idx]
            kpts[joint_idx] = root
            prev = root
            joint_idx += 1

            flex = finger_flex * (1.1 if f_idx in (0, 1) else 1.0)
            for seg_i, length in enumerate(self._seg_lens):
                cur_flex = flex * (seg_i + 1) * 0.8
                # Finger extension & curl in local frame
                local_dir = np.array([
                    0.005 * math.sin(cur_flex),
                    -length * math.cos(cur_flex),
                    length * math.sin(cur_flex)
                ], dtype=np.float32)
                nxt = prev + R @ local_dir
                kpts[joint_idx] = nxt
                prev = nxt
                joint_idx += 1

        return kpts

    def _project_2d(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """Project (21, 3) 3D keypoints to (21, 2) image plane coordinates."""
        kpts_2d = np.zeros((len(keypoints_3d), 2), dtype=np.float32)
        valid_z = np.clip(keypoints_3d[:, 2], 0.1, 10.0)
        kpts_2d[:, 0] = self.fx * (keypoints_3d[:, 0] / valid_z) + self.cx
        kpts_2d[:, 1] = self.fy * (keypoints_3d[:, 1] / valid_z) + self.cy
        return kpts_2d

    def generate_foreseen_rollout(
        self,
        start_hand_pose: Optional[HandPose],
        target_object: BoundingBox3D,
        affordance_map: AffordanceMap,
        intent: str = "foresee me picking this remote control",
        num_steps: int = 60,
        learned_bias: Optional[np.ndarray] = None
    ) -> ForeseenTrajectory:
        """
        Generate a 60-frame kinematically stable reference trajectory tau_ref.

        learned_bias: (3,) accumulated mean (real - foreseen) wrist offset from prior
        completed episodes for this session (see DiscrepancyEngine.compile_episode_
        discrepancy). Shifts the suggested grasp point toward how this user has
        actually been moving, so the plan visibly improves across iterations instead
        of suggesting the same generic approach every time.
        """
        # Determine start wrist position
        if start_hand_pose is not None and len(start_hand_pose.keypoints_3d) > 0:
            p_start = start_hand_pose.keypoints_3d[0].copy()
            rot_start = start_hand_pose.mano_params.wrist_rotation.copy() if start_hand_pose.mano_params else np.zeros(3, dtype=np.float32)
        else:
            p_start = np.array([0.08, 0.08, 0.48], dtype=np.float32)
            rot_start = np.zeros(3, dtype=np.float32)

        bias = learned_bias if learned_bias is not None else np.zeros(3, dtype=np.float32)

        # Target grasp wrist position derived from object center & affordance
        obj_center = target_object.center.copy()
        # Position hand slightly behind and above target object, nudged by whatever
        # this user has demonstrated in prior attempts on this same grasp.
        p_grasp = obj_center + np.array([0.0, -0.03, -0.06], dtype=np.float32) + bias
        rot_grasp = np.array([0.25, 0.0, 0.1], dtype=np.float32)

        # Post-grasp lifted position
        p_lift = p_grasp + np.array([0.0, -0.09, 0.02], dtype=np.float32)

        waypoints: List[ForeseenWaypoint] = []
        dt = 2.0 / num_steps # 2.0 seconds total

        for step in range(1, num_steps + 1):
            t_frac = (step - 1) / float(num_steps - 1) # 0.0 to 1.0
            time_offset = step * dt

            # Three kinematic phases:
            # 1. Approach & Pre-Grasp (0.0 to 0.40)
            # 2. Enclosure & Contact (0.40 to 0.65)
            # 3. Lift & Manipulation (0.65 to 1.0)
            if t_frac <= 0.40:
                sub_t = minimum_jerk_step(t_frac / 0.40)
                wrist_pos = p_start + sub_t * (p_grasp - p_start)
                wrist_rot = rot_start + sub_t * (rot_grasp - rot_start)
                finger_flex = 0.2 * sub_t # Fingers open wide
                obj_pos = np.concatenate([obj_center, target_object.rotation])
                contact_val = 0.0
                gripper = 0.0

            elif t_frac <= 0.65:
                sub_t = minimum_jerk_step((t_frac - 0.40) / 0.25)
                wrist_pos = p_grasp
                wrist_rot = rot_grasp
                finger_flex = 0.2 + 0.65 * sub_t # Fingers enclose object
                obj_pos = np.concatenate([obj_center, target_object.rotation])
                contact_val = float(sub_t)
                gripper = float(sub_t)

            else:
                sub_t = minimum_jerk_step((t_frac - 0.65) / 0.35)
                wrist_pos = p_grasp + sub_t * (p_lift - p_grasp)
                wrist_rot = rot_grasp
                finger_flex = 0.85 # Firm grasp hold
                # Object moves rigidly attached to hand
                lifted_obj_center = obj_center + sub_t * (p_lift - p_grasp)
                obj_pos = np.concatenate([lifted_obj_center, target_object.rotation])
                contact_val = 1.0
                gripper = 1.0

            # Forward kinematics for 21 joints
            kpts_3d = self._generate_hand_keypoints_3d(wrist_pos, wrist_rot, finger_flex)
            kpts_2d = self._project_2d(kpts_3d)

            # 5-fingertip contact vector
            contact_state = np.full(5, contact_val, dtype=np.float32)

            wp = ForeseenWaypoint(
                timestep=step,
                time_offset=time_offset,
                hand_keypoints_3d=kpts_3d,
                hand_keypoints_2d=kpts_2d,
                wrist_pose=np.concatenate([wrist_pos, wrist_rot]),
                object_pose=obj_pos,
                contact_state=contact_state,
                gripper_aperture=gripper
            )
            waypoints.append(wp)

        return ForeseenTrajectory(
            intent=intent,
            target_label=target_object.label,
            waypoints=waypoints,
            duration=2.0
        )

    def plan(self, start_state: SimState, goal_pose: np.ndarray) -> Trajectory:
        """Legacy plan interface."""
        wp = Waypoint(target_pose=goal_pose, timestamp=1.0)
        return Trajectory(waypoints=[wp], duration=1.0)
