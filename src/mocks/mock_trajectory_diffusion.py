"""Synthetic Diffusion Policy reference rollout generator mock."""

import math
from typing import List, Optional
import numpy as np

from src.perception.action_schema import ActionPlan
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
        self._fingertips = [4, 8, 12, 16, 20]

        # Wrist orientation at the grasp: rolled ~163 deg about the camera X
        # axis so the palm faces DOWN over the object and the fingers point at
        # it. The canonical pose in _generate_hand_keypoints_3d has the fingers
        # extending along local -Y (i.e. straight up out of the wrist), which is
        # the pose a hand holds when it is NOT reaching for anything.
        # Yaw is not cosmetic: at zero yaw the hand closes edge-on to the
        # viewer and every finger hides behind the one in front of it.
        self._rot_grasp = np.array([2.85, 0.60, 0.12], dtype=np.float32)

        # Thumb opposition. In the canonical pose every digit extends along
        # local -Y, so the thumb is just a shorter finger lying in the same
        # plane as the others - it can approach an object but can never meet
        # them around it, which is why the hand could only ever rest on the
        # manipuland. A real thumb's metacarpal is rotated across the palm; this
        # rotates the thumb's whole chain about the palm normal so its extension
        # and curl carry it toward the fingertips it has to oppose.
        _TH = math.radians(-62.0)
        self._thumb_oppose = np.array([
            [math.cos(_TH), -math.sin(_TH), 0.0],
            [math.sin(_TH), math.cos(_TH), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        # 0.85 curled the fingers past a right angle into a closed fist, which
        # cannot enclose anything; 0.55 closes firmly on the object while still
        # showing daylight between the fingers.
        self._flex_closed = 0.55

        # How the wrist is oriented at the START of the approach, expressed as an
        # offset from the grasp orientation. A hand orients itself as it reaches;
        # it does not tumble on the way in.
        self._pre_grasp_offset = np.array([-0.28, -0.18, 0.0], dtype=np.float32)

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

            is_thumb = f_idx == 0
            flex = finger_flex * (1.35 if is_thumb else 1.0)
            # Abduction: fingers fan outward from the palm's midline as they
            # curl. Without it every finger sweeps the same plane and the whole
            # hand collapses into one slab the moment it closes.
            #
            # The thumb is the exception and it is what makes a grasp a grasp:
            # it curls ACROSS the palm toward the other fingertips rather than
            # away from them. With every digit fanning the same way the hand can
            # only ever rest on an object, never hold one - which is exactly how
            # the reenactment read.
            # Abduction is damped as the hand closes. The fan is what stops
            # every finger sweeping one plane and collapsing into a slab, but
            # left undamped it peaks exactly at contact - the fingers closed
            # around 14 cm of empty air either side of a 4 cm object. This does
            # not reverse the fan, it just keeps it from widening into the grip.
            flex_frac = float(np.clip(finger_flex / max(self._flex_closed, 1e-6), 0.0, 1.0))
            if is_thumb:
                # Opposition, conversely, grows as the hand closes.
                spread = (0.95 * abs(float(self._finger_roots[f_idx][0]))
                          * (0.25 + 0.75 * flex_frac))
            else:
                spread = (0.34 * float(self._finger_roots[f_idx][0])
                          * (1.0 - 0.80 * flex_frac))
            for seg_i, length in enumerate(self._seg_lens):
                cur_flex = flex * (seg_i + 1) * 0.8
                # Finger extension & curl in local frame
                local_dir = np.array([
                    0.005 * math.sin(cur_flex) + spread * math.sin(cur_flex),
                    -length * math.cos(cur_flex),
                    length * math.sin(cur_flex)
                ], dtype=np.float32)
                if is_thumb:
                    local_dir = self._thumb_oppose @ local_dir
                nxt = prev + R @ local_dir
                kpts[joint_idx] = nxt
                prev = nxt
                joint_idx += 1

        return kpts

    def _solve_grasp_wrist(self, obj_center: np.ndarray, approach_offset: np.ndarray) -> np.ndarray:
        """Wrist position that puts the closed fingertips ON the object.

        Placing the wrist by a hand-tuned offset from the object centre only
        looks right by accident: the hand is ~17 cm from wrist to fingertip, so
        an offset picked to look plausible in a flat 2-D overlay leaves the
        fingers grasping empty air the moment the same plan is viewed in 3D.
        Instead the canonical closed-hand pose is built once at the origin, the
        fingertip centroid measured, and the wrist placed so that centroid lands
        on the intended contact point.
        """
        probe = self._generate_hand_keypoints_3d(
            np.zeros(3, dtype=np.float32), self._rot_grasp, self._flex_closed)
        # The grasp centre is where the THUMB opposes the fingers, not the
        # centroid of all five tips. With the thumb crossing the palm the plain
        # centroid sits off to one side, which puts the object beside the hand
        # instead of between the thumb and fingers.
        pinch = 0.5 * (probe[4] + probe[[8, 12]].mean(axis=0))
        tip_offset = pinch - probe[0]
        return (obj_center + approach_offset - tip_offset).astype(np.float32)

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
        learned_bias: Optional[np.ndarray] = None,
        action: Optional[ActionPlan] = None
    ) -> ForeseenTrajectory:
        """
        Generate a 60-frame kinematically stable reference trajectory tau_ref.

        learned_bias: (3,) accumulated mean (real - foreseen) wrist offset from prior
        completed episodes for this session (see DiscrepancyEngine.compile_episode_
        discrepancy). Shifts the suggested grasp point toward how this user has
        actually been moving, so the plan visibly improves across iterations instead
        of suggesting the same generic approach every time.

        action: how the spoken verb should be carried out. Previously the parsed
        action was read by nobody and every utterance produced the same reach and
        lift, so "push the cup" and "pick up the cup" were indistinguishable. The
        plan describes the motion along axes this generator can execute -
        approach direction, what happens on contact, what follows - which is what
        lets an unseen phrase produce sensible motion. Passing None keeps the
        original pick-and-lift exactly as it was.
        """
        legacy = action is None
        plan = action or ActionPlan()
        # Determine start wrist position
        if start_hand_pose is not None and len(start_hand_pose.keypoints_3d) > 0:
            p_start = start_hand_pose.keypoints_3d[0].copy()
        else:
            # No hand observed: start from a READY STANDOFF relative to the object
            # rather than a fixed point in camera space. A hard-coded home pose is
            # only ever near the target by luck - when the object is somewhere else
            # the plan opens with a long traverse across empty space that carries no
            # information about the grasp, and dominates any view of it.
            # Camera frame: +X right, +Y down, +Z away, so this is up-and-left of
            # the object and nearer the camera.
            if legacy or plan.approach == "above":
                standoff = np.array([-0.05, -0.11, -0.04], dtype=np.float32)
            elif plan.approach == "side":
                standoff = np.array([-0.16, -0.03, -0.04], dtype=np.float32)
            else:
                standoff = np.array([-0.03, -0.05, -0.15], dtype=np.float32)
            p_start = target_object.center + standoff

        bias = learned_bias if learned_bias is not None else np.zeros(3, dtype=np.float32)

        # Target grasp wrist position derived from object center & affordance.
        obj_center = target_object.center.copy()
        # Contact point: the object's TOP SURFACE, not its centre. Aiming the
        # fingertips at the centre buries them inside a solid object - harmless
        # in a flat overlay, obviously wrong the moment the same plan is
        # rendered in 3D. Camera frame has +Y down, so -Y is up.
        size = np.asarray(target_object.size, dtype=np.float32)
        obj_half_h = float(size[1]) * 0.5
        if legacy or plan.approach == "above":
            # Down onto the top surface.
            approach_offset = np.array([0.0, -0.95 * obj_half_h, 0.0], dtype=np.float32)
            lateral = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif plan.approach == "side":
            # In from the left flank, at the object's mid height - the only way a
            # push or a slide reads as deliberate rather than as a failed grasp.
            approach_offset = np.array([-0.95 * float(size[0]) * 0.5, 0.0, 0.0],
                                       dtype=np.float32)
            lateral = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:                                            # "front", from the camera
            approach_offset = np.array([0.0, 0.0, -0.95 * float(size[2]) * 0.5],
                                       dtype=np.float32)
            lateral = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # A non-contacting action stops short of the surface rather than reaching
        # into it: pointing at something and touching it are different claims.
        if plan.contact == "none":
            approach_offset = approach_offset * 2.6
        # Nudged by whatever this user has demonstrated in prior attempts.
        p_grasp = self._solve_grasp_wrist(obj_center, approach_offset) + bias
        rot_grasp = self._rot_grasp.copy()

        # The approach begins ALREADY oriented for the grasp, offset only enough
        # to settle visibly. It used to begin at the identity rotation - or, with
        # a hand detected, at the tracker's wrist_rotation, which is a
        # [pitch, yaw, 0] triple derived from the palm direction and NOT the same
        # parameterisation as _rot_grasp, where rx = 2.85 encodes "inverted".
        # Either way the wrist interpolated ~163 degrees during the approach: the
        # hand entered upside down, fingertips 14 cm ABOVE the wrist, and
        # cartwheeled over on the way in.
        rot_start = rot_grasp + self._pre_grasp_offset

        # Post-grasp lifted position (-Y is up in the camera frame). 9 cm was
        # under 40 px on screen and easy to miss entirely; 14 cm reads clearly
        # against the locked camera without pushing the wrist out of frame.
        if legacy:
            flex_closed = self._flex_closed
            p_lift = p_grasp + np.array([0.0, -0.14, 0.02], dtype=np.float32)
            rot_end = rot_grasp
        else:
            # Grip 0..1 onto the flexion the hand model understands. 0.85 - the
            # schema's default for a plain grasp - lands on 0.58, just past the
            # 0.55 tuned by hand for the original pick-and-lift.
            flex_closed = float(np.clip(0.20 + 0.45 * plan.grip, 0.05, 0.75))
            travel = float(plan.travel_m)
            up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            toward_user = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            if plan.follow_through == "lift":
                delta = up * travel + np.array([0.0, 0.0, 0.02], dtype=np.float32)
            elif plan.follow_through == "toward_user":
                delta = toward_user * travel + up * (0.35 * travel)
            elif plan.follow_through == "slide":
                # Along the surface, not off it.
                delta = lateral * travel
            elif plan.follow_through == "tilt":
                delta = up * (0.25 * travel)
            elif plan.follow_through == "retreat":
                delta = (p_start - p_grasp) * 0.55
            else:                                        # "hold" / "none"
                delta = np.zeros(3, dtype=np.float32)
            p_lift = p_grasp + delta
            # Tilt is expressed on the wrist roll axis, which is what pouring,
            # drinking and unscrewing all actually are.
            rot_end = rot_grasp + np.array(
                [0.0, 0.0, np.radians(plan.tilt_deg)], dtype=np.float32)

        waypoints: List[ForeseenWaypoint] = []
        dt = 2.0 / num_steps # 2.0 seconds total

        for step in range(1, num_steps + 1):
            t_frac = (step - 1) / float(num_steps - 1) # 0.0 to 1.0
            time_offset = step * dt

            # Three kinematic phases. The split is weighted toward contact and
            # lift rather than approach: the approach carries no information
            # about the grasp, and at the frame rates this renders at, a lift
            # squeezed into the last third of the rollout goes by in a handful
            # of frames and reads as the object never having moved.
            # 1. Approach & Pre-Grasp (0.0 to 0.28)
            # 2. Enclosure & Contact (0.28 to 0.50)
            # 3. Lift & Manipulation (0.50 to 1.0)
            if t_frac <= 0.28:
                sub_t = minimum_jerk_step(t_frac / 0.28)
                wrist_pos = p_start + sub_t * (p_grasp - p_start)
                wrist_rot = rot_start + sub_t * (rot_grasp - rot_start)
                finger_flex = 0.2 * sub_t # Fingers open wide
                obj_pos = np.concatenate([obj_center, target_object.rotation])
                contact_val = 0.0
                gripper = 0.0

            elif t_frac <= 0.50:
                sub_t = minimum_jerk_step((t_frac - 0.28) / 0.22)
                wrist_pos = p_grasp
                wrist_rot = rot_grasp
                finger_flex = 0.2 + (flex_closed - 0.2) * sub_t # Fingers enclose object
                obj_pos = np.concatenate([obj_center, target_object.rotation])
                contact_val = float(sub_t)
                gripper = float(sub_t)

            else:
                sub_t = minimum_jerk_step((t_frac - 0.50) / 0.50)
                wrist_pos = p_grasp + sub_t * (p_lift - p_grasp)
                wrist_rot = rot_grasp + sub_t * (rot_end - rot_grasp)
                finger_flex = flex_closed # Firm grasp hold
                # The object only follows a hand that actually closed on it. A
                # push shoves it along; a touch or a point leaves it alone. It
                # used to be carried in every case, so pointing at a cup dragged
                # the cup through the air.
                carried = legacy or plan.contact in ("grasp", "pinch")
                shoved = (not legacy) and plan.contact == "push"
                if carried or shoved:
                    moved = obj_center + sub_t * (p_lift - p_grasp)
                else:
                    moved = obj_center
                obj_rot = np.asarray(target_object.rotation, dtype=np.float32).copy()
                if not legacy and plan.tilt_deg and carried and len(obj_rot) >= 3:
                    obj_rot[2] += float(np.radians(plan.tilt_deg)) * sub_t
                obj_pos = np.concatenate([moved, obj_rot])
                contact_val = 0.0 if (not legacy and plan.contact == "none") else 1.0
                gripper = float(plan.grip) if not legacy else 1.0

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
