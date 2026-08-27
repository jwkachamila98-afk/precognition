# Visuomotor Hand Policy Architecture: API Specification & Wire Protocols

This document details the complete WebSocket JSON message schemas, structured payload definitions, and telemetry data models across the 9 phases of the Visuomotor Hand Policy Architecture.

---

## 1. Outbound Client Frame Message (`FrameMessage`)

Transmitted from `local_client.py` to `remote_server.py` over WebSocket connection at 30 FPS.

```json
{
  "frame_id": 1420,
  "client_timestamp": 1724773120.452,
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wA...",
  "width": 640,
  "height": 480,
  "intent": "grasp the red coffee cup by the handle",
  "workflow_phase": "USER_EXECUTING",
  "control_command": null,
  "compression": "jpeg",
  "msg_type": "frame_request"
}
```

### Field Definitions:
- `frame_id` *(int)*: Monotonically increasing client frame sequence index.
- `client_timestamp` *(float)*: Unix epoch timestamp in seconds at camera capture.
- `image_base64` *(str)*: Base64-encoded compressed JPEG video frame ($640 \times 480$ BGR).
- `intent` *(str)*: Natural language user instruction or speech transcription transcript.
- `workflow_phase` *(str)*: Current state machine phase (`IDLE`, `FORESEEING`, `WAIT_USER`, `USER_EXECUTING`, `ADAPTING`).
- `control_command` *(Optional[str])*: Control signal override (`SAVE_CHECKPOINT`, `LOAD_CHECKPOINT`, `RESET_BASELINE`, `ADVANCE_PHASE`).

---

## 2. Inbound Server Telemetry Payload (`InferenceResponse`)

Returned from `remote_server.py` to `local_client.py` after real-time perception, intent reasoning, diffusion rollout, discrepancy compilation, safety verification, and robot hardware command execution.

```json
{
  "frame_id": 1420,
  "client_timestamp": 1724773120.452,
  "server_timestamp": 1724773120.468,
  "hand_poses": [
    {
      "hand_id": 0,
      "side": "right",
      "keypoints_3d": [[0.05, 0.12, 0.48], "...21 points [X,Y,Z] in meters..."],
      "keypoints_2d": [[320.0, 240.0], "...21 points [u,v] in pixels..."],
      "confidence": 0.96,
      "mano_params": {
        "wrist_translation": [0.05, 0.12, 0.48],
        "wrist_rotation": [0.0, 0.0, 0.0],
        "joint_rotations": [0.0, 0.0, "...45 MANO pose angles in rad..."],
        "shape_betas": [0.0, "...10 beta parameters..."]
      }
    }
  ],
  "depth_heatmap_base64": "/9j/4AAQSkZJRg...",
  "parsed_intent": {
    "target_object": "coffee cup",
    "spatial_attributes": ["red"],
    "action_type": "grasp",
    "affordance_hotspot": "handle",
    "confidence": 0.95,
    "is_active": true
  },
  "parsed_scene": {
    "bounding_boxes": [
      {
        "center": [0.08, 0.12, 0.58],
        "size": [0.10, 0.10, 0.12],
        "orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "label": "coffee cup",
        "confidence": 0.92
      }
    ]
  },
  "affordance_map": {
    "target_label": "coffee cup",
    "hotspots": [[0.08, 0.12, 0.58]],
    "grasp_direction": [0.0, 0.0, -1.0],
    "approach_vector": [0.0, 0.0, -1.0]
  },
  "foreseen_trajectory": {
    "intent": "grasp the red coffee cup by the handle",
    "target_label": "coffee cup",
    "duration": 2.0,
    "waypoints": [
      {
        "timestep": 0,
        "time_offset": 0.0,
        "hand_keypoints_3d": [[0.0, 0.0, 0.5], "..."],
        "hand_keypoints_2d": [[320, 240], "..."],
        "wrist_pose": [0.05, 0.10, 0.50, 0.0, 0.0, 0.0],
        "object_pose": [0.08, 0.12, 0.58, 0.0, 0.0, 0.0],
        "contact_state": [0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper_aperture": 0.0
      }
    ]
  },
  "policy_residuals": [-0.012, 0.005, 0.024, -0.008, 0.015, -0.002, 0.031],
  "reward_score": 0.842,
  "discrepancy_norm": 0.0142,
  "workflow_phase": "USER_EXECUTING",
  "phase_progress": 0.65,
  "episode_report": null,
  "robot_state": {
    "joint_positions": [0.038, 0.105, 0.524, -0.008, 0.015, -0.002, 0.031],
    "joint_velocities": [0.12, -0.04, 0.08, 0.0, 0.05, 0.0, 0.02],
    "joint_efforts": [0.35, 0.82, -0.15, 0.0, 0.12, 0.0, 0.05],
    "gripper_aperture": 0.35,
    "is_connected": true,
    "is_e_stopped": false
  },
  "safety_status": {
    "is_safe": true,
    "is_e_stopped": false,
    "warning_flags": [],
    "clamped_joint_positions": [0.038, 0.105, 0.524, -0.008, 0.015, -0.002, 0.031],
    "clamped_joint_velocities": [0.12, -0.04, 0.08, 0.0, 0.05, 0.0, 0.02],
    "min_obstacle_clearance_meters": 0.084,
    "heartbeat_latency_ms": 16.2
  },
  "benchmark_summary": {
    "total_trials": 3,
    "error_reduction_pct": 42.5,
    "mean_reward": 0.782,
    "latest_error_mm": 18.4,
    "initial_error_mm": 32.0
  },
  "adaptation_status": "ACTIVE",
  "buffer_step_count": 65,
  "policy_loss": 0.024,
  "gripper_action": 0.35,
  "server_processing_ms": 16.2,
  "msg_type": "inference_response"
}
```

---

## 3. Structured Data Models

### 3.1 `112-Dimensional State Vector s_t`
Constructed by `DiscrepancyEngine` for residual policy forward inference:
- **Indices 0..62 (63-dim)**: Keypoint tracking error vectors $(\mathbf{p}_{\text{real}}^{(i)} - \mathbf{p}_{\text{sim}}^{(i)}) \in \mathbb{R}^{21 \times 3}$.
- **Indices 63..68 (6-dim)**: Real wrist translation and orientation.
- **Indices 69..74 (6-dim)**: Simulated reference wrist translation and orientation.
- **Indices 75..84 (10-dim)**: Target object 3D bounding box center, dimensions, and yaw.
- **Indices 85..89 (5-dim)**: 5-fingertip contact state flags $c_t \in \{0, 1\}$.
- **Indices 90..96 (7-dim)**: Previous residual action $\Delta \theta_{t-1}$.
- **Indices 97..103 (7-dim)**: Instantaneous joint velocities $\dot{q}_t$.
- **Indices 104..110 (7-dim)**: Instantaneous joint tracking discrepancies.
- **Index 111 (1-dim)**: Normalized trajectory progress $t / T \in [0, 1]$.

### 3.2 `EpisodeDiscrepancyReport`
Compiled upon completion of each physical manipulation execution:
- `mean_pose_error`: Cumulative MSE error in meters between $\tau_{\text{sim}}$ and $\tau_{\text{real}}$.
- `max_pose_error`: Peak Euclidean divergence in meters.
- `smoothness_variance`: 2nd-order motion jerk variance.
- `contact_misalignment`: Final contact distance error in meters.
- `episode_reward`: Aggregate scalar reward $R_{\text{episode}} \in [-1.0, 1.0]$.
- `loss_delta`: PPO policy update gradient loss improvement.

---

## 4. Safety Guardrail Thresholds

| Parameter | Threshold Value | Violation Action |
|:---|:---|:---|
| **Max Joint Velocity** | $\le 2.0\text{ rad/s}$ | Velocity saturation clamping |
| **Max Joint Acceleration** | $\le 8.0\text{ rad/s}^2$ | Rate-limited low-pass filter |
| **Workspace X Bounds** | $[-0.60, +0.60]\text{ m}$ | Hard boundary clamping |
| **Workspace Y Bounds** | $[-0.50, +0.50]\text{ m}$ | Hard boundary clamping |
| **Workspace Z Bounds** | $[+0.10, +0.90]\text{ m}$ | Hard boundary clamping |
| **Obstacle Clearance** | $\ge 0.020\text{ m}$ ($2.0\text{ cm}$) | Soft stop / collision alert |
| **Heartbeat Dropout** | $\le 0.250\text{ s}$ ($250\text{ ms}$) | Emergency velocity ramp down |
