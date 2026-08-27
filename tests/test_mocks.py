"""Unit tests for Phase 1 & 2 components, mock generators, and transport protocol."""

import unittest
import numpy as np
from config.config_parser import AppConfig
from src.perception.hand_tracker import HandSide
from src.perception.scene_parser import BoundingBox3D
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_simulator import MockSimulator
from src.mocks.mock_policy import MockPolicy
from src.simulation.simulator import SimAction
from src.policy.policy_base import PolicyObservation
from src.transport.protocol import (
    FrameMessage,
    InferenceResponse,
    encode_image_to_base64,
)


class TestMockPipeline(unittest.TestCase):

    def test_app_config_loading(self):
        config = AppConfig()
        self.assertIn(config.system.mode, ["mock_local", "mock_remote", "mock", "remote"])
        self.assertEqual(config.camera.fps, 30)
        self.assertEqual(config.perception.hand_tracker.num_keypoints, 21)
        self.assertEqual(config.system.get_normalized_mode(), "mock_local")

    def test_mock_hand_tracker(self):
        tracker = MockHandTracker(hand_side=HandSide.RIGHT)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        poses = tracker.estimate(dummy_frame)
        self.assertEqual(len(poses), 1)

        pose = poses[0]
        self.assertEqual(pose.side, HandSide.RIGHT)
        self.assertEqual(pose.keypoints_3d.shape, (21, 3))
        self.assertEqual(pose.keypoints_2d.shape, (21, 2))
        self.assertTrue(0.0 <= pose.confidence <= 1.0)
        self.assertIsNotNone(pose.mano_params)
        self.assertEqual(pose.mano_params.wrist_rotation.shape, (3,))
        self.assertEqual(pose.mano_params.joint_rotations.shape, (45,))

        u_coords = pose.keypoints_2d[:, 0]
        v_coords = pose.keypoints_2d[:, 1]
        self.assertTrue(np.all(u_coords > -100) and np.all(u_coords < 800))
        self.assertTrue(np.all(v_coords > -100) and np.all(v_coords < 600))

    def test_mock_depth_estimator(self):
        estimator = MockDepthEstimator(min_depth=0.2, max_depth=2.5, target_shape=(240, 320))
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        depth_map = estimator.estimate_depth(dummy_frame)
        self.assertEqual(depth_map.depth.shape, (240, 320))
        self.assertTrue(np.all(depth_map.depth >= 0.2))
        self.assertTrue(np.all(depth_map.depth <= 2.5))

        heatmap = depth_map.to_colored_heatmap()
        self.assertEqual(heatmap.shape, (240, 320, 3))
        self.assertEqual(heatmap.dtype, np.uint8)

    def test_mock_scene_parser(self):
        depth_est = MockDepthEstimator()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = depth_est.estimate_depth(dummy_frame)

        parser = MockSceneParser(num_points=300)

        # 1. Idle / Standby test (No object mentioned -> 0 bounding boxes)
        parsed_idle = parser.parse_scene(
            image=dummy_frame,
            depth=depth_map,
            intent="idle"
        )
        self.assertEqual(len(parsed_idle.bounding_boxes), 0)

        # 2. Active Intent test (Object mentioned -> extracts remote_control bounding box)
        parsed_active = parser.parse_scene(
            image=dummy_frame,
            depth=depth_map,
            intent="foresee me picking this remote control"
        )
        self.assertEqual(len(parsed_active.bounding_boxes), 1)
        bbox = parsed_active.bounding_boxes[0]
        self.assertEqual(bbox.label, "remote_control")
        self.assertEqual(bbox.center.shape, (3,))
        self.assertEqual(bbox.size.shape, (3,))
        self.assertEqual(bbox.corners_3d.shape, (8, 3))

        corners_2d = bbox.project_to_2d(image_shape=(480, 640))
        self.assertEqual(corners_2d.shape, (8, 2))

        self.assertIsNotNone(parsed_active.point_cloud)
        self.assertEqual(parsed_active.point_cloud.points.shape, (300, 3))

    def test_mock_simulator(self):
        sim = MockSimulator(num_dof=7)
        state = sim.reset()
        self.assertEqual(state.joint_positions.shape, (7,))
        self.assertEqual(state.ee_pose.shape, (6,))

        action = SimAction(target_joint_positions=np.array([0.5, 0.2, -0.1, 0.0, 0.4, 0.1, -0.2], dtype=np.float32))
        new_state = sim.step(action)
        self.assertEqual(new_state.joint_positions.shape, (7,))
        self.assertFalse(np.array_equal(new_state.joint_positions, state.joint_positions))

    def test_mock_policy(self):
        tracker = MockHandTracker()
        depth_est = MockDepthEstimator()
        policy = MockPolicy(action_dim=7)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = tracker.estimate(dummy_frame)
        depth = depth_est.estimate_depth(dummy_frame)

        obs = PolicyObservation(hand_poses=poses, depth_map=depth, timestamp=1.0)
        action = policy.act(obs)

        self.assertEqual(action.joint_residuals.shape, (7,))
        self.assertTrue(0.0 <= action.gripper_action <= 1.0)
        self.assertTrue(action.confidence > 0.9)

    def test_transport_serialization_roundtrip(self):
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        b64_str = encode_image_to_base64(test_img, quality=80)

        frame_msg = FrameMessage(
            frame_id=42,
            client_timestamp=100.5,
            image_base64=b64_str,
            width=100,
            height=100,
            intent="grasp cup"
        )

        json_str = frame_msg.to_json()
        reconstructed = FrameMessage.from_json(json_str)

        self.assertEqual(reconstructed.frame_id, 42)
        self.assertEqual(reconstructed.client_timestamp, 100.5)
        self.assertEqual(reconstructed.intent, "grasp cup")
        decoded_img = reconstructed.decode_image()
        self.assertEqual(decoded_img.shape, (100, 100, 3))


if __name__ == "__main__":
    unittest.main()
