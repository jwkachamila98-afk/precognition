"""Headless integration test verifying client-server WebSocket streaming with intent and 3D bounding boxes."""

import numpy as np
import pytest
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_policy import MockPolicy
from src.transport.ws_server import WSInferenceServer
from src.transport.ws_client import WSStreamingClient


@pytest.mark.asyncio
async def test_client_server_websocket_pipeline():
    server_port = 8798
    server = WSInferenceServer(
        host="127.0.0.1",
        port=server_port,
        hand_tracker=MockHandTracker(),
        depth_estimator=MockDepthEstimator(),
        scene_parser=MockSceneParser(),
        policy=MockPolicy()
    )

    # Start server
    await server.start()

    # Create client
    client = WSStreamingClient(
        host="127.0.0.1",
        port=server_port,
        compression_quality=75
    )

    try:
        # Dummy video frame
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        intent = "foresee me picking this remote control"

        # Stream 5 test frames
        for frame_idx in range(1, 6):
            response = await client.send_frame(test_frame, frame_id=frame_idx, intent=intent)
            assert response is not None
            assert response.frame_id == frame_idx
            assert len(response.hand_poses) > 0
            assert response.depth_heatmap_base64 is not None
            assert response.policy_residuals is not None

            # Verify hand poses
            poses = response.get_hand_poses()
            assert len(poses) == 1
            assert poses[0].keypoints_3d.shape == (21, 3)

            # Verify depth heatmap thumbnail
            depth_img = response.decode_depth_heatmap()
            assert depth_img is not None
            assert depth_img.ndim == 3

            # Verify 3D parsed scene and bounding primitives
            parsed = response.get_parsed_scene()
            assert parsed is not None
            assert parsed.intent == intent
            assert len(parsed.bounding_boxes) == 1
            assert parsed.bounding_boxes[0].label == "remote_control"
            assert parsed.bounding_boxes[0].corners_3d.shape == (8, 3)
    finally:
        await client.close()
        await server.stop()
