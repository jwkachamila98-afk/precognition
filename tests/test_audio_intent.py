"""Unit and integration tests for Phase 6 Speech-to-Text and Structured LLM Intent Parsing."""

import numpy as np
import importlib.util
import sys

import pytest

from src.audio.speech_to_text import MockTranscriber, WhisperTranscriber
from src.perception.intent_parser import MockLLMIntentParser, ParsedIntent, StructuredLLMIntentParser
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.mocks.mock_policy import MockResidualPolicy
from src.policy.discrepancy import DiscrepancyEngine
from src.transport.ws_server import WSInferenceServer
from src.transport.ws_client import WSStreamingClient


def test_mock_transcriber():
    transcriber = MockTranscriber()
    assert not transcriber.is_listening

    transcriber.start_listening()
    assert transcriber.is_listening

    transcript = transcriber.stop_listening()
    assert not transcriber.is_listening
    assert len(transcript) > 0
    assert "remote" in transcript or "cup" in transcript or "bottle" in transcript


@pytest.mark.skipif(
    "torch" in sys.modules or importlib.util.find_spec("torch") is not None,
    reason=(
        "faster-whisper (ctranslate2) and torch each bundle their own OpenMP "
        "runtime, and loading both into one process aborts the interpreter on "
        "macOS - a hard native abort, not an exception, so it takes the whole "
        "test session with it. local_client.py documents the same conflict and "
        "avoids NeuralResidualPolicy for exactly this reason. pytest imports "
        "torch during collection, so this test cannot run in a session where "
        "torch is installed."
    ),
)
def test_whisper_transcriber_fallback():
    transcriber = WhisperTranscriber(model_size="tiny.en", device="cpu")
    transcriber.start_listening()
    assert transcriber.is_listening
    res = transcriber.stop_listening()
    assert not transcriber.is_listening
    assert isinstance(res, str)


def test_mock_llm_intent_parser():
    parser = MockLLMIntentParser()

    # 1. Complex cup instruction
    intent_1 = parser.parse_intent("grasp the red coffee cup by the handle")
    # The target keeps the words the user said, descriptors included. It used
    # to be rewritten through a canonical table ("water cup" -> "coffee cup"),
    # which substituted a different object and discarded the very adjectives
    # that let open-vocabulary grounding tell two cups apart.
    assert intent_1.target_object == "red coffee cup"
    assert "red" in intent_1.spatial_attributes
    assert intent_1.affordance_hotspot == "handle"
    assert intent_1.action_type == "reach_and_grasp"
    assert intent_1.is_active

    # 2. Remote control instruction
    intent_2 = parser.parse_intent("foresee me picking this remote control")
    assert intent_2.target_object == "remote control"
    assert intent_2.affordance_hotspot == "sides"
    assert intent_2.is_active

    # 3. Spatial attribute bottle instruction
    intent_3 = parser.parse_intent("pick up the tall water bottle on the right")
    assert intent_3.target_object == "tall water bottle"
    assert "tall" in intent_3.spatial_attributes
    assert "right" in intent_3.spatial_attributes
    assert intent_3.is_active

    # 4. Standby / Clear instruction
    intent_4 = parser.parse_intent("clear target and return to standby")
    assert intent_4.target_object == "none"
    assert intent_4.action_type == "clear_standby"
    assert not intent_4.is_active


def test_parsed_intent_serialization():
    intent = ParsedIntent(
        target_object="coffee cup",
        spatial_attributes=["red", "left"],
        action_type="reach_and_grasp",
        affordance_hotspot="handle",
        confidence_score=0.98,
        raw_transcript="grasp red cup"
    )

    d = intent.to_dict()
    assert d["target_object"] == "coffee cup"
    assert d["spatial_attributes"] == ["red", "left"]

    reconstructed = ParsedIntent.from_dict(d)
    assert reconstructed.target_object == "coffee cup"
    assert reconstructed.affordance_hotspot == "handle"
    assert reconstructed.confidence_score == 0.98


@pytest.mark.asyncio
async def test_phase6_websocket_e2e():
    port = 8795
    server = WSInferenceServer(
        host="127.0.0.1",
        port=port,
        hand_tracker=MockHandTracker(),
        depth_estimator=MockDepthEstimator(),
        intent_parser=MockLLMIntentParser(),
        scene_parser=MockSceneParser(),
        affordance_extractor=MockAffordanceExtractor(),
        trajectory_diffusion=MockTrajectoryDiffusion(),
        discrepancy_engine=DiscrepancyEngine(),
        physics_engine=MockPhysicsEngine(),
        policy=MockResidualPolicy()
    )

    await server.start()
    client = WSStreamingClient(host="127.0.0.1", port=port)

    try:
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Send frame with natural language voice transcript
        response = await client.send_frame(
            frame,
            frame_id=1,
            intent="grasp the red coffee cup by the handle"
        )

        assert response is not None
        assert response.frame_id == 1
        assert response.parsed_intent is not None

        parsed_intent = response.get_parsed_intent()
        assert parsed_intent is not None
        assert parsed_intent.target_object == "red coffee cup"
        assert parsed_intent.affordance_hotspot == "handle"
        assert "red" in parsed_intent.spatial_attributes

        # Verify parsed scene received target
        parsed_scene = response.get_parsed_scene()
        assert parsed_scene is not None
        assert len(parsed_scene.bounding_boxes) == 1
        assert parsed_scene.bounding_boxes[0].label == "coffee_mug"
    finally:
        await client.close()
        await server.stop()


def test_the_target_keeps_the_words_the_user_actually_said():
    """Saying "water cup" used to come back as "coffee cup".

    KNOWN_OBJECTS rewrote the spoken noun into a canonical name, so the system
    went looking for a different object than the one named. That mapping existed
    because detection could only name eighty COCO classes and every phrase had
    to be forced onto one of them; with open-vocabulary detection the user's own
    words are the better query, and the descriptors are what tell two similar
    objects apart.
    """
    parser = MockLLMIntentParser()
    cases = {
        "I'm going to pick up this water cup and drink from it": "water cup",
        "grab the red coffee cup": "red coffee cup",
        "push the small black remote aside": "small black remote",
        "hand me the spoon": "spoon",
        "point at the houseplant": "houseplant",
        "take the utensil holder": "utensil holder",
        "foresee me picking this remote control": "remote control",
    }
    for utterance, expected in cases.items():
        got = parser.parse_intent(utterance).target_object
        assert got == expected, f"{utterance!r} -> {got!r}, expected {expected!r}"


def test_a_longer_known_name_wins_over_a_shorter_one():
    """"remote" must not beat "remote control" just by being earlier in the table."""
    parser = MockLLMIntentParser()
    assert parser.parse_intent("the remote control please").target_object == "remote control"


def test_filler_words_are_never_treated_as_the_object():
    parser = MockLLMIntentParser()
    for utterance in ("pick up this thing", "grab that one", "take it"):
        target = parser.parse_intent(utterance).target_object
        assert target in ("none", ""), f"{utterance!r} -> {target!r}"
