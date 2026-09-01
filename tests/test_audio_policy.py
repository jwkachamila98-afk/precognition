"""Audio output is notification tones, in every mode (tests/test_audio_policy.py).

Spoken guidance was removed because it arrived a second or two after the
moment it described and talked over someone concentrating on a reach. That
reasoning is about timing and attention, not about which TTS backend is
used - but the removal was only applied to the remote path. A local session
still narrated itself through macOS `say` ("Go. Reach for the remote control
now."), while a remote one played a tone, and the tone cue itself was nested
inside the remote-only branch of the render loop.
"""

import inspect
import types

from apps.local_client import LocalClientRunner
from src.policy.workflow_state import ExecutionPhase


def _runner():
    """A runner stub carrying only what the phase cue actually touches."""
    r = LocalClientRunner.__new__(LocalClientRunner)
    played = []
    r.sounds = types.SimpleNamespace(play=lambda cue: played.append(cue),
                                     enabled=True)
    r.workflow = types.SimpleNamespace(voice_guidance_enabled=True)
    r._training_target_announced = None
    return r, played


def test_arriving_in_a_phase_makes_a_sound():
    for phase in (ExecutionPhase.FORESEEING, ExecutionPhase.WAIT_USER,
                  ExecutionPhase.USER_EXECUTING, ExecutionPhase.ADAPTING,
                  ExecutionPhase.AUTONOMOUS_DEMO):
        r, played = _runner()
        r._sound_phase_change(phase, ExecutionPhase.IDLE)
        assert played, f"arriving in {phase} made no sound"


def test_the_phase_cue_is_not_gated_on_the_remote_mode():
    """The cue used to live inside `elif self.mode == "mock_remote":`, so a
    local session never played one.

    Nesting is what matters, and a substring search cannot see it - the mode
    branch appears earlier in run() than the cue does either way. Indentation
    can: a statement inside that branch is indented deeper than the branch
    header itself.
    """
    lines = inspect.getsource(LocalClientRunner.run).splitlines()
    indent = lambda s: len(s) - len(s.lstrip())

    branch = next(l for l in lines if 'elif self.mode == "mock_remote":' in l)
    # The cue's own `if`, not the call inside it - the call is one level
    # deeper by construction, whichever block it lives in.
    guard = next(l for l in lines
                 if "if workflow_phase != self._last_announced_phase" in l)

    assert indent(guard) <= indent(branch), (
        "the phase cue is nested inside the remote-mode branch again; local "
        "sessions would fall silent there")


def test_nothing_constructs_a_talking_speaker():
    """No backend that vocalises may be wired into the client - the objection
    was to speech itself, not to one vendor's implementation."""
    src = inspect.getsource(__import__("apps.local_client", fromlist=["x"]))
    for talking in ("SystemSpeaker(", "GeminiSpeaker("):
        assert talking not in src, f"{talking} is back in the client"
