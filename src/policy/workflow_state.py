"""Staged 'Foresee-then-Execute' Workflow State Machine and Lifecycle Controller."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np

from src.audio.text_to_speech import SpeechSynthesizerABC
from src.perception.hand_tracker import HandPose
from src.simulation.trajectory_generator import ForeseenTrajectory

logger = logging.getLogger(__name__)


class ExecutionPhase(str, Enum):
    """Execution lifecycle phases for the Foresee-then-Execute paradigm."""
    IDLE = "IDLE"                         # Listening for voice / text intent
    FORESEEING = "FORESEEING"             # Simulating and visualizing 60-step ghost hand rollout
    WAIT_USER = "WAIT_USER"               # Rollout complete; waiting for user trigger to begin physical move
    USER_EXECUTING = "USER_EXECUTING"     # Tracking real physical hand vs reference waypoints
    ADAPTING = "ADAPTING"                 # Compiling cumulative trajectory discrepancy & updating residual policy
    RESTARTING = "RESTARTING"             # Brief pause before looping back into FORESEEING with the
                                           # updated co-adaptation bias, so the same grasp visibly improves
    AUTONOMOUS_DEMO = "AUTONOMOUS_DEMO"   # On-demand, hands-off simulated pick: replans fresh from
                                           # wherever the object currently is and runs the trained
                                           # residual policy's correction over it - no real hand needed


class WorkflowControlSignal(str, Enum):
    """Control commands sent across transport to trigger phase transitions."""
    START_FORESEE = "START_FORESEE"
    START_USER_EXECUTION = "START_USER_EXECUTION"
    FINISH_EPISODE = "FINISH_EPISODE"
    RESET_IDLE = "RESET_IDLE"
    ADVANCE_PHASE = "ADVANCE_PHASE"
    START_AUTONOMOUS_DEMO = "START_AUTONOMOUS_DEMO"


@dataclass
class WorkflowState:
    """Snapshot of current workflow state machine status."""
    phase: ExecutionPhase = ExecutionPhase.IDLE
    phase_progress: float = 0.0          # 0.0 to 1.0 progress within current phase
    step_index: int = 0                  # Current waypoint / execution frame index
    total_steps: int = 60                # Total expected steps in phase
    phase_elapsed_sec: float = 0.0       # Elapsed time in current phase
    intent_active: bool = False
    target_label: str = "none"


class WorkflowController:
    """
    Manages the 5-phase Foresee-then-Execute lifecycle:
    IDLE -> FORESEEING -> WAIT_USER -> USER_EXECUTING -> ADAPTING -> IDLE
    """

    def __init__(
        self,
        foresee_steps: int = 60,
        foresee_duration_sec: float = 3.5,
        wait_user_timeout: float = 3.0,
        execution_max_steps: int = 90,
        adapting_duration_sec: float = 3.5,
        restart_delay_sec: float = 2.5,
        autonomous_demo_duration_sec: float = 6.0,
        auto_advance: bool = True,
        speaker: Optional[SpeechSynthesizerABC] = None,
        voice_guidance_enabled: bool = True
    ) -> None:
        self.foresee_steps = foresee_steps
        # Long enough that the client's full real-motion replay (up to
        # execution_max_steps frames, ~2-3s of recorded execution) plays through
        # at least once before the phase auto-advances, rather than being cut
        # off mid-clip.
        self.foresee_duration_sec = foresee_duration_sec
        self.wait_user_timeout = wait_user_timeout
        self.execution_max_steps = execution_max_steps
        # Same reasoning as foresee_duration_sec, but for the post-execution
        # "here's what you just did" review replay.
        self.adapting_duration_sec = adapting_duration_sec
        self.restart_delay_sec = restart_delay_sec
        # Long enough for the ~2s synthetic plan to play through at least once,
        # with a beat to register the final grasp/lift pose before returning to IDLE.
        self.autonomous_demo_duration_sec = autonomous_demo_duration_sec
        self.auto_advance = auto_advance
        # A demo requested while the workflow was mid-episode, waiting for a
        # phase that will accept it. See handle_control_command.
        self._pending_demo_at: Optional[float] = None
        # Counted only while a phase WOULD accept the demo (see
        # poll_pending_demo), so this bounds "how long a startable request may
        # sit unstarted", not how long the user's attempt is allowed to take.
        self.pending_demo_ttl_sec = 20.0
        self.speaker = speaker
        self.voice_guidance_enabled = voice_guidance_enabled

        self._phase = ExecutionPhase.IDLE
        self._step_index = 0
        self._phase_start_time = time.time()
        self._target_label = "none"

        # Buffers for full-sequence rollouts
        self.stored_foreseen_trajectory: Optional[ForeseenTrajectory] = None
        self.recorded_physical_poses: List[HandPose] = []
        self.last_adaptation_report: Optional[dict] = None

    @property
    def current_phase(self) -> ExecutionPhase:
        return self._phase

    @property
    def phase_progress(self) -> float:
        """Normalized progress [0.0, 1.0] of the active phase."""
        if self._phase == ExecutionPhase.IDLE:
            return 0.0
        elif self._phase == ExecutionPhase.FORESEEING:
            return min(1.0, self._step_index / float(self.foresee_steps))
        elif self._phase == ExecutionPhase.WAIT_USER:
            elapsed = time.time() - self._phase_start_time
            return min(1.0, elapsed / max(self.wait_user_timeout, 0.1))
        elif self._phase == ExecutionPhase.USER_EXECUTING:
            return min(1.0, len(self.recorded_physical_poses) / float(self.foresee_steps))
        elif self._phase == ExecutionPhase.ADAPTING:
            elapsed = time.time() - self._phase_start_time
            return min(1.0, elapsed / max(self.adapting_duration_sec, 0.1))
        elif self._phase == ExecutionPhase.RESTARTING:
            elapsed = time.time() - self._phase_start_time
            return min(1.0, elapsed / max(self.restart_delay_sec, 0.1))
        elif self._phase == ExecutionPhase.AUTONOMOUS_DEMO:
            elapsed = time.time() - self._phase_start_time
            return min(1.0, elapsed / max(self.autonomous_demo_duration_sec, 0.1))
        return 0.0

    @property
    def step_index(self) -> int:
        return self._step_index

    def get_state(self) -> WorkflowState:
        return WorkflowState(
            phase=self._phase,
            phase_progress=self.phase_progress,
            step_index=self._step_index,
            total_steps=self.foresee_steps,
            phase_elapsed_sec=time.time() - self._phase_start_time,
            intent_active=(self._target_label != "none"),
            target_label=self._target_label
        )

    def _phase_instruction(self, phase: ExecutionPhase) -> Optional[str]:
        """Compose the spoken instruction announced on entering a given phase."""
        target = self._target_label.replace("_", " ") if self._target_label not in ("none", "") else "an object"
        if phase == ExecutionPhase.IDLE:
            return "Standby. Say what to pick up, for example wine glass."
        elif phase == ExecutionPhase.FORESEEING:
            return f"Get ready to grasp the {target}."
        elif phase == ExecutionPhase.WAIT_USER:
            return "Your turn. Press C when ready."
        elif phase == ExecutionPhase.USER_EXECUTING:
            return f"Go. Reach for the {target} now."
        elif phase == ExecutionPhase.ADAPTING:
            return "Nice. Here's a replay of what you just did."
        elif phase == ExecutionPhase.RESTARTING:
            return f"Restarting with what I learned about how you grasp the {target}."
        elif phase == ExecutionPhase.AUTONOMOUS_DEMO:
            return f"Watch. Simulating how I'd grasp the {target}, using everything learned from your attempts."
        return None

    def transition_to(self, new_phase: ExecutionPhase) -> None:
        """Explicitly switch state machine phase."""
        if new_phase == self._phase:
            return

        old_phase = self._phase
        self._phase = new_phase
        self._step_index = 0
        self._phase_start_time = time.time()

        if new_phase == ExecutionPhase.IDLE:
            self.recorded_physical_poses.clear()
            # Returning to IDLE normally means the user withdrew their intent, so
            # the target is cleared. The Autonomous Demo is the exception: it is a
            # one-off showcase that ENDS by returning to IDLE, and the user's
            # intent is untouched by it. Clearing the target there made the demo a
            # strictly once-per-intent affair - every subsequent press of the
            # hotkey was refused by the guard in handle_control_command, silently,
            # while the object was still sitting in frame.
            if old_phase != ExecutionPhase.AUTONOMOUS_DEMO:
                self._target_label = "none"
        elif new_phase == ExecutionPhase.USER_EXECUTING:
            self.recorded_physical_poses.clear()

        logger.info(f"Workflow State Transition: [{old_phase.value}] -> [{new_phase.value}]")

        if self.speaker is not None and self.voice_guidance_enabled:
            instruction = self._phase_instruction(new_phase)
            if instruction:
                self.speaker.speak(instruction)

    def trigger_intent(self, target_label: str, foreseen_traj: Optional[ForeseenTrajectory] = None) -> None:
        """Trigger workflow start upon new voice/text intent."""
        self._target_label = target_label
        self.stored_foreseen_trajectory = foreseen_traj
        self.recorded_physical_poses.clear()

        if target_label.lower() not in ("none", "idle", "clear", ""):
            self.transition_to(ExecutionPhase.FORESEEING)
        else:
            self.transition_to(ExecutionPhase.IDLE)

    def step_foresee(self) -> bool:
        """Advance foreseen ghost rollout by elapsed wall-clock time (not call count),
        so the preview always plays back at its intended real-time duration regardless
        of how often step_foresee() happens to be invoked (e.g. server frame throughput
        under network/GPU load). Returns True once the rollout duration has elapsed."""
        if self._phase != ExecutionPhase.FORESEEING:
            return False

        elapsed = time.time() - self._phase_start_time
        frac = min(1.0, elapsed / max(self.foresee_duration_sec, 0.01))
        self._step_index = int(frac * self.foresee_steps)
        if frac >= 1.0:
            if self.auto_advance:
                self.transition_to(ExecutionPhase.WAIT_USER)
            return True
        return False

    def step_wait_user(self) -> bool:
        """Check wait countdown timer. Returns True when timer expires."""
        if self._phase != ExecutionPhase.WAIT_USER:
            return False

        elapsed = time.time() - self._phase_start_time
        if elapsed >= self.wait_user_timeout and self.auto_advance:
            self.transition_to(ExecutionPhase.USER_EXECUTING)
            return True
        return False

    def record_execution_step(self, physical_pose: Optional[HandPose]) -> bool:
        """Record real-time physical hand pose. Returns True when physical sequence matches length."""
        if self._phase != ExecutionPhase.USER_EXECUTING:
            return False

        if physical_pose is not None:
            self.recorded_physical_poses.append(physical_pose)
            self._step_index += 1

        if len(self.recorded_physical_poses) >= self.foresee_steps or self._step_index >= self.execution_max_steps:
            if self.auto_advance:
                self.transition_to(ExecutionPhase.ADAPTING)
            return True
        return False

    def step_adapting(self) -> bool:
        """Hold in ADAPTING for a fixed wall-clock duration so the post-execution
        replay review (rendered client-side from the user's own recorded motion)
        gets real screen time, instead of the phase advancing to RESTARTING on
        the very next processed frame - which previously gave the review moment
        almost no time to be seen at all. Returns True once elapsed."""
        if self._phase != ExecutionPhase.ADAPTING:
            return False

        elapsed = time.time() - self._phase_start_time
        if elapsed >= self.adapting_duration_sec and self.auto_advance:
            self.transition_to(ExecutionPhase.RESTARTING)
            return True
        return False

    def step_restarting(self) -> bool:
        """Brief pause after ADAPTING so the user has a clear moment to register that
        a new, adapted attempt is about to begin. Loops back into FORESEEING for the
        SAME target (preserving _target_label) rather than dropping to IDLE, so the
        improved plan is immediately visible without re-speaking the object name.
        Returns True once the restart has fired."""
        if self._phase != ExecutionPhase.RESTARTING:
            return False

        elapsed = time.time() - self._phase_start_time
        if elapsed >= self.restart_delay_sec and self.auto_advance:
            self.trigger_intent(self._target_label)
            return True
        return False

    def poll_pending_demo(self) -> bool:
        """Start a deferred Autonomous Demo once the workflow will accept one.

        Call once per frame. Returns True if a deferred request was started.
        Requests expire so that one swallowed during a long episode cannot
        surprise the user by firing minutes later.
        """
        if self._pending_demo_at is None:
            return False

        if self._target_label in ("none", "idle", "clear", ""):
            self._pending_demo_at = None
            logger.info("Autonomous Demo request dropped: the target was cleared.")
            return False

        if self._phase in (ExecutionPhase.USER_EXECUTING, ExecutionPhase.ADAPTING,
                           ExecutionPhase.AUTONOMOUS_DEMO):
            # Hold the request AND the clock. The expiry exists so a forgotten
            # request cannot fire out of nowhere, not to impose a deadline on
            # the attempt the user is in the middle of - a real attempt runs
            # 45 s or more, so a wall-clock timer measured from the keypress
            # simply discarded the request every time.
            self._pending_demo_at = time.time()
            return False

        if time.time() - self._pending_demo_at > self.pending_demo_ttl_sec:
            self._pending_demo_at = None
            logger.info("Autonomous Demo request expired before a phase would accept it.")
            return False

        self._pending_demo_at = None
        logger.info("Starting the Autonomous Demo that was deferred earlier.")
        self.transition_to(ExecutionPhase.AUTONOMOUS_DEMO)
        return True

    def step_autonomous_demo(self) -> bool:
        """Hold the hands-off Autonomous Demo for a fixed wall-clock duration,
        then return to IDLE - this is a one-off, on-demand showcase (triggered
        by START_AUTONOMOUS_DEMO), not a step in the normal Foresee-Execute-Adapt
        cycle, so it doesn't loop or chain into anything else. Returns True once
        the demo has finished."""
        if self._phase != ExecutionPhase.AUTONOMOUS_DEMO:
            return False

        elapsed = time.time() - self._phase_start_time
        if elapsed >= self.autonomous_demo_duration_sec and self.auto_advance:
            self.transition_to(ExecutionPhase.IDLE)
            return True
        return False

    def advance_phase(self) -> ExecutionPhase:
        """Manual trigger to jump to the subsequent phase."""
        if self._phase == ExecutionPhase.IDLE:
            self.transition_to(ExecutionPhase.FORESEEING)
        elif self._phase == ExecutionPhase.FORESEEING:
            self.transition_to(ExecutionPhase.WAIT_USER)
        elif self._phase == ExecutionPhase.WAIT_USER:
            self.transition_to(ExecutionPhase.USER_EXECUTING)
        elif self._phase == ExecutionPhase.USER_EXECUTING:
            self.transition_to(ExecutionPhase.ADAPTING)
        elif self._phase == ExecutionPhase.ADAPTING:
            self.transition_to(ExecutionPhase.RESTARTING)
        elif self._phase == ExecutionPhase.RESTARTING:
            self.trigger_intent(self._target_label)  # skip the wait, restart immediately
        return self._phase

    def handle_control_command(self, cmd: str) -> None:
        """Handle transport control signals."""
        cmd_upper = cmd.upper()
        if cmd_upper == WorkflowControlSignal.START_FORESEE.value:
            self.transition_to(ExecutionPhase.FORESEEING)
        elif cmd_upper == WorkflowControlSignal.START_USER_EXECUTION.value:
            self.transition_to(ExecutionPhase.USER_EXECUTING)
        elif cmd_upper == WorkflowControlSignal.FINISH_EPISODE.value:
            self.transition_to(ExecutionPhase.ADAPTING)
        elif cmd_upper == WorkflowControlSignal.RESET_IDLE.value:
            self.transition_to(ExecutionPhase.IDLE)
        elif cmd_upper == WorkflowControlSignal.ADVANCE_PHASE.value:
            self.advance_phase()
        elif cmd_upper == WorkflowControlSignal.START_AUTONOMOUS_DEMO.value:
            # Only meaningful with an active target, and not while a real attempt
            # is actually in progress - an on-demand showcase shouldn't barge in
            # mid-execution or mid-adaptation.
            if self._target_label in ("none", "idle", "clear", ""):
                logger.info("Autonomous Demo requested with no active target; ignoring.")
            elif self._phase in (ExecutionPhase.USER_EXECUTING, ExecutionPhase.ADAPTING):
                # DEFER rather than drop. The request travels a frame behind the
                # keypress, and the phases auto-advance on their own timers, so
                # on a slow host the workflow routinely moves into a refusing
                # phase in the gap - the user presses the key and nothing at all
                # happens, with no way to tell that from a broken feature.
                self._pending_demo_at = time.time()
                logger.info(
                    f"Autonomous Demo requested during [{self._phase.value}]; "
                    f"deferred until the current attempt finishes."
                )
            else:
                self._pending_demo_at = None
                self.transition_to(ExecutionPhase.AUTONOMOUS_DEMO)
