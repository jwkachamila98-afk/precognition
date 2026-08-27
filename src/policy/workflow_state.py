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


class WorkflowControlSignal(str, Enum):
    """Control commands sent across transport to trigger phase transitions."""
    START_FORESEE = "START_FORESEE"
    START_USER_EXECUTION = "START_USER_EXECUTION"
    FINISH_EPISODE = "FINISH_EPISODE"
    RESET_IDLE = "RESET_IDLE"
    ADVANCE_PHASE = "ADVANCE_PHASE"


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
        wait_user_timeout: float = 3.0,
        execution_max_steps: int = 90,
        auto_advance: bool = True,
        speaker: Optional[SpeechSynthesizerABC] = None,
        voice_guidance_enabled: bool = True
    ) -> None:
        self.foresee_steps = foresee_steps
        self.wait_user_timeout = wait_user_timeout
        self.execution_max_steps = execution_max_steps
        self.auto_advance = auto_advance
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
            return 1.0
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
        target = self._target_label if self._target_label not in ("none", "") else "the object"
        if phase == ExecutionPhase.IDLE:
            return "Standby. Tell me what to pick up."
        elif phase == ExecutionPhase.FORESEEING:
            return f"Foreseeing how to grasp the {target}."
        elif phase == ExecutionPhase.WAIT_USER:
            return "Ready when you are. Go ahead and move your hand."
        elif phase == ExecutionPhase.USER_EXECUTING:
            return "Tracking your motion now."
        elif phase == ExecutionPhase.ADAPTING:
            return "Got it. Adjusting based on what I saw."
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
        """Step foreseen ghost rollout. Returns True when 60 steps complete."""
        if self._phase != ExecutionPhase.FORESEEING:
            return False

        self._step_index += 1
        if self._step_index >= self.foresee_steps:
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
            self.transition_to(ExecutionPhase.IDLE)
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
