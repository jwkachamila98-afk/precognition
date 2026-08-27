"""Thread-safe Component-Level Latency Profiling Suite."""

import collections
import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional
import numpy as np


@dataclass
class StageStats:
    """Latency metrics for an individual pipeline stage."""
    name: str
    last_ms: float
    mean_ms: float
    std_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    sample_count: int


class LatencyProfiler:
    """
    Thread-safe component-level latency profiler for real-time robotic perception & policies.
    Tracks execution times across pipeline stages and computes rolling averages, std dev,
    and 99th-percentile (P99) latency.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._lock = threading.Lock()
        self._stage_history: Dict[str, collections.deque] = {}
        self._active_timers: Dict[str, float] = {}

    def start_stage(self, stage_name: str) -> None:
        """Start recording timing for a named pipeline stage."""
        with self._lock:
            self._active_timers[stage_name] = time.perf_counter()

    def stop_stage(self, stage_name: str) -> float:
        """Stop timing for a stage and record duration in milliseconds."""
        now = time.perf_counter()
        with self._lock:
            t_start = self._active_timers.pop(stage_name, None)
            if t_start is None:
                return 0.0
            elapsed_ms = (now - t_start) * 1000.0

            if stage_name not in self._stage_history:
                self._stage_history[stage_name] = collections.deque(maxlen=self.window_size)
            self._stage_history[stage_name].append(elapsed_ms)
            return elapsed_ms

    @contextlib.contextmanager
    def profile(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager for concise profiling: with profiler.profile('stage_name'):"""
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.stop_stage(stage_name)

    def get_stage_stats(self, stage_name: str) -> Optional[StageStats]:
        """Compute rolling latency statistics for a specific stage."""
        with self._lock:
            history = self._stage_history.get(stage_name)
            if not history:
                return None
            arr = np.array(history, dtype=np.float64)
            return StageStats(
                name=stage_name,
                last_ms=float(arr[-1]),
                mean_ms=float(np.mean(arr)),
                std_ms=float(np.std(arr)),
                p99_ms=float(np.percentile(arr, 99)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
                sample_count=len(arr)
            )

    def get_all_stats(self) -> Dict[str, StageStats]:
        """Compute stats for all tracked pipeline stages."""
        with self._lock:
            stage_names = list(self._stage_history.keys())

        stats = {}
        for name in stage_names:
            s = self.get_stage_stats(name)
            if s is not None:
                stats[name] = s
        return stats

    def get_total_latency_ms(self) -> float:
        """Sum of mean latencies across all stages."""
        stats = self.get_all_stats()
        return sum(s.mean_ms for s in stats.values())

    def format_table(self, fps: float = 0.0) -> str:
        """Format a real-time ASCII table breakdown of component latencies."""
        stats = self.get_all_stats()
        if not stats:
            return "No profiling samples collected yet."

        header = f"+--------------------------------------------+----------+----------+----------+----------+\n"
        header += f"| Pipeline Stage                             | Mean(ms) | Std(ms)  | P99(ms)  | Last(ms) |\n"
        header += f"+--------------------------------------------+----------+----------+----------+----------+"
        
        rows = [header]
        total_mean = 0.0
        total_p99 = 0.0

        for name, s in stats.items():
            display_name = (name[:40] + "..") if len(name) > 42 else name
            row = f"| {display_name:<42} | {s.mean_ms:8.2f} | {s.std_ms:8.2f} | {s.p99_ms:8.2f} | {s.last_ms:8.2f} |"
            rows.append(row)
            total_mean += s.mean_ms
            total_p99 += s.p99_ms

        divider = f"+--------------------------------------------+----------+----------+----------+----------+"
        rows.append(divider)
        fps_info = f" (Rolling FPS: {fps:.1f})" if fps > 0 else ""
        total_row = f"| TOTAL PIPELINE LATENCY{fps_info:<20} | {total_mean:8.2f} |          | {total_p99:8.2f} |          |"
        rows.append(total_row)
        rows.append(divider)

        return "\n".join(rows)

    def reset(self) -> None:
        """Clear all timing history."""
        with self._lock:
            self._stage_history.clear()
            self._active_timers.clear()
