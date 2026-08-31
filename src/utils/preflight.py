"""Startup verification (src/utils/preflight.py).

Every heavy component in this system has a mock fallback, and each one logs a
warning and carries on. That is the right behaviour while developing on a
laptop with nothing installed. It is the wrong behaviour in front of an
audience: the app starts, the HUD fills with plausible numbers, and nobody can
tell that the hand tracker is synthesising a hand and the policy is a stub.

So before anything starts, the capabilities the run actually depends on are
checked and reported. If a required one is missing the process refuses to
start and says exactly what is wrong and how to fix it, rather than quietly
demonstrating fabricated data.

`--allow-degraded` restores the old permissive behaviour for development.
"""

from __future__ import annotations

import importlib
import os
import shutil
import socket
from dataclasses import dataclass
from typing import Callable, List, Optional

# ANSI, but only when stdout is a terminal that will render it.
def _supports_colour() -> bool:
    return hasattr(os.sys.stdout, "isatty") and os.sys.stdout.isatty()


_C = {
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "dim": "\033[2m", "bold": "\033[1m", "off": "\033[0m",
}


def _c(text: str, colour: str) -> str:
    return f"{_C[colour]}{text}{_C['off']}" if _supports_colour() else text


@dataclass
class Check:
    """One verified capability."""
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    required: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return _c("OK", "green")
        return _c("FAIL", "red") if self.required else _c("WARN", "yellow")


def _module(name: str, label: str, fix: str, required: bool = True,
            version_attr: str = "__version__") -> Check:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, version_attr, "")
        return Check(label, True, str(version), required=required)
    except Exception as exc:
        return Check(label, False, f"{type(exc).__name__}: {exc}", fix, required)


def check_torch(require_cuda: bool = False) -> List[Check]:
    checks = [_module("torch", "PyTorch", "pip install torch")]
    if not checks[0].ok:
        return checks
    import torch
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "cpu only"
    checks.append(Check(
        "CUDA device", cuda or not require_cuda, name,
        "This host has no usable GPU. Run the server on a GPU pod, or pass "
        "--allow-degraded to run the models on CPU.",
        required=require_cuda))
    return checks


def check_camera(device_id: int = 0) -> Check:
    """Open the camera and confirm frames actually arrive.

    Opening succeeds on macOS even for modes the device cannot deliver, so the
    only honest test is to read a frame.
    """
    try:
        import cv2
    except Exception as exc:
        return Check("Camera", False, f"OpenCV missing: {exc}", "pip install opencv-python")
    cap = None
    try:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            return Check("Camera", False, f"device {device_id} would not open",
                         "Close other apps using the camera, or check "
                         "System Settings > Privacy & Security > Camera.")
        for _ in range(6):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                h, w = frame.shape[:2]
                return Check("Camera", True, f"device {device_id} delivering {w}x{h}")
        return Check("Camera", False, f"device {device_id} opened but delivered no frames",
                     "Another app may hold the camera, or the requested mode is "
                     "unsupported. Try a different --device index.")
    finally:
        if cap is not None:
            cap.release()


def check_microphone() -> Check:
    try:
        import sounddevice as sd
        inputs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        if not inputs:
            return Check("Microphone", False, "no input devices",
                         "Connect a microphone, or check System Settings > "
                         "Privacy & Security > Microphone.", required=False)
        return Check("Microphone", True, inputs[0]["name"], required=False)
    except Exception as exc:
        return Check("Microphone", False, f"{type(exc).__name__}: {exc}",
                     "pip install sounddevice", required=False)


def check_gemini_key(required: bool = True) -> Check:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return Check("Gemini API key", False, "GEMINI_API_KEY not set",
                     "export GEMINI_API_KEY=... - without it speech becomes "
                     "canned presets and object grounding falls back to COCO-80.",
                     required=required)
    return Check("Gemini API key", True, f"set ({len(key)} chars)", required=required)


def check_server_reachable(url: str, timeout: float = 8.0) -> Check:
    """Confirm the inference server is actually accepting connections."""
    host, _, port = url.replace("ws://", "").replace("wss://", "").partition(":")
    port_num = int(port.split("/")[0]) if port else 8765
    try:
        with socket.create_connection((host, port_num), timeout=timeout):
            return Check("Inference server", True, f"{host}:{port_num} accepting connections")
    except Exception as exc:
        return Check("Inference server", False, f"{host}:{port_num} - {type(exc).__name__}",
                     "The GPU pod may be stopped, still installing, or its port "
                     "may have remapped on restart. Run tools/launch_demo.py, "
                     "which discovers the current port for you.")


def check_port_free(port: int) -> Check:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
        return Check(f"Port {port}", True, "free to bind")
    except OSError as exc:
        return Check(f"Port {port}", False, str(exc),
                     f"Another process holds {port}. Stop it, or pass --port.")


def check_disk(min_free_gb: float = 2.0) -> Check:
    free_gb = shutil.disk_usage(".").free / 1e9
    return Check("Disk space", free_gb >= min_free_gb, f"{free_gb:.1f} GB free",
                 f"Model weights need room; keep at least {min_free_gb:.0f} GB free.",
                 required=False)


def client_checks(server_url: Optional[str], device_id: int = 0,
                  needs_voice: bool = True) -> List[Check]:
    checks = [
        _module("cv2", "OpenCV", "pip install opencv-python"),
        _module("numpy", "NumPy", "pip install numpy"),
        _module("mediapipe", "MediaPipe", "pip install mediapipe"),
        _module("websockets", "websockets", "pip install websockets"),
        check_camera(device_id),
    ]
    if needs_voice:
        checks.append(check_microphone())
        checks.append(check_gemini_key(required=False))
    if server_url:
        checks.append(check_server_reachable(server_url))
    checks.append(check_disk())
    return checks


def server_checks(port: int = 8765, require_cuda: bool = True) -> List[Check]:
    checks = [
        _module("cv2", "OpenCV", "pip install opencv-python-headless"),
        _module("numpy", "NumPy", "pip install numpy"),
        _module("mediapipe", "MediaPipe", "pip install mediapipe"),
        _module("websockets", "websockets", "pip install websockets"),
    ]
    checks += check_torch(require_cuda=require_cuda)
    checks += [
        _module("ultralytics", "Ultralytics (YOLO)", "pip install ultralytics"),
        _module("timm", "timm (MiDaS backbone)", "pip install timm"),
        check_gemini_key(required=False),
        check_port_free(port),
        check_disk(),
    ]
    return checks


def render(checks: List[Check], title: str) -> str:
    width = max((len(c.name) for c in checks), default=10) + 2
    lines = [""]
    lines.append(_c(f"  {title}", "bold"))
    lines.append(_c("  " + "-" * (width + 46), "dim"))
    for c in checks:
        lines.append(f"  {c.name:<{width}} {c.status:<16} {_c(c.detail, 'dim')}")
    return "\n".join(lines)


def enforce(checks: List[Check], title: str, allow_degraded: bool = False,
            printer: Callable[[str], None] = print) -> bool:
    """Print the report; return True if it is safe to continue.

    A failed REQUIRED check stops the run unless explicitly overridden, because
    continuing means demonstrating mock data dressed as real output.
    """
    printer(render(checks, title))
    failed = [c for c in checks if not c.ok and c.required]
    warned = [c for c in checks if not c.ok and not c.required]

    for c in warned:
        printer("")
        printer(_c(f"  ! {c.name}: {c.detail}", "yellow"))
        if c.fix:
            printer(_c(f"    {c.fix}", "dim"))

    if not failed:
        printer("")
        printer(_c("  All required checks passed.", "green"))
        printer("")
        return True

    printer("")
    printer(_c(f"  {len(failed)} required check(s) failed:", "red"))
    for c in failed:
        printer("")
        printer(_c(f"  x {c.name}: {c.detail}", "red"))
        if c.fix:
            printer(f"    {c.fix}")
    printer("")
    if allow_degraded:
        printer(_c("  --allow-degraded set: starting anyway. Output may be "
                   "SYNTHETIC and must not be presented as real.", "yellow"))
        printer("")
        return True
    printer(_c("  Refusing to start. Fix the above, or pass --allow-degraded to "
               "run on mocks.", "dim"))
    printer("")
    return False
