"""Unified Demo Launcher & Automated Startup Suite (apps/run_demo.py).

Performs system pre-flight environment checks (Python runtime, PyTorch, MediaPipe, Audio, CUDA),
manages backend WebSocket server lifecycle, launches the local visualizer client, and handles clean
signal trapping (SIGINT / Ctrl+C) for safe shutdown.

Usage:
    # 1. Local execution with automatic background server:
    python apps/run_demo.py --mode mock_remote

    # 2. Standalone Mac CPU mode:
    python apps/run_demo.py --mode mock_local

    # 3. Connect Mac visualizer directly to a remote Cloud GPU instance:
    python apps/run_demo.py --server-url ws://<CLOUD_GPU_IP>:8765
"""

import argparse
import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("UnifiedLauncher")


def run_preflight_checks() -> dict:
    """Execute pre-flight environment checks."""
    print("=" * 70)
    print("      VISUOMOTOR HAND POLICY ARCHITECTURE - PRE-FLIGHT CHECKS")
    print("=" * 70)

    # 1. Python version check
    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 9)
    print(f"[*] Python Runtime:     {py_ver:<15} [{'OK' if py_ok else 'FAIL'}]")

    # 2. NumPy & OpenCV
    try:
        import numpy as np
        import cv2
        cv_ver = cv2.__version__
        cv_ok = True
    except ImportError:
        cv_ver = "Not found"
        cv_ok = False
    print(f"[*] OpenCV & NumPy:     v{cv_ver:<14} [{'OK' if cv_ok else 'FAIL'}]")

    # 3. MediaPipe
    try:
        import mediapipe as mp
        mp_ver = getattr(mp, "__version__", "installed")
        mp_ok = True
    except ImportError:
        mp_ver = "Not found (using Mock)"
        mp_ok = False
    print(f"[*] MediaPipe Vision:   {mp_ver:<15} [{'OK' if mp_ok else 'MOCK FALLBACK'}]")

    # 4. PyTorch / CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_str = f"v{torch.__version__} (CUDA: {torch.cuda.get_device_name(0) if cuda_avail else 'CPU'})"
        torch_ok = True
    except ImportError:
        device_str = "PyTorch not found"
        torch_ok = False
    print(f"[*] PyTorch Backend:    {device_str:<15} [{'OK' if torch_ok else 'WARN'}]")

    # 5. Audio / Whisper
    try:
        import faster_whisper
        audio_str = "faster-whisper available"
        audio_ok = True
    except ImportError:
        audio_str = "MockTranscriber (CPU fallback)"
        audio_ok = False
    print(f"[*] Audio STT Engine:   {audio_str:<15} [{'OK' if audio_ok else 'MOCK'}]")

    print("=" * 70 + "\n")
    return {
        "python_ok": py_ok,
        "opencv_ok": cv_ok,
        "mediapipe_ok": mp_ok,
        "torch_ok": torch_ok
    }


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if TCP port is currently listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_server(port: int, host: str = "127.0.0.1", timeout: float = 8.0) -> bool:
    """Poll port until backend server becomes healthy."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_port_in_use(port, host):
            return True
        time.sleep(0.15)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Demo Launcher for Visuomotor Hand Policy")
    parser.add_argument("--mode", type=str, choices=["mock_remote", "mock_local"], default="mock_remote", help="Execution mode (default: mock_remote)")
    parser.add_argument("--server-url", type=str, default=None, help="Remote WebSocket server URL (e.g. ws://<CLOUD_GPU_IP>:8765)")
    parser.add_argument("--voice", type=str, choices=["mock", "whisper"], default="whisper", help="Audio transcriber engine")
    parser.add_argument("--tracker", type=str, choices=["mediapipe", "mock"], default="mediapipe", help="Hand tracker backend")
    parser.add_argument("--profile", action="store_true", help="Enable terminal latency profiling breakdown")
    parser.add_argument("--record", action="store_true", help="Enable automatic dataset session recording")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port for local server (default: 8765)")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--gemini-key", type=str, default=None, help="Gemini API key for voice transcription + TTS (defaults to $GEMINI_API_KEY)")
    args = parser.parse_args()

    # 1. Environment Checks
    checks = run_preflight_checks()
    if not checks["python_ok"] or not checks["opencv_ok"]:
        logger.error("Pre-flight checks failed. Please install dependencies with 'pip install -r requirements.txt'.")
        sys.exit(1)

    server_process: Optional[subprocess.Popen] = None

    def cleanup(signum=None, frame=None):
        """Clean shutdown handler."""
        logger.info("\nCaught shutdown signal. Terminating demo components...")
        if server_process and server_process.poll() is None:
            logger.info("Terminating remote server subprocess...")
            server_process.terminate()
            try:
                server_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
        logger.info("Unified Demo shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    python_executable = sys.executable

    # 2. Launch Local Remote Server if in mock_remote mode AND no remote server-url is provided
    if args.mode == "mock_remote" and not args.server_url:
        if is_port_in_use(args.port):
            logger.info(f"Existing service detected on port {args.port}. Connecting to active server...")
        else:
            logger.info(f"Spawning background Remote Server process on ws://127.0.0.1:{args.port}...")
            server_cmd = [
                python_executable,
                str(PROJECT_ROOT / "apps" / "remote_server.py"),
                "--port", str(args.port),
                "--tracker", args.tracker
            ]
            server_process = subprocess.Popen(server_cmd)
            
            logger.info("Waiting for WebSocket server to initialize...")
            if wait_for_server(args.port):
                logger.info(f"Remote Server is active on port {args.port}.")
            else:
                logger.warning(f"Server port {args.port} did not respond within timeout. Attempting client connection anyway...")

    # 3. Launch Local Client
    client_cmd = [
        python_executable,
        str(PROJECT_ROOT / "apps" / "local_client.py"),
        "--transcriber", args.voice,
        "--tracker", args.tracker,
        "--device", str(args.device)
    ]

    if args.gemini_key:
        client_cmd.extend(["--gemini-key", args.gemini_key])

    if args.server_url:
        client_cmd.extend(["--server-url", args.server_url])
        logger.info(f"Configuring Local Client to connect to remote cloud endpoint: {args.server_url}")
    else:
        client_cmd.extend(["--mode", args.mode])

    if args.profile:
        client_cmd.append("--profile")
    if args.record:
        client_cmd.append("--record")

    target_display_mode = args.server_url if args.server_url else args.mode
    logger.info(f"Launching Local Client Visualizer ({target_display_mode})...")
    try:
        subprocess.run(client_cmd)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
