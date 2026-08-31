"""One-command demo launcher (tools/launch_demo.py).

Starting a session by hand meant: check the pod is up, look up its public port
(RunPod remaps it on every boot), poll until the server finishes installing,
then start the client with the right URL. Four steps, one of which - the port -
changes every time and has produced several "connection refused" false alarms.

This does all of it:

    python tools/launch_demo.py                    # discover pod, wait, launch
    python tools/launch_demo.py --server-url ws://host:port
    python tools/launch_demo.py --local            # no GPU pod at all

Pod discovery needs RUNPOD_API_KEY. Without it, pass --server-url.
"""

import argparse
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REST = "https://rest.runpod.io/v1"


def _c(text, colour):
    codes = {"green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
             "dim": "\033[2m", "bold": "\033[1m"}
    return f"{codes[colour]}{text}\033[0m" if sys.stdout.isatty() else text


def _api(path, api_key):
    req = urllib.request.Request(f"{REST}{path}",
                                 headers={"Authorization": f"Bearer {api_key}"})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(_c(f"  RunPod API {e.code}: {e.read().decode()[:200]}", "red"))
        return None
    except Exception as e:
        print(_c(f"  RunPod API unreachable: {e}", "red"))
        return None


def discover_pod_url(api_key, pod_name=None):
    """Find a running pod's public endpoint for the inference port.

    The port is read from the live runtime rather than remembered, because it
    is reassigned every time the container restarts.
    """
    data = _api("/pods", api_key)
    if not data:
        return None
    pods = data if isinstance(data, list) else data.get("pods", data.get("data", []))
    for pod in pods:
        if pod.get("desiredStatus") != "RUNNING" and pod.get("status") != "RUNNING":
            continue
        if pod_name and pod_name not in (pod.get("name") or ""):
            continue
        runtime = pod.get("runtime") or {}
        for port in (runtime.get("ports") or []):
            private = port.get("privatePort", port.get("private"))
            public = port.get("publicPort", port.get("public"))
            ip = port.get("ip")
            if private == 8765 and ip and public:
                print(f"  pod {_c(pod.get('name', pod.get('id')), 'bold')} "
                      f"({pod.get('id')}) in {pod.get('dataCenterId', '?')}")
                return f"ws://{ip}:{public}"
    return None


def wait_for_server(url, timeout_s=1200):
    """Block until the inference server accepts a connection."""
    import socket
    host, _, port = url.replace("ws://", "").partition(":")
    port = int(port)
    start = time.time()
    spinner, i = "|/-\\", 0
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=5):
                elapsed = time.time() - start
                sys.stdout.write("\r" + " " * 78 + "\r")
                print(_c(f"  server ready after {elapsed:.0f}s", "green"))
                return True
        except OSError:
            pass
        elapsed = int(time.time() - start)
        sys.stdout.write(
            f"\r  {spinner[i % 4]} waiting for {host}:{port} "
            f"(installing, ~5 min on a cold pod) {elapsed}s")
        sys.stdout.flush()
        i += 1
        time.sleep(4)
    sys.stdout.write("\r" + " " * 78 + "\r")
    print(_c(f"  server did not come up within {timeout_s}s", "red"))
    return False


def main():
    ap = argparse.ArgumentParser(description="Launch a Precognition demo session")
    ap.add_argument("--server-url", type=str, default=None,
                    help="ws://host:port of the inference server (skips pod discovery)")
    ap.add_argument("--pod-name", type=str, default="precognition",
                    help="Substring matching the pod name to look for")
    ap.add_argument("--local", action="store_true",
                    help="Run entirely on this machine, no GPU pod")
    ap.add_argument("--api-key", type=str, default=os.environ.get("RUNPOD_API_KEY"))
    ap.add_argument("--no-wait", action="store_true", help="Do not wait for the server")
    ap.add_argument("--device", type=int, default=None, help="Camera device index")
    args = ap.parse_args()

    print()
    print(_c("  PRECOGNITION", "bold") + _c("  demo launcher", "dim"))
    print()

    client = [sys.executable, str(PROJECT_ROOT / "apps" / "local_client.py"),
              "--tracker", "mediapipe"]
    if args.device is not None:
        client += ["--device", str(args.device)]

    if args.local:
        print("  mode: local (no GPU pod)")
        client += ["--mode", "mock_local"]
    else:
        url = args.server_url
        if not url:
            if not args.api_key:
                print(_c("  No --server-url and no RUNPOD_API_KEY set.", "red"))
                print(_c("  Pass --server-url ws://host:port, or --local to run "
                         "without a pod.", "dim"))
                print()
                return 2
            print("  discovering pod...")
            url = discover_pod_url(args.api_key, args.pod_name)
            if not url:
                print(_c("  No running pod exposing port 8765 was found.", "red"))
                print(_c("  Start one, or pass --server-url explicitly.", "dim"))
                print()
                return 2
        print(f"  server: {_c(url, 'bold')}")
        if not args.no_wait and not wait_for_server(url):
            return 1
        client += ["--mode", "remote", "--server-url", url]

    if not os.environ.get("GEMINI_API_KEY"):
        print(_c("  note: GEMINI_API_KEY not set - speech and open-vocabulary "
                 "grounding will be limited.", "yellow"))

    print(_c("  starting client...", "dim"))
    print()
    return subprocess.call(client, cwd=str(PROJECT_ROOT),
                           env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})


if __name__ == "__main__":
    sys.exit(main())
