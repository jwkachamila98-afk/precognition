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

A pod this launcher discovered is STOPPED when the session ends - on a
normal exit, a crash, or Ctrl-C. It used to be left running, and an idle
RUNNING pod bills exactly like a working one; that is what took the account
to a 402 and left the demo without a GPU. Pass --keep-pod for back-to-back
sessions. A pod reached through an explicit --server-url is never stopped:
this launcher did not start it and does not know whose it is.
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


def _api(path, api_key, method="GET"):
    req = urllib.request.Request(f"{REST}{path}", method=method,
                                 headers={"Authorization": f"Bearer {api_key}"})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            body = resp.read().decode()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        print(_c(f"  RunPod API {e.code}: {e.read().decode()[:200]}", "red"))
        return None
    except Exception as e:
        print(_c(f"  RunPod API unreachable: {e}", "red"))
        return None


def stop_pod(pod_id, api_key):
    """Stop a pod. Returns True if RunPod accepted the request.

    An idle RUNNING pod bills at the same rate as a working one, so a session
    that ends without this is a session that keeps charging - which is what
    took the account to a 402 and left the demo with no GPU.
    """
    print(_c(f"  stopping pod {pod_id}...", "dim"))
    if _api(f"/pods/{pod_id}/stop", api_key, method="POST") is None:
        print(_c(f"  COULD NOT STOP POD {pod_id} - it may still be billing.",
                 "red"))
        print(_c("  Stop it at https://console.runpod.io/pods", "dim"))
        return False
    print(_c("  pod stopped (its disk persists; restart it for the next run)",
             "green"))
    return True


def discover_pod_url(api_key, pod_name=None):
    """Find a running pod's public endpoint for the inference port.

    The port is read from the live runtime rather than remembered, because it
    is reassigned every time the container restarts. Returns (url, pod_id) so
    the caller can stop the pod it found when the session ends.
    """
    data = _api("/pods", api_key)
    if not data:
        return None, None
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
                return f"ws://{ip}:{public}", pod.get("id")
    return None, None


def server_is_up(url, timeout=6):
    """Whether the INFERENCE SERVER is listening - not merely the port.

    A bare TCP connect is not evidence. RunPod publishes a pod's ports through
    a proxy that accepts connections whether or not anything is listening
    behind them, so connect() succeeds the second the pod boots and the
    launcher would announce "server ready after 0s" while apt and pip still
    had five minutes to run. The client then started against nothing.

    Completing a WebSocket handshake needs the server process itself, so that
    is what gets tested.
    """
    try:
        import asyncio

        import websockets
    except ImportError:                       # no websockets: fall back
        import socket
        host, _, port = url.replace("ws://", "").partition(":")
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError:
            return False

    async def _handshake():
        try:
            async with websockets.connect(url, open_timeout=timeout,
                                          close_timeout=2):
                return True
        except Exception:
            return False

    try:
        return asyncio.run(_handshake())
    except Exception:
        return False


def wait_for_server(url, timeout_s=1200):
    """Block until the inference server is genuinely serving."""
    host, _, port = url.replace("ws://", "").partition(":")
    start = time.time()
    spinner, i = "|/-\\", 0
    while time.time() - start < timeout_s:
        if server_is_up(url):
            elapsed = time.time() - start
            sys.stdout.write("\r" + " " * 78 + "\r")
            print(_c(f"  server ready after {elapsed:.0f}s", "green"))
            return True
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
    ap.add_argument("--keep-pod", action="store_true",
                    help="Leave the GPU pod RUNNING after the client exits. It "
                         "keeps billing - only for back-to-back sessions.")
    args = ap.parse_args()

    print()
    print(_c("  PRECOGNITION", "bold") + _c("  demo launcher", "dim"))
    print()

    client = [sys.executable, str(PROJECT_ROOT / "apps" / "local_client.py"),
              "--tracker", "mediapipe"]
    if args.device is not None:
        client += ["--device", str(args.device)]

    # Set only when THIS launcher discovered the pod, so a session started
    # against someone else's --server-url never stops a pod it doesn't own.
    pod_to_stop = None

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
            url, pod_id = discover_pod_url(args.api_key, args.pod_name)
            if not url:
                print(_c("  No running pod exposing port 8765 was found.", "red"))
                print(_c("  Start one, or pass --server-url explicitly.", "dim"))
                print()
                return 2
            if pod_id and not args.keep_pod:
                pod_to_stop = pod_id
        print(f"  server: {_c(url, 'bold')}")
        if pod_to_stop:
            print(_c("  the pod will be STOPPED when this session ends "
                     "(--keep-pod to leave it running)", "dim"))
        elif not args.local and args.keep_pod:
            print(_c("  --keep-pod: the pod will keep running, and keep "
                     "billing, after this session", "yellow"))
        if not args.no_wait and not wait_for_server(url):
            # The pod is up and billing but unusable - stopping it is the whole
            # point of this change, so do it on the failure path too.
            if pod_to_stop:
                stop_pod(pod_to_stop, args.api_key)
            return 1
        client += ["--mode", "remote", "--server-url", url]

    if not os.environ.get("GEMINI_API_KEY"):
        print(_c("  note: GEMINI_API_KEY not set - speech and open-vocabulary "
                 "grounding will be limited.", "yellow"))

    print(_c("  starting client...", "dim"))
    print()
    try:
        return subprocess.call(client, cwd=str(PROJECT_ROOT),
                               env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
    except KeyboardInterrupt:
        print()
        print(_c("  interrupted", "dim"))
        return 130
    finally:
        # Every exit path, including Ctrl-C and a client crash: an idle pod
        # left RUNNING is what emptied the account before.
        if pod_to_stop:
            print()
            stop_pod(pod_to_stop, args.api_key)
            print()


if __name__ == "__main__":
    sys.exit(main())
