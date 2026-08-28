"""Automated RunPod GPU Deployment and Lifecycle Management Utility (deploy/runpod_deploy.py).

Allows 1-command provisioning, status monitoring, and termination of Cloud GPU instances
for the Visuomotor Hand Policy Inference Server.

Usage:
    # Check account status & active pods:
    python deploy/runpod_deploy.py status

    # Launch a GPU instance:
    python deploy/runpod_deploy.py launch --gpu "NVIDIA GeForce RTX 3080"

    # Terminate a running instance:
    python deploy/runpod_deploy.py terminate --pod <pod_id>
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

DEFAULT_API_KEY = os.environ.get("RUNPOD_API_KEY")
GRAPHQL_URL = "https://api.runpod.io/graphql"


def execute_graphql(query: str, api_key: str = DEFAULT_API_KEY) -> Dict[str, Any]:
    """Send authenticated GraphQL query to RunPod API."""
    ctx = ssl._create_unverified_context()
    url = f"{GRAPHQL_URL}?api_key={api_key}"
    payload = json.dumps({"query": query}).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "RunPod-Agent-SDK"
        }
    )

    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "errors" in data:
                print(f"[ERROR] GraphQL Errors: {data['errors']}")
            return data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"[HTTP {e.code}] {err_body}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        sys.exit(1)


def get_status(api_key: str = DEFAULT_API_KEY) -> None:
    """Print RunPod user account details and active pods."""
    query = """
    query {
        myself {
            id
            pods {
                id
                name
                desiredStatus
                costPerHr
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
    }
    """
    res = execute_graphql(query, api_key)
    user_data = res.get("data", {}).get("myself", {})
    user_id = user_data.get("id", "Unknown")
    pods = user_data.get("pods", [])

    print("=" * 65)
    print("               RUNPOD AGENT STATUS & POD INVENTORY")
    print("=" * 65)
    print(f"[*] Account User ID:  {user_id}")
    print(f"[*] Total Active Pods: {len(pods)}")
    print("-" * 65)

    if not pods:
        print("[i] No active pods running currently.")
    else:
        for p in pods:
            pod_id = p.get("id")
            name = p.get("name")
            status = p.get("desiredStatus")
            cost = p.get("costPerHr", 0.0)
            runtime = p.get("runtime") or {}
            uptime = runtime.get("uptimeInSeconds", 0)

            print(f"[*] Pod: {name} (ID: {pod_id})")
            print(f"    - Status:   {status}")
            print(f"    - Rate:     ${cost:.3f}/hr")
            print(f"    - Uptime:   {uptime} seconds")

            # Connect over the DIRECT TCP mapping, not the HTTPS proxy.
            # RunPod's *-8765.proxy.runpod.net hostname does not forward to the
            # container's 8765: the HTTP proxy is bound to a different internal
            # port, so a WebSocket upgrade against it is rejected with HTTP 404
            # even while the server is healthy and listening on 0.0.0.0:8765.
            # The runtime port list carries the real public IP and port.
            ws_url = None
            for port in (runtime.get("ports") or []):
                if port.get("privatePort") == 8765 and port.get("ip"):
                    ws_url = f"ws://{port['ip']}:{port['publicPort']}"
                    break

            if ws_url:
                print(f"    - WebSocket Endpoint: {ws_url}")
                print(f"    - Connect Command:    python apps/run_demo.py --server-url {ws_url}")
            else:
                print("    - WebSocket Endpoint: (not mapped yet - pod still starting)")
            print("-" * 65)

    print("=" * 65 + "\n")


def launch_pod(
    gpu_type: str = "NVIDIA GeForce RTX 3080",
    image_name: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    pod_name: str = "visuomotor-server-gpu",
    api_key: str = DEFAULT_API_KEY,
    cloud_type: str = "SECURE"
) -> None:
    """Provision a new GPU instance on RunPod with port 8765 exposed."""
    print(f"[*] Requesting GPU pod launch on RunPod ({gpu_type}, {cloud_type} cloud)...")

    mutation = f"""
    mutation {{
        podFindAndDeployOnDemand(input: {{
            cloudType: {cloud_type},
            gpuCount: 1,
            volumeInGb: 20,
            containerDiskInGb: 20,
            minVcpuCount: 4,
            minMemoryInGb: 16,
            gpuTypeId: "{gpu_type}",
            name: "{pod_name}",
            imageName: "{image_name}",
            ports: "8765/http,8765/tcp,22/tcp",
            volumeMountPath: "/workspace",
            dockerArgs: "bash -c 'mkdir -p /workspace/Precognition && (git clone https://github.com/jwkachamila98-afk/precognition.git /workspace/Precognition || (cd /workspace/Precognition && git pull)) ; cd /workspace/Precognition ; pip install websockets opencv-python-headless mediapipe scipy pyyaml certifi ; python apps/remote_server.py --host 0.0.0.0 --port 8765 ; sleep infinity'"
        }}) {{
            id
            imageName
            desiredStatus
            machineId
        }}
    }}
    """
    res = execute_graphql(mutation, api_key)
    deploy_data = res.get("data", {}).get("podFindAndDeployOnDemand")
    if deploy_data:
        pod_id = deploy_data.get("id")
        print(f"[✓] GPU Pod Successfully Created! (ID: {pod_id})")
        print("\n[✓] Once the pod initializes (~2-3 min), run this to get its endpoint:")
        print("    python deploy/runpod_deploy.py status")
        print("[i] Connect over the DIRECT TCP mapping it prints (ws://<ip>:<port>).")
        print("    The wss://<pod>-8765.proxy.runpod.net hostname does NOT work: the")
        print("    HTTP proxy is bound to a different internal port and rejects the")
        print("    WebSocket upgrade with HTTP 404 even when the server is healthy.\n")
    else:
        print("[!] Pod creation response did not return an instance ID. Check account credits or GPU availability.")


def terminate_pod(pod_id: str, api_key: str = DEFAULT_API_KEY) -> None:
    """Terminate and stop billing for a RunPod instance."""
    print(f"[*] Terminating pod {pod_id} on RunPod...")
    mutation = f"""
    mutation {{
        podTerminate(input: {{
            podId: "{pod_id}"
        }})
    }}
    """
    res = execute_graphql(mutation, api_key)
    print(f"[✓] Pod {pod_id} terminated successfully. Billing stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod GPU Deployment & Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # status
    parser_status = subparsers.add_parser("status", help="Check RunPod account & active pods")
    parser_status.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="RunPod API Key")

    # launch
    parser_launch = subparsers.add_parser("launch", help="Launch a new GPU pod on RunPod")
    parser_launch.add_argument("--gpu", type=str, default="NVIDIA GeForce RTX 3080", help="GPU Type ID (e.g. 'NVIDIA GeForce RTX 3080', 'NVIDIA RTX A4000')")
    parser_launch.add_argument("--image", type=str, default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04", help="Docker image")
    parser_launch.add_argument("--name", type=str, default="visuomotor-server-gpu", help="Pod name")
    parser_launch.add_argument("--cloud-type", type=str, choices=["SECURE", "COMMUNITY"], default="SECURE", help="RunPod cloud tier (COMMUNITY has broader GPU availability)")
    parser_launch.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="RunPod API Key")

    # terminate
    parser_term = subparsers.add_parser("terminate", help="Terminate a running pod")
    parser_term.add_argument("--pod", type=str, required=True, help="Pod ID to terminate")
    parser_term.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="RunPod API Key")

    args = parser.parse_args()

    if args.command == "status" or not args.command:
        get_status(getattr(args, "api_key", DEFAULT_API_KEY))
    elif args.command == "launch":
        launch_pod(
            gpu_type=args.gpu,
            image_name=args.image,
            pod_name=args.name,
            api_key=args.api_key,
            cloud_type=args.cloud_type
        )
    elif args.command == "terminate":
        terminate_pod(args.pod, args.api_key)


if __name__ == "__main__":
    main()
