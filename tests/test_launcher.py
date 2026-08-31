"""The demo launcher's pod lifecycle (tests/test_launcher.py).

A GPU pod left RUNNING bills exactly like a working one. The launcher used to
exit without stopping the pod it had just used, which is what took the RunPod
account to a 402 and left the demo with no GPU at all. These pin the two
halves of the fix: the pod is stopped on EVERY exit path, and a pod this
launcher did not discover is never touched.

Nothing here reaches the network - `_api` is replaced with a call recorder,
and a test that hit RunPod for real would be a test that costs money.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _launcher():
    """A fresh module instance, so one test's monkeypatching cannot leak."""
    spec = importlib.util.spec_from_file_location(
        "launch_demo_under_test", PROJECT_ROOT / "tools" / "launch_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv, *, client_result=0, discovered=("ws://1.2.3.4:5", "pod-abc")):
    """Run main() with the pod API, the server wait, and the client stubbed.

    Returns (exit_code, [pod ids the launcher asked to stop]).
    """
    mod = _launcher()
    stopped = []
    mod.stop_pod = lambda pod_id, api_key: stopped.append(pod_id) or True
    mod.discover_pod_url = lambda api_key, name: discovered
    mod.wait_for_server = lambda url, timeout_s=1200: True
    mod._api = lambda *a, **k: pytest.fail("the launcher reached the network")

    def _call(*a, **k):
        if isinstance(client_result, BaseException):
            raise client_result
        return client_result

    mod.subprocess = types.SimpleNamespace(call=_call)
    sys.argv = list(argv)
    return mod.main(), stopped


API = ["launch_demo.py", "--api-key", "KEY"]


@pytest.mark.parametrize("label,client_result,code", [
    ("a normal exit", 0, 0),
    ("a client crash", 1, 1),
    ("Ctrl-C", KeyboardInterrupt(), 130),
])
def test_a_discovered_pod_is_stopped_on_every_exit_path(label, client_result, code):
    rc, stopped = _run(API, client_result=client_result)
    assert rc == code
    assert stopped == ["pod-abc"], f"pod left running after {label}"


def test_keep_pod_leaves_it_running():
    """The opt-out, for back-to-back sessions where a 5-minute cold rebuild
    between them costs more than the idle minutes do."""
    rc, stopped = _run(API + ["--keep-pod"])
    assert rc == 0
    assert stopped == []


def test_a_pod_reached_by_explicit_url_is_never_stopped():
    """--server-url may point at a pod this launcher did not start and does
    not own; stopping it would take down someone else's session."""
    rc, stopped = _run(["launch_demo.py", "--server-url", "ws://h:1", "--no-wait"])
    assert stopped == []


def test_local_mode_touches_no_pod_at_all():
    rc, stopped = _run(["launch_demo.py", "--local"])
    assert rc == 0
    assert stopped == []


def test_a_pod_that_never_comes_up_is_still_stopped():
    """The worst case for the bill: the pod is up and charging, the server
    never finishes installing, and the launcher gives up."""
    mod = _launcher()
    stopped = []
    mod.stop_pod = lambda pod_id, api_key: stopped.append(pod_id) or True
    mod.discover_pod_url = lambda api_key, name: ("ws://1.2.3.4:5", "pod-abc")
    mod.wait_for_server = lambda url, timeout_s=1200: False
    mod.subprocess = types.SimpleNamespace(
        call=lambda *a, **k: pytest.fail("client started without a server"))
    sys.argv = list(API)
    assert mod.main() == 1
    assert stopped == ["pod-abc"], "a pod that never served was left billing"
