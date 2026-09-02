#!/usr/bin/env python3
"""Wait for an exact OpenAI-compatible model identity and context."""
from __future__ import annotations

import argparse
import ipaddress
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--container")
    parser.add_argument("--process-pid", type=int)
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.base_url)
    host_allowed = parsed.hostname == "localhost"
    if parsed.hostname and not host_allowed:
        try:
            address = ipaddress.ip_address(parsed.hostname)
            host_allowed = address.is_loopback or address.is_private
        except ValueError:
            host_allowed = False
    if (
        parsed.scheme != "http"
        or not host_allowed
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        parser.error("--base-url must be private/loopback HTTP without credentials or a path")
    if args.context <= 0 or args.timeout <= 0 or args.interval <= 0 or (args.process_pid is not None and args.process_pid <= 0):
        parser.error("context, timeout, interval, and process PID must be positive")
    return args


def get(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def ready(base_url: str, model: str, context: int) -> bool:
    health_status, _ = get(base_url.rstrip("/") + "/health")
    if health_status != 200:
        return False
    model_status, body = get(base_url.rstrip("/") + "/v1/models")
    if model_status != 200:
        return False
    payload = json.loads(body)
    entries = payload.get("data")
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(entry, dict)
        and entry.get("id") == model
        and entry.get("max_model_len") == context
        for entry in entries
    )


def container_state(name: str) -> str:
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                name,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return "docker-unavailable"
    if result.returncode == 127:
        return "docker-unavailable"
    if result.returncode != 0:
        return "container-unavailable"
    value = result.stdout.strip()
    if value == "false" or value.startswith("false|"):
        return "exited"
    if value in ("true", "true|none"):
        return "running"
    if value == "true|healthy":
        return "healthy"
    if value == "true|unhealthy":
        return "unhealthy"
    if value == "true|starting":
        return "starting"
    return "container-unavailable"


def process_running(pid: int) -> bool:
    try:
        state = open(f"/proc/{pid}/stat", encoding="utf-8").read().split()[2]
    except (FileNotFoundError, IndexError, OSError):
        return False
    return state != "Z"


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    container_seen = False
    while True:
        container_ok = args.container is None
        if args.container:
            state = container_state(args.container)
            if state == "docker-unavailable":
                print("docker unavailable during readiness", file=sys.stderr)
                return 1
            if state == "exited" or (state == "container-unavailable" and container_seen):
                print("container exited before readiness", file=sys.stderr)
                return 1
            if state == "unhealthy":
                print("container became unhealthy before readiness", file=sys.stderr)
                return 1
            if state in ("running", "healthy", "starting"):
                container_seen = True
                # A configured healthcheck is the release-warmup gate. Containers
                # without one retain legacy running-state behavior.
                container_ok = state in ("running", "healthy")
            elif args.process_pid is not None and not process_running(args.process_pid):
                print("candidate process exited before container readiness", file=sys.stderr)
                return 1
        try:
            if container_ok and ready(args.base_url, args.model, args.context):
                print(f"READY model={args.model} context={args.context}")
                return 0
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
            pass
        if time.monotonic() >= deadline:
            print("readiness timeout: health/model/context gate did not pass", file=sys.stderr)
            return 1
        time.sleep(min(args.interval, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
