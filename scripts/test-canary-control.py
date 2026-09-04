#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAIT = ROOT / "scripts" / "wait-ready.py"
CANARY = ROOT / "scripts" / "run-canary.sh"


class Handler(BaseHTTPRequestHandler):
    alias = "overlord-testing"
    context = 262144
    healthy = True

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_response(200 if self.healthy else 503)
            self.end_headers()
            return
        if self.path == "/v1/models":
            body = json.dumps(
                {"data": [{"id": self.alias, "max_model_len": self.context}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class WaitReadyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def run_wait(self, *, model: str = "overlord-testing", context: int = 262144):
        return subprocess.run(
            [
                sys.executable,
                str(WAIT),
                "--base-url",
                self.base,
                "--model",
                model,
                "--context",
                str(context),
                "--timeout",
                "0.25",
                "--interval",
                "0.02",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_exact_health_model_and_context(self) -> None:
        result = self.run_wait()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "READY model=overlord-testing context=262144")

    def test_accepts_private_address_but_rejects_public_or_hostname(self) -> None:
        common = [
            sys.executable, str(WAIT), "--model", "m", "--context", "1",
            "--timeout", "0.01", "--interval", "0.01",
        ]
        private = subprocess.run(
            common + ["--base-url", "http://10.255.255.254:1"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(private.returncode, 1, private.stderr)
        for url in ("http://8.8.8.8:1", "http://example.com:1", "https://127.0.0.1:1"):
            result = subprocess.run(
                common + ["--base-url", url], text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 2, (url, result.stderr))

    def test_rejects_wrong_alias(self) -> None:
        result = self.run_wait(model="wrong")
        self.assertEqual(result.returncode, 1)
        self.assertIn("readiness timeout", result.stderr)
        self.assertNotIn("overlord-testing", result.stderr)

    def test_rejects_wrong_context(self) -> None:
        result = self.run_wait(context=131072)
        self.assertEqual(result.returncode, 1)
        self.assertIn("readiness timeout", result.stderr)

    def test_missing_container_waits_but_docker_unavailable_fails_fast(self) -> None:
        cases = ((1, "readiness timeout", "0.05"), (127, "docker unavailable", "5"))
        for exit_code, expected, timeout in cases:
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                docker = Path(directory) / "docker"
                docker.write_text(f"#!/bin/sh\nexit {exit_code}\n")
                docker.chmod(0o755)
                started = time.monotonic()
                result = subprocess.run(
                    [sys.executable, str(WAIT), "--base-url", self.base,
                     "--model", "overlord-testing", "--context", "262144",
                     "--timeout", timeout, "--interval", "0.01", "--container", "missing"],
                    env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]},
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)
                self.assertLess(time.monotonic() - started, 1.0)

    def test_missing_container_fails_fast_when_candidate_process_exited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text("#!/bin/sh\nexit 1\n")
            docker.chmod(0o755)
            result = subprocess.run(
                [sys.executable, str(WAIT), "--base-url", self.base,
                 "--model", "overlord-testing", "--context", "262144",
                 "--timeout", "5", "--interval", "0.01", "--container", "missing",
                 "--process-pid", "99999999"],
                env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]},
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("candidate process exited", result.stderr)

    def test_fails_fast_when_container_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text("#!/bin/sh\n[ \"$1\" = inspect ] || exit 2\necho false\n")
            docker.chmod(0o755)
            started = time.monotonic()
            result = subprocess.run(
                [
                    sys.executable,
                    str(WAIT),
                    "--base-url",
                    self.base,
                    "--model",
                    "overlord-testing",
                    "--context",
                    "262144",
                    "--timeout",
                    "5",
                    "--interval",
                    "0.02",
                    "--container",
                    "candidate",
                ],
                env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIn("container exited before readiness", result.stderr)

    def test_container_health_must_finish_release_warmup(self) -> None:
        for state, expected in (("true|starting", 1), ("true|healthy", 0)):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                docker = Path(directory) / "docker"
                docker.write_text(f"#!/bin/sh\necho '{state}'\n")
                docker.chmod(0o755)
                result = subprocess.run(
                    [sys.executable, str(WAIT), "--base-url", self.base,
                     "--model", "overlord-testing", "--context", "262144",
                     "--timeout", "0.08", "--interval", "0.01", "--container", "candidate"],
                    env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]},
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, expected, result.stderr)


class RunCanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.log = self.dir / "events.log"
        self.control = self.dir / "control.env"
        self.candidate = self.dir / "candidate.env"
        common = (
            "MODEL_DIR=/model\nDRAFT_DIR=/draft\nCACHE_DIR=/cache\nGPU_DEVICES=0,2\n"
            "BIND_ADDRESS=127.0.0.1\nSERVED_MODEL_NAME=overlord-testing\n"
            "MAX_MODEL_LEN=1048576\n"
        )
        self.control.write_text(common + "PORT=8000\nCONTAINER_NAME=control\n")
        self.candidate.write_text(common + "PORT=8002\nCONTAINER_NAME=candidate\n")
        self.systemctl = self.make_script(
            "systemctl", 'printf "systemctl %s\\n" "$*" >> "$EVENT_LOG"\n'
        )
        self.docker = self.make_script(
            "docker", 'printf "docker %s\\n" "$*" >> "$EVENT_LOG"\n'
        )
        self.wait = self.make_script(
            "wait", 'printf "wait %s\\n" "$*" >> "$EVENT_LOG"\n'
        )
        self.serve = self.make_script(
            "serve",
            'printf "serve-start %s\\n" "$ENV_FILE" >> "$EVENT_LOG"\n'
            'trap \'printf "serve-stop\\n" >> "$EVENT_LOG"; exit 0\' INT TERM\n'
            "while :; do sleep 0.05; done\n",
        )
        self.validate = self.dir / "validate.sh"
        self.validate.write_text('printf "validate %s\\n" "$ENV_FILE" >> "$EVENT_LOG"\n')
        self.check_ok = self.make_script(
            "check-ok", 'printf "check-ok\\n" >> "$EVENT_LOG"\n'
        )
        self.check_fail = self.make_script(
            "check-fail", 'printf "check-fail\\n" >> "$EVENT_LOG"\nexit 7\n'
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_script(self, name: str, body: str) -> Path:
        path = self.dir / name
        path.write_text("#!/bin/sh\nset -eu\n" + body)
        path.chmod(0o755)
        return path

    def run_canary(self, check: Path) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "CONTROL_ENV": str(self.control),
            "SYSTEMCTL_BIN": str(self.systemctl),
            "DOCKER_BIN": str(self.docker),
            "WAIT_SCRIPT": str(self.wait),
            "SERVE_SCRIPT": str(self.serve),
            "VALIDATE_SCRIPT": str(self.validate),
            "EVENT_LOG": str(self.log),
            "CANARY_ARTIFACT_DIR": str(self.dir / "artifacts"),
            "SERVICE_UNIT": "unit.service",
        }
        return subprocess.run(
            ["bash", str(CANARY), str(self.candidate), "--", str(check)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_real_systemctl_gets_user_bus_defaults(self) -> None:
        text = CANARY.read_text()
        self.assertIn('XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"', text)
        self.assertIn('DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path:${XDG_RUNTIME_DIR}/bus}"', text)

    def test_service_template_has_exact_readiness_gate(self) -> None:
        template = (ROOT / "systemd" / "glm53-2x-rtxpro6000.service.in").read_text()
        self.assertIn(
            "ExecStartPost=@REPO_DIR@/scripts/wait-ready.py --base-url http://@READINESS_HOST@:@PORT@ --model @SERVED_MODEL_NAME@ --context @MAX_MODEL_LEN@ --timeout 1800 --container @CONTAINER_NAME@",
            template,
        )

    def test_installer_renders_all_readiness_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            binary = Path(directory) / "bin"
            home.mkdir()
            binary.mkdir()
            systemctl = binary / "systemctl"
            systemctl.write_text("#!/bin/sh\nexit 0\n")
            systemctl.chmod(0o755)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "install-user-service.sh")],
                env={
                    **os.environ,
                    "HOME": str(home),
                    "ENV_FILE": str(self.control),
                    "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                },
                text=True,
                capture_output=True,
                check=False,
            )
            unit = home / ".config" / "systemd" / "user" / "glm53-2x-rtxpro6000.service"
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = unit.read_text()
        self.assertNotIn("@PORT@", rendered)
        self.assertNotIn("@SERVED_MODEL_NAME@", rendered)
        self.assertNotIn("@MAX_MODEL_LEN@", rendered)
        self.assertNotIn("@CONTAINER_NAME@", rendered)
        self.assertNotIn("@READINESS_HOST@", rendered)
        self.assertIn("--base-url http://127.0.0.1:8000", rendered)
        self.assertIn("--model overlord-testing", rendered)
        self.assertIn("--context 1048576", rendered)

    def test_success_restores_control_after_checks(self) -> None:
        result = self.run_canary(self.check_ok)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.log.read_text().splitlines()
        self.assertLess(events.index(f"validate {self.candidate}"), events.index("systemctl --user stop unit.service"))
        self.assertLess(events.index("check-ok"), events.index("systemctl --user start unit.service"))
        self.assertTrue(any(line.startswith("wait --base-url http://127.0.0.1:8000") for line in events))
        artifact = self.dir / "artifacts" / "candidate.log"
        self.assertTrue(artifact.is_file())
        self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)

    def test_wildcard_control_bind_restores_through_loopback_readiness(self) -> None:
        text = self.control.read_text().replace(
            "BIND_ADDRESS=127.0.0.1", "BIND_ADDRESS=0.0.0.0"
        )
        self.control.write_text(text)
        result = self.run_canary(self.check_ok)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.log.read_text().splitlines()
        self.assertTrue(
            any(
                line.startswith("wait --base-url http://127.0.0.1:8000")
                for line in events
            )
        )

    def test_failed_checks_still_restore_control(self) -> None:
        result = self.run_canary(self.check_fail)
        self.assertEqual(result.returncode, 7)
        events = self.log.read_text().splitlines()
        self.assertIn("check-fail", events)
        self.assertIn("systemctl --user start unit.service", events)
        self.assertTrue(any(line == "docker rm -f candidate" for line in events))

    def test_hup_is_trapped_for_restoration(self) -> None:
        self.assertIn("trap 'exit 129' HUP", CANARY.read_text())

    def test_restoration_failure_overrides_check_status(self) -> None:
        self.systemctl.write_text(
            "#!/bin/sh\nset -eu\nprintf 'systemctl %s\\n' \"$*\" >> \"$EVENT_LOG\"\n"
            "case \"$*\" in *' start '*) exit 9;; esac\n"
        )
        self.systemctl.chmod(0o755)
        result = self.run_canary(self.check_fail)
        self.assertEqual(result.returncode, 125)
        self.assertIn("RESTORATION FAILED", result.stderr)

    def test_partial_stop_failure_still_restores_control(self) -> None:
        self.systemctl.write_text(
            "#!/bin/sh\nset -eu\nprintf 'systemctl %s\\n' \"$*\" >> \"$EVENT_LOG\"\n"
            "case \"$*\" in *' stop '*) exit 9;; esac\n"
        )
        self.systemctl.chmod(0o755)
        result = self.run_canary(self.check_ok)
        self.assertEqual(result.returncode, 9)
        events = self.log.read_text().splitlines()
        self.assertIn("systemctl --user start unit.service", events)
        self.assertTrue(any(line.startswith("wait --base-url") for line in events))

    def test_inactive_control_fails_before_stop_or_candidate_launch(self) -> None:
        self.systemctl.write_text(
            "#!/bin/sh\nset -eu\nprintf 'systemctl %s\\n' \"$*\" >> \"$EVENT_LOG\"\n"
            "case \"$*\" in *' is-active '*) exit 3;; esac\n"
        )
        self.systemctl.chmod(0o755)
        result = self.run_canary(self.check_ok)
        self.assertEqual(result.returncode, 2)
        events = self.log.read_text().splitlines()
        self.assertFalse(any(' stop ' in f' {line} ' for line in events))
        self.assertFalse(any(line.startswith('serve-start') for line in events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
