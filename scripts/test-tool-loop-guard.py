#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "tool-loop-guard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tool_loop_guard", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def failed_cycle(call_id: str, command: str) -> list[dict]:
    arguments = json.dumps(
        {
            "cmd": command,
            "yield_time_ms": 10_000,
            "max_output_tokens": 1_000,
        }
    )
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": "Process exited with code 1\nOriginal token count: 0\nOutput:\n",
        },
    ]


class ToolLoopGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = load_module()
        self.command = 'rg -n -i "coords" "Content\\WorldMap\\Blueprints\\WhereToGo.uasset" 2>$null'

    def payload(self, cycles: int) -> dict:
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Inspect the world map."},
        ]
        for index in range(cycles):
            messages.extend(failed_cycle(f"call_{index}", self.command))
        return {
            "model": "overlord-testing",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "stream": True,
        }

    def test_two_identical_failed_cycles_arm_guard_and_raise_penalty(self):
        protected, state = self.guard.protect_payload(self.payload(2))
        self.assertTrue(state.armed)
        self.assertEqual(protected["repetition_penalty"], 1.1)
        self.assertTrue(protected["stream"])
        self.assertTrue(state.client_requested_stream)

    def test_one_failed_cycle_does_not_arm_guard_but_gets_baseline_penalty(self):
        protected, state = self.guard.protect_payload(self.payload(1))
        self.assertFalse(state.armed)
        self.assertEqual(protected["repetition_penalty"], 1.05)
        self.assertTrue(protected["stream"])

    def test_plain_chat_without_tools_does_not_receive_agent_penalty(self):
        payload = {
            "model": "overlord-testing",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        protected, state = self.guard.protect_payload(payload)
        self.assertFalse(state.armed)
        self.assertNotIn("repetition_penalty", protected)

    def test_existing_stronger_penalty_is_preserved(self):
        payload = self.payload(1)
        payload["repetition_penalty"] = 1.2
        protected, _ = self.guard.protect_payload(payload)
        self.assertEqual(protected["repetition_penalty"], 1.2)

    def test_new_user_turn_resets_old_repeated_failure_state(self):
        payload = self.payload(2)
        payload["messages"].append({"role": "user", "content": "Now do a different task."})
        protected, state = self.guard.protect_payload(payload)
        self.assertFalse(state.armed)
        self.assertEqual(protected["repetition_penalty"], 1.05)
        self.assertTrue(protected["stream"])

    def test_changed_failure_output_allows_retry(self):
        payload = self.payload(2)
        payload["messages"][-1]["content"] = (
            "Process exited with code 1\nOutput:\nPermission denied\n"
        )
        protected, state = self.guard.protect_payload(payload)
        self.assertFalse(state.armed)
        self.assertEqual(protected["repetition_penalty"], 1.05)

    def test_successful_completed_turn_breaks_failure_sequence(self):
        payload = self.payload(2)
        payload["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "success",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": self.command}),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "success",
                    "content": "Process exited with code 0\nOutput:\nfound it\n",
                },
            ]
        )
        _, state = self.guard.protect_payload(payload)
        self.assertFalse(state.armed)

    def test_http_status_200_and_embedded_child_failure_are_successes(self):
        for content in (
            '{"status":200,"body":"ok"}',
            "Process exited with code 0\nOutput:\nchild exited with code 1",
        ):
            payload = self.payload(2)
            payload["messages"][-3]["content"] = content
            payload["messages"][-1]["content"] = content
            _, state = self.guard.protect_payload(payload)
            self.assertFalse(state.armed, content)

    def test_http_status_4xx_and_5xx_are_failures(self):
        for status in (400, 404, 429, 500, 503):
            payload = self.payload(2)
            content = json.dumps({"status": status, "body": "failed"})
            payload["messages"][-3]["content"] = content
            payload["messages"][-1]["content"] = content
            _, state = self.guard.protect_payload(payload)
            self.assertTrue(state.armed, status)

    def test_different_malformed_arguments_have_distinct_fingerprints(self):
        first = {"name": "exec_command", "arguments": "{bad-a"}
        second = {"name": "exec_command", "arguments": "{bad-b"}
        self.assertNotEqual(
            self.guard.function_fingerprint(first),
            self.guard.function_fingerprint(second),
        )

    def test_parallel_duplicates_in_one_assistant_turn_do_not_arm(self):
        payload = self.payload(0)
        payload["messages"] += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "parallel_1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": json.dumps({"cmd": self.command})},
                    },
                    {
                        "id": "parallel_2",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": json.dumps({"cmd": self.command})},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "parallel_1", "content": "Process exited with code 1\nOutput:\n"},
            {"role": "tool", "tool_call_id": "parallel_2", "content": "Process exited with code 1\nOutput:\n"},
        ]
        _, state = self.guard.protect_payload(payload)
        self.assertFalse(state.armed)

    def test_interleaved_parallel_turns_detect_repeated_failed_call(self):
        payload = self.payload(0)
        for turn in range(2):
            payload["messages"] += [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"repeat_{turn}",
                            "type": "function",
                            "function": {"name": "exec_command", "arguments": json.dumps({"cmd": self.command})},
                        },
                        {
                            "id": f"other_{turn}",
                            "type": "function",
                            "function": {"name": "exec_command", "arguments": json.dumps({"cmd": f"other-{turn}"})},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": f"repeat_{turn}", "content": "Process exited with code 1\nOutput:\n"},
                {"role": "tool", "tool_call_id": f"other_{turn}", "content": f"Process exited with code {turn + 2}\nOutput:\n"},
            ]
        _, state = self.guard.protect_payload(payload)
        self.assertTrue(state.armed)

    def test_metadata_corruption_does_not_change_exec_command_fingerprint(self):
        clean = {
            "name": "exec_command",
            "arguments": json.dumps({"cmd": self.command, "max_output_tokens": 1000}),
        }
        corrupted = {
            "name": "exec_command",
            "arguments": json.dumps(
                {
                    "cmd": self.command,
                    "max_output scripts: continue</arg_value><arg_key>max_output_tokens": "1000",
                }
            ),
        }
        self.assertEqual(
            self.guard.function_fingerprint(clean),
            self.guard.function_fingerprint(corrupted),
        )

    def test_repeated_upstream_call_is_replaced_with_terminal_non_tool_response(self):
        _, state = self.guard.protect_payload(self.payload(2))
        upstream = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "overlord-testing",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_again",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": json.dumps({"cmd": self.command}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        filtered, blocked = self.guard.sanitize_response(upstream, state)
        self.assertTrue(blocked)
        choice = filtered["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertIn("LOOP_GUARD", choice["message"]["content"])

    def test_different_upstream_call_is_allowed(self):
        _, state = self.guard.protect_payload(self.payload(2))
        upstream = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "overlord-testing",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_changed",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": json.dumps({"cmd": "Get-ChildItem Content\\WorldMap"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        filtered, blocked = self.guard.sanitize_response(upstream, state)
        self.assertFalse(blocked)
        self.assertEqual(filtered, upstream)

    def test_only_matching_choice_is_blocked(self):
        _, state = self.guard.protect_payload(self.payload(2))
        repeated = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "same",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": self.command}),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }
        safe = {
            "index": 1,
            "message": {"role": "assistant", "content": "safe alternative"},
            "finish_reason": "stop",
        }
        filtered, blocked = self.guard.sanitize_response(
            {"choices": [repeated, safe]}, state
        )
        self.assertTrue(blocked)
        self.assertIn(
            "LOOP_GUARD", filtered["choices"][0]["message"]["content"]
        )
        self.assertEqual(filtered["choices"][1], safe)

    def test_safe_stream_is_preserved_and_blocked_stream_is_valid(self):
        _, state = self.guard.protect_payload(self.payload(2))
        safe_event = {
            "id": "chatcmpl-safe",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "overlord-testing",
            "system_fingerprint": "fp_test",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "changed",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "Get-ChildItem"}),
                        },
                    }],
                },
                "logprobs": None,
                "finish_reason": "tool_calls",
            }],
        }
        safe_wire = (
            "data: " + json.dumps(safe_event) + "\n\ndata: [DONE]\n\n"
        ).encode()
        unchanged, blocked = self.guard.sanitize_sse_response(safe_wire, state)
        self.assertFalse(blocked)
        self.assertEqual(unchanged, safe_wire)

        repeated_event = json.loads(json.dumps(safe_event))
        repeated_event["choices"][0]["delta"]["tool_calls"][0]["function"][
            "arguments"
        ] = json.dumps({"cmd": self.command})
        repeated_wire = (
            "data: " + json.dumps(repeated_event) + "\n\ndata: [DONE]\n\n"
        ).encode()
        filtered, blocked = self.guard.sanitize_sse_response(repeated_wire, state)
        self.assertTrue(blocked)
        events = [
            json.loads(line[6:])
            for line in filtered.decode().splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(events[0]["system_fingerprint"], "fp_test")
        self.assertNotIn("tool_calls", events[0]["choices"][0]["delta"])
        self.assertIn(
            "LOOP_GUARD", events[0]["choices"][0]["delta"]["content"]
        )

    def test_guarded_nonstream_response_can_be_encoded_as_valid_sse(self):
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "overlord-testing",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "LOOP_GUARD: stopped"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
        wire = self.guard.encode_sse_response(response).decode()
        self.assertTrue(wire.endswith("data: [DONE]\n\n"))
        events = [
            json.loads(line.removeprefix("data: "))
            for line in wire.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "LOOP_GUARD: stopped")
        self.assertEqual(events[0]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[1]["choices"], [])
        self.assertEqual(events[1]["usage"]["total_tokens"], 13)


if __name__ == "__main__":
    unittest.main()
