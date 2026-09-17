"""Tests for openjev.baseline: prompt building, parsing, HTTP call. No network."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from openjev.baseline import (
    BaselineError,
    baseline_decide,
    build_baseline_messages,
    call_chat_completions,
    parse_baseline_output,
)
from openjev.schema import StructuredSchema

SCHEMA = StructuredSchema(
    {
        "action": {
            "type": "enum",
            "choices": ["APPROVE", "REJECT", "ESCALATE"],
            "description": "Decision on the transaction.",
        },
        "amount_valid": {"type": "boolean", "description": "Is the amount plausible?"},
        "tags": {"type": "multi", "choices": ["fraud", "velocity"], "description": "Risk tags."},
    }
)


class TestBuildBaselineMessages:
    def test_exact_text_for_three_field_schema(self):
        messages = build_baseline_messages(SCHEMA, "Wire of $9,000 to a new payee.")
        assert messages == [
            {
                "role": "user",
                "content": (
                    "You will be given a context and a list of fields to decide.\n"
                    "Output ONLY a JSON object — no markdown, no explanation —"
                    " with exactly these keys:\n"
                    '- "action" (one of exactly: "APPROVE", "REJECT", "ESCALATE")'
                    " — Decision on the transaction.\n"
                    '- "amount_valid" (boolean: true or false) — Is the amount plausible?\n'
                    '- "tags" (array whose items are exactly one or more of: "fraud", "velocity")'
                    " — Risk tags.\n"
                    "\n"
                    "Context:\n"
                    "Wire of $9,000 to a new payee."
                ),
            }
        ]
        assert len(messages) == 1 and messages[0]["role"] == "user"


class TestParseBaselineOutput:
    def test_valid_output(self):
        values, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": ["fraud"]}', SCHEMA
        )
        assert errors == []
        assert values == {"action": "APPROVE", "amount_valid": True, "tags": ["fraud"]}

    def test_json_inside_prose_is_extracted(self):
        values, errors = parse_baseline_output(
            'Sure! Here is the result:\n{"action": "REJECT", "amount_valid": false, "tags": []}\n',
            SCHEMA,
        )
        assert errors == []
        assert values["action"] == "REJECT"

    def test_malformed_json_reports_error_and_never_raises(self):
        values, errors = parse_baseline_output('{"action": "APPROVE", "amount_valid": tru}', SCHEMA)
        assert values == {"action": None, "amount_valid": None, "tags": None}
        assert len(errors) == 1 and errors[0].startswith("invalid JSON")

    def test_no_braces_at_all(self):
        values, errors = parse_baseline_output("I cannot help with that.", SCHEMA)
        assert set(values) == {"action", "amount_valid", "tags"}
        assert errors == ["no JSON object found in output"]

    def test_wrong_enum_value(self):
        values, errors = parse_baseline_output(
            '{"action": "MAYBE", "amount_valid": true, "tags": []}', SCHEMA
        )
        assert values["action"] is None
        assert any("invalid value for action" in e and "MAYBE" in e for e in errors)

    def test_extra_keys_are_ignored_not_errors(self):
        values, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": [], "notes": "hi"}', SCHEMA
        )
        assert errors == []
        assert "notes" not in values

    def test_missing_key(self):
        values, errors = parse_baseline_output('{"action": "APPROVE"}', SCHEMA)
        assert values["amount_valid"] is None and values["tags"] is None
        assert sorted(errors) == ["missing key: amount_valid", "missing key: tags"]

    def test_boolean_wrong_type(self):
        values, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": "yes", "tags": []}', SCHEMA
        )
        assert values["amount_valid"] is None
        assert any("wrong type for amount_valid" in e for e in errors)

    def test_multi_item_not_in_choices(self):
        values, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": ["fraud", "nope"]}', SCHEMA
        )
        assert values["tags"] is None
        assert any("invalid items for tags" in e and "nope" in e for e in errors)


class _CannedHandler(BaseHTTPRequestHandler):
    """Returns a canned OpenAI-style body; records the last request."""

    last_headers: dict = {}
    last_body: dict = {}
    status = 200
    payload = {"choices": [{"message": {"content": '{"action": "APPROVE"}'}}]}

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        type(self).last_headers = dict(self.headers)
        type(self).last_body = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps(type(self).payload).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output clean
        pass


class TestCallChatCompletions:
    @pytest.fixture()
    def server(self):
        httpd = HTTPServer(("127.0.0.1", 0), _CannedHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_port}", httpd
        httpd.shutdown()
        httpd.server_close()

    def test_returns_assistant_content(self, server):
        base_url, _ = server
        content = call_chat_completions(
            base_url, "glm-5.3", [{"role": "user", "content": "x"}], api_key=None
        )
        assert content == '{"action": "APPROVE"}'

    def test_bearer_header_and_payload_shape(self, server):
        base_url, _ = server
        call_chat_completions(
            base_url + "/", "glm-5.3", [{"role": "user", "content": "x"}], api_key="secret-key"
        )
        assert _CannedHandler.last_headers.get("Authorization") == "Bearer secret-key"
        assert _CannedHandler.last_body["model"] == "glm-5.3"
        assert _CannedHandler.last_body["response_format"] == {"type": "json_object"}
        assert _CannedHandler.last_body["temperature"] == 0.0

    def test_no_bearer_header_without_api_key(self, server):
        base_url, _ = server
        call_chat_completions(base_url, "glm-5.3", [{"role": "user", "content": "x"}], api_key=None)
        assert "Authorization" not in _CannedHandler.last_headers

    def test_500_raises_baseline_error_with_status_and_snippet(self, server):
        base_url, _ = server
        _CannedHandler.status = 500
        _CannedHandler.payload = {"error": "x" * 500}
        try:
            with pytest.raises(BaselineError) as excinfo:
                call_chat_completions(
                    base_url, "glm-5.3", [{"role": "user", "content": "x"}], api_key=None
                )
            assert excinfo.value.status == 500
            assert len(str(excinfo.value)) < 260  # first 200 chars of body + prefix
        finally:
            _CannedHandler.status = 200
            _CannedHandler.payload = {"choices": [{"message": {"content": "{}"}}]}

    def test_error_is_runtimeerror(self):
        assert issubclass(BaselineError, RuntimeError)


class TestBaselineDecide:
    def test_end_to_end_against_canned_server(self):
        httpd = HTTPServer(("127.0.0.1", 0), _CannedHandler)
        _CannedHandler.payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"action": "APPROVE", "amount_valid": true, "tags": ["fraud"]}'
                    }
                }
            ]
        }
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            result = baseline_decide(
                f"http://127.0.0.1:{httpd.server_port}",
                "glm-5.3",
                None,
                SCHEMA,
                "wire transfer context",
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
        assert result["schema_valid"] is True
        assert result["errors"] == []
        assert result["values"] == {
            "action": "APPROVE",
            "amount_valid": True,
            "tags": ["fraud"],
        }
        assert result["latency_ms"] >= 0.0 and isinstance(result["raw"], str)
