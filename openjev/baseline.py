"""Naive-JSON baseline for the eval harness (OpenAI-compatible chat API).

The eval compares openjev's constrained path against the same model (or a
bigger API model) writing the whole JSON object itself: same schema, same
context, same information. Bad model output is a measurement, not a bug —
parsing never raises on malformed output, it reports errors instead.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from openjev.schema import StructuredSchema

__all__ = [
    "BaselineError",
    "baseline_decide",
    "build_baseline_messages",
    "call_chat_completions",
    "parse_baseline_output",
]


class BaselineError(RuntimeError):
    """A chat-completions call failed (non-2xx response)."""

    def __init__(self, status: int, body: str):
        self.status = status
        snippet = body[:200]
        super().__init__(f"chat/completions returned {status}: {snippet}")


def _field_line(name: str, field) -> str:
    """One 'key: type and allowed values' line for the instruction block."""
    if field.field_type == "boolean":
        return f'- "{name}" (boolean: true or false) — {field.description}'
    if field.field_type == "multi":
        choices = ", ".join(f'"{c}"' for c in field.choices)
        return (
            f'- "{name}" (array whose items are exactly one or more of: {choices})'
            f" — {field.description}"
        )
    choices = ", ".join(f'"{c}"' for c in field.choices)
    return f'- "{name}" (one of exactly: {choices}) — {field.description}'


def build_baseline_messages(schema: StructuredSchema, context: str) -> list[dict]:
    """Build the single-user-message prompt for the naive-JSON baseline.

    Deterministic text, no system role: instruct the model to output ONLY a
    JSON object with exactly the schema's keys, then give it the context.
    """
    field_lines = "\n".join(_field_line(name, field) for name, field in schema.fields.items())
    content = (
        "You will be given a context and a list of fields to decide.\n"
        "Output ONLY a JSON object — no markdown, no explanation — with exactly these keys:\n"
        f"{field_lines}\n\n"
        "Context:\n"
        f"{context}"
    )
    return [{"role": "user", "content": content}]


def call_chat_completions(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    api_key: str | None,
    timeout: float = 120.0,
    temperature: float = 0.0,
) -> str:
    """POST to ``{base_url}/chat/completions`` and return the assistant content.

    stdlib only. Sends ``response_format={"type": "json_object"}`` best-effort
    (ignored by servers that don't support it). Raises BaselineError on any
    non-2xx response with the status and the first 200 chars of the body.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise BaselineError(e.code, e.read().decode("utf-8", errors="replace")) from e
    choices = json.loads(body).get("choices") or []
    if not choices:
        raise BaselineError(200, f"no choices in response body: {body[:200]}")
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def parse_baseline_output(text: str, schema: StructuredSchema) -> tuple[dict, list[str]]:
    """Extract, parse, and schema-validate a baseline model's JSON output.

    Returns ``(values, errors)`` where values maps each schema field to its
    parsed value (None for fields that failed) and errors is a list of
    human-readable problem strings. Never raises on bad model output.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return {name: None for name in schema.get_field_names()}, ["no JSON object found in output"]
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {name: None for name in schema.get_field_names()}, [f"invalid JSON: {e}"]
    if not isinstance(parsed, dict):
        return {name: None for name in schema.get_field_names()}, ["JSON value is not an object"]

    values: dict = {}
    errors: list[str] = []
    for name, field in schema.fields.items():
        if name not in parsed:
            values[name] = None
            errors.append(f"missing key: {name}")
            continue
        raw = parsed[name]
        if field.field_type == "boolean":
            if isinstance(raw, bool):
                values[name] = raw
            else:
                values[name] = None
                errors.append(f"wrong type for {name}: expected boolean, got {type(raw).__name__}")
        elif field.field_type == "multi":
            if not isinstance(raw, list):
                values[name] = None
                errors.append(f"wrong type for {name}: expected array, got {type(raw).__name__}")
                continue
            invalid = [
                item for item in raw if not isinstance(item, str) or item not in field.choices
            ]
            if invalid:
                values[name] = None
                errors.append(f"invalid items for {name}: {invalid} (allowed: {field.choices})")
            else:
                values[name] = raw
        else:  # enum
            if isinstance(raw, str) and raw in field.choices:
                values[name] = raw
            else:
                values[name] = None
                got = repr(raw)
                errors.append(f"invalid value for {name}: {got} (allowed: {field.choices})")
    return values, errors


def baseline_decide(
    base_url: str,
    model: str,
    api_key: str | None,
    schema: StructuredSchema,
    context: str,
) -> dict:
    """One baseline decision: prompt, call, parse, and time it.

    Returns ``{"values", "errors", "raw", "latency_ms", "schema_valid"}``;
    ``schema_valid`` is True iff the parse produced no errors. Never raises
    on bad model output — only on transport-level failures (BaselineError).
    """
    messages = build_baseline_messages(schema, context)
    t0 = time.perf_counter()
    raw = call_chat_completions(base_url, model, messages, api_key=api_key)
    latency_ms = (time.perf_counter() - t0) * 1000
    values, errors = parse_baseline_output(raw, schema)
    return {
        "values": values,
        "errors": errors,
        "raw": raw,
        "latency_ms": latency_ms,
        "schema_valid": not errors,
    }
