"""Schema linting: detect enum choices the engine cannot score in one row.

The batched suffix pass gives each field one row keyed by the choice's first
token. Two choices that share a first token force the slower fallback: one
teacher-forced row per choice, scored and compared across different token
paths. ``lint_schema`` reports that (and related problems) before a schema
ships.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from openjev.schema import StructuredSchema


@dataclass(frozen=True)
class Finding:
    """One lint result for a schema field.

    Attributes:
        field: Name of the schema field the finding applies to.
        kind: One of ``"collision"``, ``"empty_choice"``, ``"duplicate_choice"``.
        message: Human-readable description of the problem.
        suggestion: A concrete rename, when one can be computed mechanically.
    """

    field: str
    kind: str
    message: str
    suggestion: str | None = None


def _rotation_suggestion(choices: list[str]) -> str | None:
    """Suggest renames that break a first-token tie, or None if the rule does not apply.

    Rule: when every colliding choice shares the same first underscore-separated
    word, move that word to the end so the distinguishing word leads
    (``BLOCK_TRANSACTION`` / ``BLOCK_USER`` -> ``TRANSACTION_BLOCK`` / ``USER_BLOCK``).
    This is best-effort; if any choice is a single word, or the rotated forms
    would still collide, no suggestion is made.
    """
    parts_list = [choice.split("_") for choice in choices]
    if any(len(parts) < 2 for parts in parts_list):
        return None
    first_words = {parts[0] for parts in parts_list}
    if len(first_words) != 1:
        return None
    rotated = ["_".join(parts[1:] + parts[:1]) for parts in parts_list]
    if len(set(rotated)) != len(rotated):
        return None
    return ", ".join(rotated)


def lint_schema(schema: StructuredSchema, tokenizer) -> list[Finding]:
    """Lint a schema's enum choices for engine-visible problems.

    Token lists come from ``StructuredSchema.compile_batch_plan`` — the exact
    lists the engine scores — so the lint never re-tokenizes by hand and can
    never disagree with the engine about token boundaries.

    Only categorical enum fields are linted. Boolean and multi fields decide
    true/false per option, so their choice token lists are fixed and cannot
    collide.

    Checks per enum field:
    - collision: two or more choices share the same first token. Scoring is
      still correct, but the engine falls back to one teacher-forced row per
      choice and compares them across different token paths — n rows instead
      of one.
    - duplicate_choice: the same literal appears more than once.
    - empty_choice: a choice adds no tokens beyond the shared prefix, so it is
      scored via the closing-quote token rather than value text.
    """
    findings: list[Finding] = []
    plan = schema.compile_batch_plan(tokenizer)

    for fname, fdef in schema.fields.items():
        if fdef.field_type not in ("enum", "choice", "selection"):
            continue
        entry = plan[fname]

        counts = Counter(fdef.choices)
        for choice, count in counts.items():
            if count > 1:
                findings.append(
                    Finding(
                        field=fname,
                        kind="duplicate_choice",
                        message=(
                            f'choice "{choice}" appears {count} times; the engine '
                            "cannot distinguish duplicate choices"
                        ),
                    )
                )

        groups: dict[int, list[int]] = {}
        for idx, tokens in enumerate(entry["choice_token_lists"]):
            groups.setdefault(tokens[0], []).append(idx)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            colliding = [fdef.choices[i] for i in indices]
            findings.append(
                Finding(
                    field=fname,
                    kind="collision",
                    message=(
                        "choices share their first token; the engine scores them with "
                        "one row per choice (slower). Rename to keep one row per "
                        f"field: {', '.join(colliding)}"
                    ),
                    suggestion=_rotation_suggestion(colliding),
                )
            )

        for choice in fdef.choices:
            if choice == entry["prefix"]:
                findings.append(
                    Finding(
                        field=fname,
                        kind="empty_choice",
                        message=(
                            f'choice "{choice}" adds no tokens beyond the shared '
                            "prefix; its score comes from the closing-quote token "
                            "rather than value text"
                        ),
                    )
                )

    return findings
