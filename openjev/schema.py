"""Schema definitions and batch-plan compilation for parallel constrained decisions.

Supports booleans, categorical enums (cardinality up to 255), and multi fields
(subset of choices, 2-64 options, decided as one boolean decision per option).
"""

import json
from typing import Any


def _common_token_prefix(sequences: list[list[int]]) -> list[int]:
    """Longest common token-ID prefix of every sequence (at least one required)."""
    if not sequences:
        return []
    shared: list[int] = []
    shortest = min(len(sequence) for sequence in sequences)
    for position in range(shortest):
        tokens_at = {sequence[position] for sequence in sequences}
        if len(tokens_at) != 1:
            break
        shared.append(sequences[0][position])
    return shared


class FieldDefinition:
    def __init__(
        self, name: str, field_type: str, description: str, choices: list[str] | None = None
    ):
        self.name = name
        self.field_type = field_type.lower()
        self.description = description

        if self.field_type == "boolean":
            self.choices = ["true", "false"]
        elif self.field_type == "multi":
            if not choices or len(choices) < 2:
                raise ValueError(f"Field '{name}' of type multi must have at least 2 choices.")
            if len(choices) > 64:
                raise ValueError(
                    f"Field '{name}' exceeds maximum cardinality of 64 choices "
                    f"for type multi (got {len(choices)})."
                )
            self.choices = choices
        elif self.field_type in ("enum", "choice", "selection"):
            if not choices or len(choices) == 0:
                raise ValueError(f"Field '{name}' of type enum must have choices defined.")
            if len(choices) > 255:
                raise ValueError(
                    f"Field '{name}' exceeds maximum cardinality of 255 choices "
                    f"(got {len(choices)})."
                )
            self.choices = choices
        else:
            raise ValueError(
                f"Unsupported field type '{field_type}'. "
                "Supported types: 'boolean', 'enum' and 'multi'."
            )

    @property
    def cardinality(self) -> int:
        return len(self.choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.field_type,
            "description": self.description,
            "choices": self.choices,
            "cardinality": self.cardinality,
        }


class StructuredSchema:
    def __init__(self, schema_dict: dict[str, Any]):
        self.fields: dict[str, FieldDefinition] = {}
        for field_name, spec in schema_dict.items():
            self.fields[field_name] = FieldDefinition(
                name=field_name,
                field_type=spec.get("type", "enum"),
                description=spec.get("description", ""),
                choices=spec.get("choices", None),
            )
        # Compiled plans, keyed by tokenizer identity: one schema object can be
        # reused with several models, and token IDs are tokenizer-specific.
        self._plans: dict[str, dict[str, Any]] = {}

    def get_field_names(self) -> list[str]:
        return list(self.fields.keys())

    def __getitem__(self, key: str) -> FieldDefinition:
        return self.fields[key]

    def __len__(self) -> int:
        return len(self.fields)

    def to_json_schema_prompt_str(self) -> str:
        """Returns a clean TypeScript/JSON schema representation for naive LLM prompting."""
        lines = ["{"]
        for name, field in self.fields.items():
            if field.field_type == "boolean":
                lines.append(f'  "{name}": boolean, // {field.description}')
            elif field.field_type == "multi":
                choices_str = " | ".join(f'"{c}"' for c in field.choices)
                lines.append(
                    f'  "{name}": [{choices_str}], // {field.description} (select all that apply)'
                )
            else:
                choices_limit = 20 if len(field.choices) > 50 else len(field.choices)
                choices_str = " | ".join(f'"{c}"' for c in field.choices[:choices_limit])
                if len(field.choices) > choices_limit:
                    choices_str += f" | ... ({len(field.choices)} total options)"
                lines.append(f'  "{name}": {choices_str}, // {field.description}')
        lines.append("}")
        return "\n".join(lines)

    def to_parallel_schema_str(self) -> str:
        """Returns a high-density, compact description catalog for minimal prefill token latency."""
        lines = []
        for name, field in self.fields.items():
            desc = field.description.split("\n")[0].strip()
            if field.field_type == "multi":
                desc += " (select all that apply)"
            lines.append(f'  "{name}": {desc}')
        return "\n".join(lines)

    def _tokenizer_key(self, tokenizer) -> str:
        """Stable cache key for one tokenizer identity."""
        name = getattr(tokenizer, "name_or_path", None) or repr(type(tokenizer))
        try:
            size = str(len(tokenizer))
        except TypeError:
            size = ""
        return f"{name}:{size}"

    def compile_batch_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Pre-index everything the engine needs for the batched suffix pass.

        Token-aligned: every choice is encoded as its complete, JSON-escaped
        row candidate (``  "name": "choice"`` including the closing quote;
        the bare literal for booleans) in one tokenization pass, then the
        field's shared token prefix and per-choice remainders are computed on
        token IDs. Encoding the prefix and the remainder separately would be
        wrong: BPE merges are not compositional, so the concatenated ids would
        not be the tokenization of the full candidate.

        Per enum/boolean field the plan carries ``shared_ids`` (the common
        token-ID prefix of all candidates) and per-choice ``remainders``.
        Per multi field it carries one ``suffix_ids_list`` entry per option
        (``  "name.option": ``) plus the shared true/false token lists.
        Plans are cached per tokenizer identity (name_or_path + vocab size).
        """
        key = self._tokenizer_key(tokenizer)
        cached = self._plans.get(key)
        if cached is not None:
            return cached

        plan: dict[str, dict[str, Any]] = {}
        for fname, fdef in self.fields.items():
            if fdef.field_type == "multi":
                # One boolean row per option: suffix '  "<field>.<option>": '
                # scored against the true/false literals. The engine folds the
                # per-option probabilities back into the selected subset. The
                # true/false remainders come from the token-aligned rule: the
                # option rows share their suffix, so the remainders are the
                # full ' true'/' false' sequences with the common token prefix
                # removed.
                full_lists = [
                    tokenizer.encode(
                        f"  {json.dumps(f'{fname}.{option}')}: {literal}",
                        add_special_tokens=False,
                    )
                    for option in fdef.choices
                    for literal in ("true", "false")
                ]
                shared = _common_token_prefix(full_lists)
                plan[fname] = {
                    "options": list(fdef.choices),
                    "suffix_ids_list": [
                        tokenizer.encode(
                            f"  {json.dumps(f'{fname}.{option}')}: ",
                            add_special_tokens=False,
                        )
                        for option in fdef.choices
                    ],
                    "shared_ids": shared,
                    "remainders": [full[len(shared) :] for full in full_lists],
                }
                continue

            if fdef.field_type == "boolean":
                candidates = [
                    tokenizer.encode(f"  {json.dumps(fname)}: {literal}", add_special_tokens=False)
                    for literal in ("true", "false")
                ]
            else:
                candidates = [
                    tokenizer.encode(
                        f"  {json.dumps(fname)}: {json.dumps(choice)}", add_special_tokens=False
                    )
                    for choice in fdef.choices
                ]

            shared = _common_token_prefix(candidates)
            remainders = [full[len(shared) :] for full in candidates]
            for i, remainder in enumerate(remainders):
                for j, other in enumerate(remainders):
                    if i == j or other[: len(remainder)] != remainder:
                        continue
                    if other == remainder:
                        raise ValueError(
                            f"field '{fname}': choices '{fdef.choices[i]}' and "
                            f"'{fdef.choices[j]}' are token-identical; the engine "
                            "cannot distinguish them"
                        )
                    raise ValueError(
                        f"field '{fname}': choice '{fdef.choices[i]}' is a strict "
                        f"token-prefix of '{fdef.choices[j]}' in token space; the "
                        "engine would never distinguish them"
                    )
            plan[fname] = {
                "shared_ids": shared,
                "remainders": remainders,
            }

        self._plans[key] = plan
        return plan
