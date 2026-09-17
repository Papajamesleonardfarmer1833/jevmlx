"""Schema definitions and batch-plan compilation for parallel constrained decisions.

Supports booleans, categorical enums (cardinality up to 255), and multi fields
(subset of choices, 2-64 options, decided as one boolean decision per option).
"""

import json
import weakref
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
        # Compiled plans, keyed by tokenizer object identity (weakref so the
        # cache never keeps a tokenizer alive; dict-by-id fallback for objects
        # that cannot be weakly referenced). One schema object can be reused
        # with several models, and token IDs are tokenizer-specific.
        self._plans: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
        self._plans_by_id: dict[int, dict[str, Any]] = {}

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

    def compile_batch_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Pre-index everything the engine needs for the batched suffix pass.

        Token-aligned at both boundaries: every choice is encoded as ONE
        tokenization of its complete assistant-tail candidate — ``{\n`` + the
        JSON row text + ``,\n`` (the row text being ``  "name": "choice"``
        or the bare literal, built with json.dumps so quotes and backslashes
        survive). Tokenizing a prefix and its remainder separately would be
        wrong: BPE merges are not compositional, so the concatenated ids would
        not be the tokenization of the full candidate. The prefill prompt ends
        exactly at the chat template's generation marker; ``{\n`` belongs to
        the candidate tail, not the prompt.

        Per enum/boolean field the plan carries ``shared_ids`` (the common
        token-ID prefix across the candidates of ALL fields — typically the
        ``{\n  "`` lead-in) and per-choice ``remainders``.
        Per multi field it carries one ``suffix_ids_list`` entry per option —
        that option's shared token prefix (everything before its true/false
        divergence, i.e. the row the engine runs) — and ``remainders`` with
        the option's two true/false continuations. Plans are cached per
        tokenizer identity (name_or_path + vocab size).
        """
        try:
            cached = self._plans.get(tokenizer)  # weakref: keyed by object identity
        except TypeError:
            # Tokenizer is not weak-referenceable: fall back to id(), relying
            # on the caller keeping the tokenizer alive for the session.
            cached = self._plans_by_id.get(id(tokenizer))
        if cached is not None:
            return cached

        plan: dict[str, dict[str, Any]] = {}

        def candidate_text(name: str, value_text: str) -> str:
            """The complete assistant tail for one row: '{\n' + row + ',\n'."""
            return "{\n" + f"  {json.dumps(name)}: {value_text}" + ",\n"

        for fname, fdef in self.fields.items():
            if fdef.field_type == "multi":
                # One boolean row per option, scored where true/false diverge.
                # The plan is built PER OPTION from that option's own candidate
                # pair: shared = everything up to the true/false divergence
                # (the option's full row lead-in plus any common token start of
                # 'true'/'false'), remainders = the two continuations. A single
                # cross-option prefix would put branch nodes at the option-name
                # position and read the true/false logits at the wrong spot.
                suffix_ids_list = []
                remainders_per_option = []
                for option in fdef.choices:
                    pair = [
                        tokenizer.encode(
                            candidate_text(f"{fname}.{option}", literal),
                            add_special_tokens=False,
                        )
                        for literal in ("true", "false")
                    ]
                    option_shared = _common_token_prefix(pair)
                    option_remainders = [full[len(option_shared) :] for full in pair]
                    if option_remainders[0] == option_remainders[1]:
                        raise ValueError(
                            f"field '{fname}': option '{option}' tokenizes to "
                            "identical true/false candidates; the engine cannot "
                            "distinguish them"
                        )
                    suffix_ids_list.append(option_shared)
                    remainders_per_option.append(option_remainders)
                plan[fname] = {
                    "options": list(fdef.choices),
                    "suffix_ids_list": suffix_ids_list,
                    "remainders": remainders_per_option,
                }
                continue

            if fdef.field_type == "boolean":
                value_texts = ["true", "false"]
            else:
                value_texts = [json.dumps(choice) for choice in fdef.choices]
            candidates = [
                tokenizer.encode(candidate_text(fname, value_text), add_special_tokens=False)
                for value_text in value_texts
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

        # Schema-wide lead-in (typically '{\n  "') shared by every field's
        # candidates: lifted out of shared_ids so the engine can keep it in
        # the prefill broadcast cache. Remainders stay relative to the full
        # per-field shared prefix; rows are lead_in + shared_ids + path.
        field_shared_prefixes = [p["shared_ids"] for p in plan.values() if "shared_ids" in p]
        if not field_shared_prefixes:
            # Multi-only schema: option rows are self-contained, nothing to lift.
            plan["_lead_in_ids"] = []
            try:
                self._plans[tokenizer] = plan
            except TypeError:
                self._plans_by_id[id(tokenizer)] = plan
            return plan
        lead_in = _common_token_prefix(field_shared_prefixes)
        # An empty schema-wide lead-in is legal (e.g. char-level tokenizers
        # where '{\n' fuses with the field name): the engine then runs one row
        # per field with no broadcast prefix — each row still carries that
        # field's full shared_ids.
        for p in plan.values():
            if "shared_ids" in p:
                p["shared_ids"] = p["shared_ids"][len(lead_in) :]
        plan["_lead_in_ids"] = list(lead_in)

        try:
            self._plans[tokenizer] = plan
        except TypeError:
            self._plans_by_id[id(tokenizer)] = plan
        return plan
