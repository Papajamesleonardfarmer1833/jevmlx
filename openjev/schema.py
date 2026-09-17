"""Schema definitions and batch-plan compilation for parallel constrained decisions.

Supports booleans and categorical enums with cardinality up to 255.
"""

import os
from typing import Any


class FieldDefinition:
    def __init__(
        self, name: str, field_type: str, description: str, choices: list[str] | None = None
    ):
        self.name = name
        self.field_type = field_type.lower()
        self.description = description

        if self.field_type == "boolean":
            self.choices = ["true", "false"]
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
                f"Unsupported field type '{field_type}'. Supported types: 'boolean' and 'enum'."
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
            lines.append(f'  "{name}": {desc}')
        return "\n".join(lines)

    def compile_batch_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Pre-indexes everything the engine needs for the batched suffix pass.

        Per field: the suffix token ids (the ``  "name": "`` tail, including the
        common prefix shared by all choices) and each choice's full token list
        (the remainder after the common prefix; the bare literal for booleans).
        A choice that adds nothing beyond the common prefix is scored via the
        closing-quote token so its row still ends the JSON string.
        """
        cached = getattr(self, "_batch_plan", None)
        if cached is not None:
            return cached

        plan: dict[str, dict[str, Any]] = {}
        for fname, fdef in self.fields.items():
            if fdef.field_type == "boolean":
                prefix = ""
                suffix_ids = tokenizer.encode(f'  "{fname}": ', add_special_tokens=False)
                choice_token_lists = [
                    tokenizer.encode(v, add_special_tokens=False) for v in ("true", "false")
                ]
            else:
                prefix = os.path.commonprefix(fdef.choices)
                suffix_ids = tokenizer.encode(f'  "{fname}": "{prefix}', add_special_tokens=False)
                choice_token_lists = []
                for choice in fdef.choices:
                    toks = tokenizer.encode(choice[len(prefix) :], add_special_tokens=False)
                    if not toks:
                        toks = tokenizer.encode('"', add_special_tokens=False)
                    choice_token_lists.append(toks)
            plan[fname] = {
                "suffix_ids": suffix_ids,
                "prefix": prefix,
                "choice_token_lists": choice_token_lists,
            }

        self._batch_plan = plan
        return plan
