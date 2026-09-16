# 05 — How the "parallel constrained decoding" engine works (deep dive)

> Based on a line-by-line read of `core/engine_mlx.py`, `core/schema.py`, `core/prompt_builder.py` (cloned 2026-09-16).

## The core trick in one paragraph

An autoregressive LLM left to its own devices must *generate* a JSON string token by token: `{`, `"`, key, `"`, `:`, `"`, value, ... — hundreds of sequential forward passes, any of which can go off the rails. This engine **never generates the JSON string**. Instead it: (1) defines each schema field as a *slot* with a tiny set of allowed answers, (2) prefills the context **once** into a KV cache, (3) **broadcasts that cache across one batch dimension per field**, (4) runs **one batched forward pass** where each row of the batch is the literal text `  "field_name": ` for a different field, and (5) at each row's final token position, looks only at the logits of the tokens that could start a valid answer, softmaxes over those, picks the argmax, and assembles the JSON programmatically. Structure is guaranteed because the structure was never generated — only *selected*.

## Step by step (with the actual code paths)

Let `M` = number of fields, `C_f` = choices for field `f`.

**1. Compile the schema once** — `schema.compile_parallel_metadata(tokenizer)`
- For a boolean field: suffix = `  "name": `, candidates = first token ID of `true` / `false`.
- For an enum field: compute `os.path.commonprefix(choices)`, strip it, then suffix = `  "name": "<common_prefix>` and candidates = first token ID of each remaining choice suffix.
- Everything is cached on the schema object (`self._parallel_metadata`) so repeat calls are free.
- **Constraint:** `enum` and `boolean` only; ≤ 255 choices per field; if two choices share the same first token, the field is flagged `has_collisions`.

**2. Build the prompt** — hardcoded ChatML:
```
<|im_start|>system
Classify JSON attributes:
  "field": description
  ...
<|im_end|>
<|im_start|>user
{context}<|im_end|>
<|im_start|>assistant
{
```
Note: the schema catalog sent to the model contains only **names + one-line descriptions** — not the choice lists. The choices are enforced purely at the logit level.

**3. Prefill** — one forward pass over the whole prompt; KV cache kept.

**4. Broadcast** — for every cache layer: `mx.repeat(keys, M, axis=0)` (+ same for values). Memory now = **M × KV cache of the context**. This is the key scaling cost of the whole approach.

**5. One batched forward** — a `(M, max_suffix_len)` int array of suffixes goes through the model in one call. Output shape `(M, max_suffix_len, vocab)`.

**6. Pick per field** — for row `i`, take the logits at the field's decision position (last non-pad token of its suffix), slice to that field's candidate token IDs, softmax with temperature (default **1.0** in this path), argmax → choice + probability + full candidate distribution. No collisions? That's it.

**7. Collision path (the ugly part)** — if two choices start with the same token, the engine switches to a **sequential** continuation loop per affected field: up to 4 extra single-field forward passes, multiplying token probabilities, then string-matching the generated fragment back to a choice. If nothing matches, it falls back to `choices[0]`, and the reported confidence is **synthesized**: `clamp(Π token probs, 0.75, 0.9999)` with a uniform distribution spread across the other choices. This is *not* a calibrated probability — it's a repair heuristic. Fields whose choices share multi-token prefixes (common in e.g. `P1_HIGH`/`P1_CRITICAL`-style taxonomies) are exactly the ones where the headline latency and the "calibrated" claim degrade.

**8. Assemble** — build the dict and `json` serialization programmatically:
```json
{ "field": { "value": "…", "prob": 0.9924 }, … }
```
plus `field_telemetry` with top-5 choices each.

## Why it's fast (and when it isn't)

- **Field count barely matters**: cost is one prefill + one batched pass whose batch size = `M`. Compare to naive generation where cost is ~linear in output tokens (≈10–30 tokens per field).
- **Field count still matters a little**: the broadcast KV repeat and the batched attention scale linearly in `M`, and collision handling adds up to 4 sequential passes per colliding field.
- Measured on this machine (see `06`): 1.5B model, 28-field fintech preset → **414 ms parallel vs 3261 ms naive (7.9x)**. Support-triage preset (same 28 fields, more prefix collisions) → **756 ms vs 3312 ms (4.4x)** — exactly the collision cost showing up in wall clock.

## Honest scorecard vs. the TypeSafe claims

| Property | Verdict |
|---|---|
| Typed outputs, no string generation | ✅ True by construction |
| 100% schema validity (keys + enum membership) | ✅ True — with the caveat that a *wrong* value can be guaranteed-valid |
| One-shot parallel evaluation of all fields | ✅ True for the common (no-collision) case; collisions serialize |
| "Calibrated" confidence | ❌ Not really: raw softmax at T=1 over candidate logits. No calibration training, no guarantee that 0.9 means 90%. Collision path even fabricates probabilities |
| Latency | ✅ Real, large win vs. naive JSON generation; matches the artifact's claims (4–8x), far from TypeSafe's 40–200x (which is vs. frontier APIs, not vs. a local naive baseline) |
| Speedup vs. *the author's own naive baseline* | ✅ Measured 4.4x–7.9x; README claims 5.6–7.0x — consistent |
| Nothing to train, works with any mlx-lm model | ✅ Just change `MODEL_ID` |

## Failure modes / sharp edges to remember

1. **"Calibrated" is marketing here.** Treat `prob` as a relative confidence signal, not a probability you can threshold on without evaluating calibration first (`09`).
2. **Collisions → latency + fake confidence.** Detect them (the engine exposes the flag internally; we can surface it) and either avoid choice sets that share first tokens or fix the engine.
3. **Only enums + booleans.** No numbers, strings, lists, free text. (TypeSafe's Score/Noul primitives have no equivalent here.)
4. **`to_json_schema_prompt_str()` truncates choice lists to 20** when a field has >50 choices — so the *naive baseline* is handicapped on the 255-choice preset by design (only relevant to benchmark fairness; the parallel path is unaffected).
5. **Hardcoded ChatML template** — fine for Qwen2.5/Qwen3, wrong for Llama/Gemma-style models without changing the prompt.
6. **Greedy naive baseline** (`mx.argmax`, no sampling) — a weak baseline; a sampling-based or grammar-constrained baseline would be stronger competition.
7. **Boolean case convention**: telemetry reports `"true"/"false"` but `parsed_json` carries a **Python bool** (`true`/`false` in JSON) — minor inconsistency.

## What a faithful-but-honest version would need

- Real **calibration evaluation/learning** (temperature scaling at minimum, RLCD-style training ideally).
- **Batched collision resolution** (token tree / trie over choice sequences) instead of per-field sequential loops.
- More **field types** (numeric score, free string with regex/trie constraints).
- A **trained score head** instead of raw next-token logits if we want to match TypeSafe's "Score/Noul" semantics.
