"""
Parallel constrained decision engine (MLX, Apple Silicon) with broadcast
prefix KV-caching.

- run_naive_generation: autoregressive JSON baseline.
- run_parallel_generation: all schema fields decided in one batched forward pass
  (chunked automatically when the broadcast cache would not fit in memory).
"""

import copy
import functools
import json
import logging
import math
import platform
import re
import time
from typing import Any

from openjev.schema import StructuredSchema

logger = logging.getLogger(__name__)

# Run before any mlx import: on a non-Apple-Silicon machine the mlx import
# itself fails with a low-level error, and the platform message is the useful one.
if platform.system() != "Darwin" or platform.machine() != "arm64":
    raise RuntimeError(
        "openjev requires Apple Silicon (macOS + arm64) with mlx-lm installed. "
        "The PyTorch/CUDA backend was removed."
    )

import mlx.core as mx  # noqa: E402  (must follow the platform check, see above)
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


@functools.lru_cache(maxsize=1)
def load_engine(model_id: str):
    """Load a model + tokenizer once per model id, with Metal shader warmup.

    The cache holds at most one model: models live in Apple Silicon's unified
    memory, which is shared with the OS and the GPU, so keeping several loaded
    at once is the fastest way to OOM. Loading a different model id evicts the
    previous one. Call :func:`clear_engine_cache` to release memory without
    loading anything else.
    """
    logger.info("Loading %s into Apple Silicon unified memory...", model_id)
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    logger.info("Engine loaded in %.2fs.", time.perf_counter() - t0)

    # Warmup: compile prefill and broadcast decode shaders ahead of time.
    logger.info("Warming up Metal shaders on Apple Silicon GPU...")
    w_toks = tokenizer.encode("Warmup context for Apple Silicon GPU")
    w_cache = make_prompt_cache(model)
    w_logits = model(mx.array(w_toks)[None], cache=w_cache)
    mx.eval(w_logits)

    b_cache = _broadcast_cache(w_cache, 28)
    s_dummy = mx.zeros((28, 6), dtype=mx.int32)
    w_suf = model(s_dummy, cache=b_cache)
    mx.eval(w_suf)
    logger.info("Metal shaders compiled & warmed up.")
    return model, tokenizer


def clear_engine_cache() -> None:
    """Drop every cached engine, releasing the model's unified memory.

    Safe to call when nothing is loaded.
    """
    load_engine.cache_clear()


def _broadcast_cache(cache, batch: int):
    """Repeat a prefill KV cache across the batch dimension."""
    b_cache = []
    for c in cache:
        nc = copy.copy(c)
        if hasattr(c, "keys") and c.keys is not None:
            nc.keys = mx.repeat(c.keys, batch, axis=0)
            nc.values = mx.repeat(c.values, batch, axis=0)
        b_cache.append(nc)
    return b_cache


def _chat_ids(tokenizer, user_content: str, assistant_prefix: str = "{\n") -> list:
    """Apply the model's own chat template to a single user message, then append
    the assistant JSON prefix as tokens (specials like BOS are added exactly once,
    by the template)."""
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=True,
    )
    return prompt_ids + tokenizer.encode(assistant_prefix, add_special_tokens=False)


def _stop_token_ids(tokenizer) -> set:
    stop = {tokenizer.eos_token_id}
    for tok_str in ["<end_of_turn>", "<|im_end|>", "<eos>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)
        if tok_id is not None and isinstance(tok_id, int) and tok_id > 0:
            stop.add(tok_id)
    return stop


def _validate_json(current_text: str, schema: StructuredSchema):
    cleaned = current_text.strip()
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)

    parsed_json = None
    is_valid_json = False
    parse_error = None
    try:
        parsed_json = json.loads(cleaned)
        is_valid_json = True
    except Exception as e:
        parse_error = str(e)

    missing_keys = []
    invalid_enums = []
    if is_valid_json and isinstance(parsed_json, dict):
        for fname, fdef in schema.fields.items():
            if fname not in parsed_json:
                missing_keys.append(fname)
            elif fdef.field_type != "boolean":
                val = str(parsed_json[fname])
                if val not in fdef.choices:
                    invalid_enums.append(f"{fname}={val}")

    schema_match = is_valid_json and not missing_keys and not invalid_enums
    return parsed_json, is_valid_json, parse_error, missing_keys, invalid_enums, schema_match


def _model_weight_bytes(model) -> int:
    """Total bytes of all model parameters (quantized weights included)."""
    return sum(int(p.nbytes) for _, p in tree_flatten(model.parameters()))


def _cache_bytes_per_row(cache) -> int:
    """KV-cache bytes a single batch row occupies across all layers."""
    return sum(
        int(c.keys.nbytes) + int(c.values.nbytes)
        for c in cache
        if hasattr(c, "keys") and c.keys is not None
    )


def _max_recommended_working_set() -> int:
    """Metal's max recommended working set size in bytes."""
    return int(mx.metal.device_info()["max_recommended_working_set_size"])


def run_naive_generation(
    model,
    tokenizer,
    context: str,
    schema: StructuredSchema,
    max_tokens: int = 700,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Standard autoregressive generation baseline:
    prompts the LLM to generate the entire JSON object token-by-token.
    """
    prompt_ids = _chat_ids(
        tokenizer,
        f"{schema.to_json_schema_prompt_str()}\n\n"
        "Analyze the following context and generate the required formatted JSON object "
        "(only valid JSON, 2-space indentation, no markdown):\n\n"
        f"{context}",
        assistant_prefix="{\n  ",
    )
    input_ids = mx.array(prompt_ids)[None]

    t0 = time.perf_counter()
    generated_tokens: list[int] = []
    current_text = "{\n  "
    cache = make_prompt_cache(model)

    # Prefill pass
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[:, -1, :]))
    generated_tokens.append(next_token)
    current_text += tokenizer.decode([next_token])

    stop_tokens = _stop_token_ids(tokenizer)
    while len(generated_tokens) < max_tokens and next_token not in stop_tokens:
        logits = model(mx.array([[next_token]]), cache=cache)
        mx.eval(logits)

        next_token = int(mx.argmax(logits[:, -1, :]))
        if next_token in stop_tokens:
            break

        generated_tokens.append(next_token)
        current_text += tokenizer.decode([next_token])

        if current_text.strip().endswith("}") and current_text.count("{") == current_text.count(
            "}"
        ):
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000
    token_count = len(generated_tokens)
    tok_per_sec = (token_count / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

    (parsed_json, is_valid_json, parse_error, missing_keys, invalid_enums, schema_match) = (
        _validate_json(current_text, schema)
    )

    return {
        "mode": "naive_autoregressive",
        "elapsed_ms": round(elapsed_ms, 2),
        "total_tokens": token_count,
        "tokens_per_second": round(tok_per_sec, 1),
        "sequential_forward_passes": token_count,
        "is_valid_json": is_valid_json,
        "schema_match": schema_match,
        "raw_text": current_text,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "missing_keys": missing_keys,
        "invalid_enums": invalid_enums,
        "has_calibrated_probabilities": False,
    }


def _fold_multi(probs_true: dict[str, float], threshold: float = 0.5) -> tuple[list[str], float]:
    """Fold per-option probabilities into a multi field's decision.

    Returns (selected options, confidence): an option is selected when its
    p_true >= threshold. Confidence is the per-option margin of the least
    certain option — min over ALL options of (p_true if the option is
    selected, else 1 - p_true) — so a field is only as confident as its
    weakest accept OR reject.
    """
    selected = [option for option, p_true in probs_true.items() if p_true >= threshold]
    confidence = min(
        (p_true if p_true >= threshold else 1.0 - p_true for p_true in probs_true.values()),
        default=1.0,
    )
    return selected, confidence


def run_parallel_generation(
    model,
    tokenizer,
    context: str,
    schema: StructuredSchema,
    temperature: float = 1.0,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Decide every schema field in one batched forward pass.

    Each choice is scored as the sum of its tokens' log-probs (teacher forced);
    choice probabilities are a softmax of those scores. The prefill KV cache is
    broadcast across rows; colliding choices get one row per choice, everything
    else one row. Batches larger than the memory guard allow run in chunks over
    the same prefill cache.
    """
    t0 = time.perf_counter()

    # 1. Batch plan: per field, suffix ids and per-choice token lists.
    plan = schema.compile_batch_plan(tokenizer)

    rows: list[list[int]] = []  # token ids per row
    row_field: list[str] = []  # field each row belongs to
    row_choice: list[int | None] = []  # choice index for choice-rows, else None
    row_option: dict[int, int] = {}  # row idx -> option index (multi fields only)
    collides: dict[str, bool] = {}
    for fname in schema.fields:
        p = plan[fname]
        if "options" in p:
            # multi: one boolean row per option (same true/false token lists).
            collides[fname] = False
            for oi, suffix_ids in enumerate(p["suffix_ids_list"]):
                # Suffix-only row, like the boolean case: the p_true logit is
                # read at the suffix's last position.
                rows.append(list(suffix_ids))
                row_field.append(fname)
                row_choice.append(None)
                row_option[len(rows) - 1] = oi
            continue
        lists = p["choice_token_lists"]
        # Collision = two choices share the FIRST token (not merely identical lists).
        collides[fname] = len({t[0] for t in lists}) < len(lists)
        if collides[fname]:
            # One teacher-forced row per choice; the suffix-only row would be unused.
            for ci, toks in enumerate(lists):
                rows.append(list(p["suffix_ids"]) + list(toks))
                row_field.append(fname)
                row_choice.append(ci)
        else:
            rows.append(list(p["suffix_ids"]))
            row_field.append(fname)
            row_choice.append(None)

    # 2. Prefill once (compact schema catalog + context).
    schema_str = schema.to_parallel_schema_str()
    base_ids = _chat_ids(
        tokenizer,
        f"Classify JSON attributes:\n{schema_str}\n\n{context}",
        assistant_prefix="{\n",
    )
    base_arr = mx.array(base_ids)[None]

    t_pre0 = time.perf_counter()
    cache = make_prompt_cache(model)
    model(base_arr, cache=cache)
    mx.eval(
        *[t for c in cache if hasattr(c, "keys") and c.keys is not None for t in (c.keys, c.values)]
    )
    t_prefill = (time.perf_counter() - t_pre0) * 1000

    # 3. Memory guard: rows are broadcast copies of the prefill cache.
    bytes_per_row = _cache_bytes_per_row(cache)
    budget = max(1, _max_recommended_working_set() // 2 - _model_weight_bytes(model))
    if max_rows is not None:
        budget = bytes_per_row * max_rows
    auto_max_rows = max(1, budget // bytes_per_row) if bytes_per_row > 0 else len(rows)
    num_passes = max(1, math.ceil(len(rows) / auto_max_rows))
    if num_passes > 1:
        logger.warning(
            "Memory guard: %d rows over %d passes (bytes_per_row=%d)",
            len(rows),
            num_passes,
            bytes_per_row,
        )

    # 4. Batched suffix forward passes (re-broadcast per chunk, no re-prefill).
    #    Rows in a chunk are right-padded to a common length; scoring reads
    #    positions from real lengths, and right-padding cannot affect logits at
    #    earlier (real) positions under causal attention.
    #    Per chunk only the needed per-token floats are extracted; the full
    #    [rows, width, vocab] output is dropped immediately (F3).
    pad_id = tokenizer.pad_token_id or 0
    t_suf0 = time.perf_counter()
    first_token_scores: dict[int, list[float]] = {}  # row idx -> per-choice first-token logit
    choice_total: dict[int, float] = {}  # row idx -> summed choice-token log-prob
    for chunk_start in range(0, len(rows), auto_max_rows):
        chunk = rows[chunk_start : chunk_start + auto_max_rows]
        chunk_len = len(chunk)
        width = max(len(r) for r in chunk)
        padded = mx.array([r + [pad_id] * (width - len(r)) for r in chunk], dtype=mx.int32)
        b_cache = _broadcast_cache(cache, chunk_len)
        mx.eval(
            *[
                t
                for c in b_cache
                if hasattr(c, "keys") and c.keys is not None
                for t in (c.keys, c.values)
            ]
        )
        out = model(padded, cache=b_cache)
        mx.eval(out)
        for i, ridx in enumerate(range(chunk_start, chunk_start + chunk_len)):
            p = plan[row_field[ridx]]
            if ridx in row_option:
                # multi option row: suffix '  "field.option": ', scored at its
                # last position against the true/false first tokens.
                s_len = len(p["suffix_ids_list"][row_option[ridx]])
                lg = out[i, s_len - 1, :]
                first_token_scores[ridx] = [float(lg[t[0]]) for t in p["choice_token_lists"]]
            elif row_choice[ridx] is None:
                s_len = len(p["suffix_ids"])
                lg = out[i, s_len - 1, :]
                first_token_scores[ridx] = [float(lg[t[0]]) for t in p["choice_token_lists"]]
            else:
                s_len = len(p["suffix_ids"])
                toks = p["choice_token_lists"][row_choice[ridx]]
                total = 0.0
                for step, tid in enumerate(toks):
                    row = out[i, s_len + step - 1, :]
                    # log_softmax(row)[tid] = row[tid] - logsumexp(row)
                    total += float(row[tid]) - float(mx.logsumexp(row))
                choice_total[ridx] = total
        del out

    t_suffix_eval = (time.perf_counter() - t_suf0) * 1000

    # 5. One scoring rule for every field: sum of choice-token log-probs,
    #    softmax over choices, confidence = max probability. No clamps.
    parsed_json: dict[str, Any] = {}
    field_telemetry: dict[str, Any] = {}

    field_rows: dict[str, list[int]] = {}
    for idx, fname in enumerate(row_field):
        field_rows.setdefault(fname, []).append(idx)

    for fname, fdef in schema.fields.items():
        p = plan[fname]
        idxs = field_rows[fname]

        if "options" in p:
            # multi: one p_true per option from its boolean row (first-token
            # logit pair true/false, softmaxed at the decision position).
            probs_true = {}
            for oi, ridx in enumerate(idxs):
                scores = first_token_scores[ridx]
                probs = mx.softmax(mx.array(scores) / max(temperature, 1e-4))
                mx.eval(probs)
                probs_true[p["options"][oi]] = float(probs[0])
            selected, confidence = _fold_multi(probs_true)
            parsed_json[fname] = {
                "value": selected,
                "prob": round(confidence, 4),
            }
            field_telemetry[fname] = {
                "value": selected,
                "type": "multi",
                "confidence": round(confidence, 4),
                "cardinality": fdef.cardinality,
                # No 'scores' key for multi: for every other type it holds raw
                # choice scores, which do not exist here. per_option carries
                # the p_true values instead; calibrate skips multi fields.
                "per_option": {o: round(pt, 4) for o, pt in probs_true.items()},
                "top_choices": [
                    {"choice": o, "probability": round(pt, 4)}
                    for o, pt in sorted(probs_true.items(), key=lambda kv: -kv[1])
                ],
            }
            continue

        choice_token_lists = p["choice_token_lists"]
        n_choices = len(choice_token_lists)

        if row_choice[idxs[0]] is None:
            # Distinct first tokens: one row, first choice token scored at the
            # field's last real suffix position (padding excluded).
            scores = first_token_scores[idxs[0]]
        else:
            # Collision: choice ci has its own teacher-forced row at idxs[ci].
            scores = [choice_total[idxs[ci]] for ci in range(n_choices)]

        scores_arr = mx.array(scores) / max(temperature, 1e-4)
        probs = mx.softmax(scores_arr)
        mx.eval(probs)
        probs_list = probs.tolist()
        w_idx = int(mx.argmax(probs))
        w_prob = float(probs_list[w_idx])

        choices_list = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
        val = (
            (choices_list[w_idx].lower() == "true")
            if fdef.field_type == "boolean"
            else choices_list[w_idx]
        )

        parsed_json[fname] = {
            "value": val,
            "prob": round(w_prob, 4),
        }

        scored_choices = [
            {"choice": c, "probability": round(pr, 4)}
            for c, pr in zip(choices_list, probs_list, strict=False)
        ]
        scored_choices.sort(key=lambda x: x["probability"], reverse=True)

        field_telemetry[fname] = {
            "value": val,
            "type": fdef.field_type,
            "confidence": round(w_prob, 4),
            "cardinality": fdef.cardinality,
            "scores": [round(s, 6) for s in scores],
            "top_choices": scored_choices[:5],
        }

    total_elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Decided %d fields in %.1f ms",
        len(schema),
        total_elapsed_ms,
        extra={
            "prefill_ms": round(t_prefill, 2),
            "suffix_eval_ms": round(t_suffix_eval, 2),
            "rows": len(rows),
            "passes": num_passes,
            "num_fields": len(schema),
        },
    )

    return {
        "elapsed_ms": round(total_elapsed_ms, 2),
        "prefill_ms": round(t_prefill, 2),
        "suffix_eval_ms": round(t_suffix_eval, 2),
        "total_tokens_generated": 0,
        "sequential_forward_passes": num_passes,
        "schema_match": True,  # keys/enums guaranteed by construction; bench_model comparison
        "parsed_json": parsed_json,
        "field_telemetry": field_telemetry,
        "num_fields": len(schema),
    }
