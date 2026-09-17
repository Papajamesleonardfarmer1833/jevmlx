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
from openjev.trie import build_trie, score_trie, softmax

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


def _chat_ids(tokenizer, user_content: str) -> list:
    """Apply the model's own chat template to a single user message (specials
    like BOS are added exactly once, by the template). The prompt ends exactly
    at the generation marker; the assistant JSON tail belongs to the candidate
    tokenization, not the prompt."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=True,
    )


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


def _rows_per_chunk(budget_bytes: int, bytes_per_row: int, max_rows: int | None) -> int:
    """Rows per suffix chunk: budget-limited cap, optionally tightened by max_rows.

    max_rows is a caller cap on the automatic heuristic, never an override of it.
    """
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows!r}")
    auto_cap = max(1, budget_bytes // bytes_per_row) if bytes_per_row > 0 else (max_rows or 1)
    if max_rows is not None:
        return min(auto_cap, max_rows)
    return auto_cap


def run_parallel_generation(
    model,
    tokenizer,
    context: str,
    schema: StructuredSchema,
    temperature: float = 1.0,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Decide every schema field in one batched forward pass.

    Scoring: per field, a token trie over the choice continuations. Rows are
    the trie's branch points (one row per node where choices diverge); each
    node's children are softmaxed over their logits at the node's decision
    position and every choice accumulates the log-probability of its branch.
    Fields whose choices never share a first token get exactly one row, same
    as before. Choice probabilities sum to 1, so confidence = P(choice).

    The prefill KV cache is broadcast across rows; batches larger than the
    chunking heuristic allows run in chunks over the same prefill cache.

    ``temperature`` is a post-hoc temperature applied to the per-branch
    logits (softmax(logits / temperature)) — it is not a token-level sampling
    temperature; generation itself is deterministic.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows!r}")

    t0 = time.perf_counter()

    # 1. Batch plan, then trie rows per field: one row per branch point of the
    #    choice remainders (fields with distinct first tokens: exactly one row).
    plan = schema.compile_batch_plan(tokenizer)

    rows: list[list[int]] = []  # token ids per row
    row_field: list[str] = []  # field each row belongs to
    row_branch: dict[int, int] = {}  # row idx -> branch-node index within its field
    row_option: dict[int, int] = {}  # row idx -> option index (multi fields only)
    tries: dict[str, list[dict]] = {}
    lead_in = plan["_lead_in_ids"]
    for fname in schema.fields:
        p = plan[fname]
        if "options" in p:
            # multi: one boolean row per option (its per-option shared prefix,
            # which ends right before the option's true/false divergence).
            for oi, suffix_ids in enumerate(p["suffix_ids_list"]):
                rows.append(lead_in + list(suffix_ids))
                row_field.append(fname)
                row_option[len(rows) - 1] = oi
            continue
        field_trie = build_trie(p["remainders"])
        tries[fname] = field_trie
        for bi, node in enumerate(field_trie):
            rows.append(lead_in + list(p["shared_ids"]) + list(node["path"]))
            row_field.append(fname)
            row_branch[len(rows) - 1] = bi

    # 2. Prefill once (compact schema catalog + context). The prompt ends at
    #    the chat template's generation marker; '{\n' and everything after is
    #    part of the candidate rows (T3 boundary alignment).
    schema_str = schema.to_parallel_schema_str()
    base_ids = _chat_ids(tokenizer, f"Classify JSON attributes:\n{schema_str}\n\n{context}")
    base_arr = mx.array(base_ids)[None]

    t_pre0 = time.perf_counter()
    cache = make_prompt_cache(model)
    model(base_arr, cache=cache)
    mx.eval(
        *[t for c in cache if hasattr(c, "keys") and c.keys is not None for t in (c.keys, c.values)]
    )
    t_prefill = (time.perf_counter() - t_pre0) * 1000

    # 3. Memory guard: rows are broadcast copies of the prefill cache. The
    #    estimate includes the [rows, width, vocab] output logits for one chunk
    #    (float32 logits are the dominant activation). This is a chunking
    #    heuristic, not a hard bound on peak Metal memory.
    bytes_per_row = _cache_bytes_per_row(cache)
    width_max = max(len(r) for r in rows) if rows else 0
    vocab_size = (
        model.args.vocab_size
        if hasattr(model, "args") and hasattr(model.args, "vocab_size")
        else model.model.embed_tokens.weight.shape[0]
    )  # simplest correct static source; falls back to the embedding row count (= vocab)
    bytes_per_row += width_max * vocab_size * 4
    weight_bytes = _model_weight_bytes(model)
    budget = max(1, _max_recommended_working_set() // 2 - weight_bytes)
    if max_rows is not None:
        budget = bytes_per_row * max_rows
    auto_max_rows = _rows_per_chunk(budget, bytes_per_row, max_rows)
    num_passes = max(1, math.ceil(len(rows) / auto_max_rows))
    if num_passes > 1:
        logger.warning(
            "Chunking heuristic: %d rows over %d passes (bytes_per_row=%d)",
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
    # Row idx -> {branch-node index: [child logits in node["children"] order]}.
    node_logits: dict[int, dict[int, list[float]]] = {}
    # Multi option rows: [p_true logit, p_false logit] at the suffix end.
    option_pair: dict[int, list[float]] = {}
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
                # multi option row: true/false logits at the option row's last
                # position (the row ends right before the true/false divergence).
                lg = out[i, len(lead_in) + len(p["suffix_ids_list"][row_option[ridx]]) - 1, :]
                option_pair[ridx] = [float(lg[t[0]]) for t in p["remainders"][row_option[ridx]]]
            else:
                # Branch-node row: child logits at the node's last position,
                # in node["children"] order.
                node = tries[row_field[ridx]][row_branch[ridx]]
                position = len(lead_in) + len(p["shared_ids"]) + len(node["path"]) - 1
                lg = out[i, position, :]
                node_logits[ridx] = {row_branch[ridx]: [float(lg[tok]) for tok in node["children"]]}
        del out

    t_suffix_eval = (time.perf_counter() - t_suf0) * 1000

    # 5. Trie scoring: P(choice) = product of branch factors along its path;
    #    proper distribution, so confidence = P(choice). Full precision: no
    #    rounding anywhere in the engine's results (presentation rounds in cli).
    parsed_json: dict[str, Any] = {}
    field_telemetry: dict[str, Any] = {}

    field_rows: dict[str, list[int]] = {}
    for idx, fname in enumerate(row_field):
        field_rows.setdefault(fname, []).append(idx)

    for fname, fdef in schema.fields.items():
        p = plan[fname]
        idxs = field_rows[fname]

        if "options" in p:
            # multi: one-vs-rest classification — each option is an independent
            # binary decision ("does this option apply?"), scored at the
            # option's own true/false divergence. per_option holds independent
            # binary probabilities (NOT a subset distribution); 'confidence'
            # is the weakest binary decision (see _fold_multi), not the
            # probability of the selected subset. calibrate skips multi fields.
            probs_true = {}
            for oi, ridx in enumerate(idxs):
                pair = option_pair[ridx]
                (p_true, p_false) = softmax(pair, temperature=temperature)
                probs_true[p["options"][oi]] = p_true
            selected, confidence = _fold_multi(probs_true)
            parsed_json[fname] = {
                "value": selected,
                "prob": confidence,
            }
            field_telemetry[fname] = {
                "value": selected,
                "type": "multi",
                "confidence": confidence,
                "cardinality": fdef.cardinality,
                # No 'scores' key for multi: for every other type it holds log
                # P(choice), which does not exist here. per_option carries the
                # p_true values instead; calibrate skips multi fields.
                "per_option": dict(probs_true),
                "top_choices": [
                    {"choice": o, "probability": pt}
                    for o, pt in sorted(probs_true.items(), key=lambda kv: -kv[1])
                ],
                "rows": len(idxs),
            }
            continue

        field_trie = tries[fname]
        n_choices = fdef.cardinality

        # logits_at_node for score_trie: branch-node rows carry their child
        # logits under the branch-node index (row_branch of that row). Bound
        # per field so the score_trie callback cannot see a later iteration's
        # dictionaries.
        logits_by_branch: dict[int, list[float]] = {}
        for ridx in idxs:
            logits_by_branch.update(node_logits[ridx])
        branch_index = {id(node): bi for bi, node in enumerate(field_trie)}

        def logits_at_node(
            node: dict, _lookup=logits_by_branch, _index=branch_index
        ) -> list[float]:
            return _lookup[_index[id(node)]]

        scores = score_trie(field_trie, n_choices, logits_at_node)
        # Confidence temperature applied once to the final per-choice scores
        # (softmax(scores / T)): ranking is invariant, calibrate.py fits this T.
        probs_list = softmax(scores, temperature=temperature)
        w_idx = max(range(n_choices), key=probs_list.__getitem__)
        w_prob = probs_list[w_idx]

        choices_list = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
        val = (
            (choices_list[w_idx].lower() == "true")
            if fdef.field_type == "boolean"
            else choices_list[w_idx]
        )

        parsed_json[fname] = {
            "value": val,
            "prob": w_prob,
        }

        scored_choices = [
            {"choice": c, "probability": pr} for c, pr in zip(choices_list, probs_list, strict=True)
        ]
        scored_choices.sort(key=lambda x: x["probability"], reverse=True)

        field_telemetry[fname] = {
            "value": val,
            "type": fdef.field_type,
            "confidence": w_prob,
            "cardinality": fdef.cardinality,
            # Constrained-path log-probabilities at T=1, dict keyed by choice
            # string (the contract calibrate.collect reads). Temperature is
            # applied once downstream, to the final distribution.
            "log_scores": {choice: lp for choice, lp in zip(choices_list, scores, strict=True)},
            "top_choices": scored_choices[:5],
            "rows": len(field_trie),
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
        # The per-choice probabilities are the constrained path probability
        # (product of masked branch softmaxes), not a normalized full-sequence
        # likelihood and not automatically calibrated.
        "confidence_model": "constrained_path",
        "parsed_json": parsed_json,
        "field_telemetry": field_telemetry,
        "num_fields": len(schema),
    }
