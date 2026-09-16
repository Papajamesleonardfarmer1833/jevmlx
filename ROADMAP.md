# ROADMAP — from research artifact to an everyday local decision engine

> Written 2026-09-16. This is the entry point for the next conversations.
> It references two folders: the **research tree** and the **release package**.
> Everything else (benchmarks, eval data, prior findings) hangs off those.

## Where things live

| What | Path | Public | Notes |
|---|---|---|---|
| Research tree | `/Users/richardbaecker/Documents/projects/rlcd-research/` | ❌ local only (**not in git**) | knowledge base 01–12, eval pipeline, raw experimental data |
| Research mirror | `/Users/richardbaecker/Documents/projects/jev-on-a-laptop/` | ✅ [github.com/rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) | docs + eval scripts + our results, no TypeSafe raw data |
| Release package | `/Users/richardbaecker/Documents/projects/parallel-decisions/` | ✅ [github.com/rorshopping/parallel-decisions](https://github.com/rorshopping/parallel-decisions) | the thing other people install — **now with the torch/CUDA backend (`engine_torch.py`, v0.3.0)** |
| Static demo | [hf.co/spaces/rorshopping/parallel-constrained-decisions](https://huggingface.co/spaces/rorshopping/parallel-constrained-decisions) | ✅ live | recorded outputs, no install |
| Interactive demo | `rlcd-research/hf-space/` | ⏸ blocked | Gradio Space needs HF **PRO**; one command when available |
| X posts | `jev-on-a-laptop/x-posts/` | — | thread not yet posted |

**Start a new session with:** "Read `~/Documents/projects/rlcd-research/ROADMAP.md` and continue from Phase N." Context for the project: `jev-on-a-laptop/docs/01-12`.

## Where we are (end of day 1)

**Done**

- Understand TypeSafe/Jev/RLCD (docs 01–03), and the fact that the HF "model" contains no weights (04).
- Reproduced the decoding technique; measured it on the M5 (1.5B/7B/8B, docs 05–08).
- Built a real evaluation on TypeSafe's public cases: 20 cases, 373 question-pairs, 4 workflows (doc 12, `evals/`).
- **Headline result:** free local Qwen2.5-7B reaches **73.8%** agreement with the frontier consensus, vs **Jev 86.6%**, Opus 89.8%, Sol 89.2%, DeepSeek v4.1 Flash 89.5%. Honest gap: ~13 points.
- Packaged the winner as an installable library + CLI + HTTP example (`parallel-decisions`, v0.1.0, 8 tests).
- Measured calibration: **ECE ≈ 0.094**, heavily overconfident, 28/40 wrong answers at ≥0.90 confidence (`CALIBRATION.md`).
- Static HF Space, X thread, quality eval (synthetic, 24 cases).

**Key open problems**

1. Probabilities are softmax, not calibrated → cannot be thresholded safely.
2. ~13-point accuracy gap to Jev; ~16 to the frontier cluster.
3. Memory: KV broadcast = `fields × context`; invoice-style workloads (48 fields × 8.5k tokens) don't fit 16 GB in one pass (needed 6-field chunking).
4. 26 of 110 choice questions have token-level answer collisions (exact scoring exists, but costs an extra pass).
5. Single-run measurements; no variance numbers.
6. Research tree has **no version control** — it exists in one place.

---

## Phase 0 — housekeeping (half a day)

- [ ] **Put the research tree under git** (private repo, e.g. `rorshopping/rlcd-research-private`). At minimum: `git init`, commit, add a private remote. *The public mirror is partial; experiments/, source/ and hf-space/ exist nowhere else.*
- [ ] Tag the package: `cd parallel-decisions && git tag v0.1.0 && git push --tags`.
- [ ] Post the X thread from `jev-on-a-laptop/x-posts/02_thread.txt` (numbers are final).
- [ ] Decide on HF PRO (~$9/mo) → enables the interactive Gradio Space; otherwise delete `hf-space/` or leave blocked. Cheap, recommended once someone else is meant to use it.

**Deliverable:** everything committed, thread posted, decision on HF recorded here.

---

## Phase 1 — Calibration (1–2 weeks) ← highest-value technical item

Goal: make `probability` usable as a decision signal, not just a ranking signal.

- [ ] `tools/fit_temperature.py` in `parallel-decisions`: fits a single temperature `T` on a labelled set by minimising NLL/ECE. Data sources, in order of preference:
  1. the user's own domain samples (start collecting now — 100–200 labelled items per domain is enough),
  2. the research eval data (`rlcd-research/evals/full_eval.json` + recorded answers),
  3. `quality-eval` synthetic cases (weakest, already built).
- [ ] API: `Decider(calibration="temperature.json")` with a tiny `Calibrator` class; raw mode stays default.
- [ ] Report before/after reliability tables in `CALIBRATION.md` (target: **ECE < 0.03** on held-out data, diagonal-ish diagram).
- [ ] **Threshold-routing demo** (`examples/routing.py`): split outputs into *act / review / refuse* using calibrated confidence, and measure the error rate of the "act" bucket. This is the thing that makes the tool trustworthy for automation.
- [ ] If temperature alone is insufficient, add Platt scaling (logistic on the confidence) — the script should be able to compare all three methods and pick by held-out ECE.

**Success criteria:** a printed table where "act" (≥ threshold) has <1% error, with the threshold derived from data rather than guessed. This single feature is what separates "toy decision engine" from "usable in a pipeline".

---

## Phase 2 — Evaluation expansion & model refresh (weeks 2–5)

- [ ] **Domain evals** (this is where it becomes *your* tool): 50–100 labelled cases per intended use:
  - phishing-report triage / campaign-response classification (PhishGuard),
  - receipt/expense classification fields (iBeleg / Quilly),
  - invoice-review decisions (openinvoice / property-management),
  - tickets if used.
  Store under `rlcd-research/evals/domains/<name>/` using the existing pipeline shape.
- [ ] **Multi-run variance**: run the best model 3× on 2 workflows; report mean ± spread. Current numbers are single-run and say so.
- [ ] **Model refresh**: re-run the full TypeSafe eval on newer local models — candidates: Qwen3.5-4B/9B (OptiQ), any newer Qwen, Llama-4-class small, Gemma-4 (relevant to iBeleg's existing model family). Reuse `evals/eval_local_chunked.py`; update the default in the package if a model wins on accuracy *and* fits memory.
- [ ] **Collision audit (was item A2)**: quantify accuracy on the 26 collision fields vs non-collision fields. If collision fields underperform, add a "avoid same-first-token choices" lint to `pd validate` with concrete rename suggestions.
- [ ] Publish updated numbers to `jev-on-a-laptop/evals/` and the HF Space.

**Success criteria:** every claim in the READMEs is backed by ≥3-run or ≥50-case data, and the default model is the best measured fit for 16 GB.

---

## Phase 3 — Package hardening for everyday use (weeks 4–8)

- [ ] **Multi-select fields** (`"type": "multi"`): choose any subset of the choices (e.g. `actions` arrays like the customer-service workflow). Implementation: per-choice binary decision in the existing batched pass — cheap.
- [ ] **True batching**: `decide_many` currently loops. Batch across contexts with a shared schema to amortise prefill.
- [ ] **Concurrency safety**: a lock around the model; document one-call-at-a-time; make `serve.py` queue requests instead of crashing.
- [ ] **Config**: one optional `pd.toml` (`model`, `calibration`, `max_fields_per_batch`, `timeout`) read by both library and CLI.
- [ ] **Observability**: per-call structured log line (fields, latencies, chunk count, calibration on/off) to stderr as JSON when `PD_LOG=json`.
- [ ] **Packaging**: publish to TestPyPI then PyPI as `parallel-decisions`; add a CHANGELOG; pin `mlx`/`mlx-lm` minimums; test on a clean venv.
- [ ] **Platforms**: document Apple Silicon as the supported path; add a clear error on non-arm64, and decide whether the PC (8 GB VRAM) gets a torch backend or stays a cross-check tool (E1 — run the existing `core/engine_torch.py` benchmark there and record results).
  - ✅ **Done 2026-09-16 (E1 + backend shipped early).** Upstream `031d1a8` added a torch engine; we integrated it into `parallel-decisions` v0.3.0 with collision/multi-select/calibration parity (`backend="torch"`, auto-selected off Apple Silicon). Measured on the RTX 2060 SUPER: ~355 ms/decision (8.1× vs CPU), process-parallelism loses to the engine's batching. See `jev-on-a-laptop/docs/14-gpu-torch-backend.md`.
- [ ] Optional: **MCP server** (`examples/mcp_server.py`) exposing `decide(context, schema)` as a tool, so agents (Claude Code, OpenCode) can call local typed decisions. High leverage for "everyday use", ~1 day.

**Success criteria:** `pip install parallel-decisions` works on a fresh Mac; a stranger follows the README to a working call in <10 minutes.

---

## Phase 4 — Real integrations (weeks 6–12) ← where "for other people" actually happens

Pick **one** to production quality rather than five prototypes.

- [ ] **PhishGuard (Python, strongest fit)**: add typed decisions for reported-email triage — e.g. `is_phish`, `category`, `severity`, `recommended_action`, `needs_human` — served by a local `parallel-decisions` instance (import or HTTP). Measure against the existing agent-based flow.
- [ ] **iBeleg / Quilly (Swift, on-device)**: these already run on-device models (Gemma 4 E2B / Apple Foundation Models). Feasibility spike: can the *technique* (constrained parallel decisions) be applied via CoreML/MLX-Swift on a small model for a subset of fields (e.g. `topic`, `deductible`, `needs_review`)? Output: a 1-page go/no-go with memory/latency numbers on device.
- [ ] **Invoice-ish projects** (openinvoice / property-management): invoice-decision fields; the TypeSafe Invoice workflow gives a ready-made 43–48-question template to copy.
- [ ] For any integration: use **calibrated** confidences (Phase 1) and document the routing policy.

**Success criteria:** at least one integration used daily, with a measured error rate and a documented review threshold.

---

## Phase 5 — Research extensions (ongoing, pick as capacity allows)

- [ ] **Domain adaptation by LoRA** (`mlx_lm.lora` on the Mac for ≤7B, or on the PC): train on the domain eval + labelled decisions; measure accuracy and calibration deltas. Likely the cheapest accuracy win.
- [ ] **RLCD-style training attempt** (stretch, 1–2 sessions of work): synthetic decision tasks, reward = calibrated correctness, LoRA scale. Even a failed attempt produces a good write-up; the TypeSafe method is undocumented so this is genuinely novel if it works.
- [ ] **Collision trie**: replace the per-field extra pass with a token-tree pass over all colliding fields at once (known optimisation, doc 05).
- [ ] **KV-cache memory work**: quantised KV or sliding-window prefill for long contexts; unlocks 48-field × long-document workloads in one pass.
- [ ] **Cross-machine validation (E1)**: run the torch engine on the PC, record speed and OOM behaviour in doc 07.
- [ ] **Write-ups**: blog-post-length version of doc 12 ("we reproduced Jev's decoding trick; here is what it costs and where the gap is"); follow-up thread with the calibration numbers once Phase 1 lands.

---

## Explicitly out of scope (for now)

- Free-text or numeric outputs (the technique covers bounded choices; `Score` primitives need a different head).
- Vision/receipt-image inputs (would need a VL model; revisit after Phase 4's feasibility spike).
- Reproducing TypeSafe's actual architecture — closed; we approximate the *shape*, and now beat nothing but the price.
- Multi-user/server-grade deployment (auth, quotas); the HTTP example is for local/LAN use.

## Metrics for the 3-month review

| Metric | Now | Target |
|---|---|---|
| Calibration ECE (held-out) | 0.094 | **< 0.03** |
| Accuracy on TypeSafe eval (local 7B) | 73.8% | **≥ 78%** (calibration + LoRA + model refresh) |
| "Act bucket" error rate in routing demo | n/a | **< 1%** |
| Domains with labelled eval sets | 0 | **≥ 2** |
| Package installs clean on fresh Mac | yes | yes + **PyPI published** |
| Real integrations in daily use | 0 | **≥ 1** |
| Eval runs per claim | single | **≥ 3** |

## Known constraints to keep respecting

- **16 GB M5 Air, fanless.** Long runs throttle; sustained heavy inference is not a use case.
- **Memory = fields × context.** Anything planning >24 fields on >8k tokens must chunk (see `engine.py::_auto_chunk_size`).
- **TypeSafe eval data is fetched, not vendored** (their site). Keep it that way in public repos.
- **No secrets in repos.** HF token lives in `hf auth login` state only.
- **Model weights are never committed.**

## First commands for the next session

```bash
# research tree (evaluation + docs)
cd ~/Documents/projects/rlcd-research
.venv/bin/python evals/score_full.py          # reproduce the 73.8% / 86.6% table
.venv/bin/python evals/calibration.py         # reproduce ECE 0.094

# release package
cd ~/Documents/projects/parallel-decisions
.venv/bin/pytest -q                           # 8 tests, no model needed
.venv/bin/python smoke_test.py                # needs the 7B model
.venv/bin/pd decide --schema examples/fraud.json \
  --context "wire transfer to Cyprus, new device, Tor exit node"

# phase 1 starter
mkdir -p ~/Documents/projects/parallel-decisions/tools
#   then: implement tools/fit_temperature.py against the eval data above
```

