# TypeSafe-derived benchmark fetcher

Downloads TypeSafe's published evaluation examples from
https://evals.typesafe.ai/ and converts them to the jevmlx eval JSONL
contract (see `benchmarks/README.md` for the harness side).

```bash
python -m benchmarks.typesafe.fetch --out cases.jsonl [--workflow NAME] [--refresh]
```

## What this is — and is not

This is a **flattened, TypeSafe-derived benchmark**, not a reproduction of
TypeSafe's evaluation workflow. TypeSafe publishes interactive, multi-step
cases; this converter extracts each published workflow step (reference node)
with its exact read-set of documents and turns its questions into flat eval
fields. Two consequences:

- **Consensus pseudo-labels.** Labels are the published reviewer consensus
  (averaged probability distributions, or majority votes), *not* independent
  gold labels. The full consensus distribution, its top1−top2 margin, and an
  ambiguity flag (margin < 0.1 or an exact tie) are stored in `meta` so
  downstream analysis can weight or exclude uncertain fields. Exact ties are
  marked ambiguous and still broken deterministically by the field's choice
  order — never by dict insertion order.
- **`benchmark_only: true` on every record.** These examples are deliberately
  selected to cover model disagreements, shared misses, and agreements —
  they are not an IID deployment sample. Never use them for calibration or
  routing thresholds; use representative domain data for that.

Nothing from TypeSafe is committed to this repository. Raw downloads are
cached content-addressed under `~/.cache/jevmlx/typesafe/` (reruns are
offline; `--refresh` re-downloads), and `dataset.lock.json` next to the
output records source URLs, content sha256s, fetch timestamps, the parser
version, and the hash of the written cases file.

## Conversion

- One record per case, or per distinct read-set when the case's workflow
  steps read different documents (`/n<k>` id suffix, `group_id` = case id).
- Field identity is `(node_id, qid, occurrence)`. A qid answered at more than
  one workflow step is named `<node_id>__<qid>__<occ>` (occurrence = 1-based
  index of the step among the steps answering that qid); unique qids keep
  their plain name.
- `noul` questions → `boolean` fields; `choice`/`score` questions → `enum`
  fields with the published options as choices (score uses the fixed 0–3
  rubric scale); free-text questions are skipped and counted.
- `split` is the documented deterministic rule: holdout iff
  `int(sha1(id)[:8], 16) % 5 == 0` (~20%). It is a pure function of the id.
  Same-family cases may straddle the split — group by `group_id` when
  evaluating.

## How to read the numbers vs evals.typesafe.ai

When a run over this dataset is reported, two metrics in `report.json`
(`metrics.agreement`, `metrics.tvd_vs_consensus`) make it comparable with
the numbers published on evals.typesafe.ai — with the caveats above firmly
in mind. **Agreement** (`agreement.overall`, per workflow in
`agreement.by_workflow`) is the share of labelled fields where jevmlx's
prediction equals the fetcher's consensus label; it is TypeSafe's headline
metric computed against our reconstructed pseudo-labels, so it measures
jevmlx against the published reviewer consensus, not against independent
ground truth. `agreement.agreement_common_subset` is the same rate computed
only on fields the fetcher did not flag ambiguous (consensus top1−top2
margin < 0.1 or an exact tie) — the closest analogue to TypeSafe's own
presentation, which does not include our low-margin cases. **TVD vs
consensus** (`tvd_vs_consensus`) is the mean total-variation distance
between jevmlx's per-field choice distribution (softmax of the constrained
log-scores at T=1) and the published reviewer distribution: lower is better,
0.0 means the model's ranking of the options exactly matches the
reviewers', and it is defined even where the argmax disagrees. Run with
`jevmlx eval --carry-consensus` (or the flag on `run_eval`) so the
consensus distributions reach the prediction lines; the report's
"Agreement vs TypeSafe consensus" table shows both metrics per workflow.
These numbers are comparable across *models and scorers on this dataset*;
treat any comparison to evals.typesafe.ai itself as indicative, since the
case selection, field reconstruction, and consensus computation are ours.
