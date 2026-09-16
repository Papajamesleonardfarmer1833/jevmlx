#!/usr/bin/env python3
"""Render the final results table + write the results markdown.

Reads evals/results/full-summary.json (produced by score_full.py) and emits
evals/RESULTS.md with the two tables and a short interpretation.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "results", "full-summary.json")
OUT = os.path.join(HERE, "RESULTS.md")

WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing", "customer_service"]
LABEL = {
    "security_incidents": "Security",
    "agent_trace_observability": "AgentTrace",
    "invoice_processing": "Invoice",
    "customer_service": "CustomerSvc",
}
ORDER = ["sol", "opus", "deepseek-v4.1-flash", "typesafe", "local-qwen2.5-7b", "local-qwen3-8b"]
NICE = {
    "sol": "Sol (published)",
    "opus": "Opus (published)",
    "typesafe": "Jev / TypeSafe (published)",
    "deepseek-v4.1-flash": "DeepSeek v4.1 Flash (max)",
    "local-qwen2.5-7b": "local Qwen2.5-7B (M5 Air)",
    "local-qwen3-8b": "local Qwen3-8B (M5 Air)",
}


def main() -> None:
    data = json.load(open(SUMMARY))
    rows = {r["model"]: r for r in data["rows"]}
    common = data.get("common_subset")
    if common is None:
        common = {}

    lines = []
    lines.append("# Full head-to-head results\n")
    lines.append("Agreement with TypeSafe's reference (consensus of GPT-6 Astra + Fable 5.1) on all")
    lines.append("publicly shipped cases of their four workflows: 20 cases, 373 reference question-pairs.\n")

    lines.append("## All answered pairs\n")
    lines.append("| Model | Overall | " + " | ".join(LABEL[w] for w in WORKFLOWS) + " |")
    lines.append("|---|---|" + "|".join("---" for _ in WORKFLOWS) + "|")
    for name in ORDER:
        r = rows.get(name)
        if not r:
            continue
        ok, tot = r["overall"]
        acc = f"{ok / tot * 100:.1f}% ({ok}/{tot})" if tot else "n/a"
        cells = []
        for w in WORKFLOWS:
            o, n = r["per_workflow"].get(w, [0, 0])
            cells.append(f"{o}/{n}" if n else "–")
        lines.append(f"| {NICE.get(name, name)} | **{acc}** | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Strict common subset (pairs answered by every model)\n")
    sizes = data.get("common_subset_size", {})
    lines.append("Sizes: " + ", ".join(f"{LABEL[w]} {sizes.get(w, 0)}" for w in WORKFLOWS) + "\n")
    lines.append("| Model | Overall | " + " | ".join(LABEL[w] for w in WORKFLOWS) + " |")
    lines.append("|---|---|" + "|".join("---" for _ in WORKFLOWS) + "|")
    for name in ORDER:
        c = common.get(name)
        if not c:
            continue
        ok, tot = c["overall"]
        acc = f"{ok / tot * 100:.1f}% ({ok}/{tot})" if tot else "n/a"
        cells = []
        for w in WORKFLOWS:
            o, n = c["per_workflow"].get(w, [0, 0])
            cells.append(f"{o}/{n}" if n else "–")
        lines.append(f"| {NICE.get(name, name)} | **{acc}** | " + " | ".join(cells) + " |")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
