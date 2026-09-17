# Docs index

These 14 documents are the upstream research history behind openjev — written during the M5 MacBook Air study of TypeSafe AI's Jev (Sep 2026). **Numbers in them are from that study** (that hardware, those models, that date); they are not claims about current builds.

| Doc | What it covers |
|---|---|
| [01 — The X Post & the TypeSafe AI Launch](01-the-x-post-and-typesafe-launch.md) | The launch post that started this project. |
| [02 — TypeSafe AI / Jev: product, docs, manifesto](02-typesafe-jev-product-details.md) | Detailed notes on the Jev product and claims. |
| [03 — "RLCD": what it means here, and the name collision](03-rlcd-explained-and-name-collision.md) | Terminology untangling. |
| [04 — The HF artifact: what `harshatheg/Qwen-2.5-1B-RLCD` actually is](04-the-hf-artifact-what-it-actually-is.md) | Provenance of the upstream engine artifact. |
| [05 — How the "parallel constrained decoding" engine works](05-how-parallel-constrained-decoding-works.md) | Deep dive: broadcast KV cache, per-field decisions. |
| [06 — Running it locally on this Mac (verified)](06-running-locally-on-this-mac.md) | Local setup and first runs. |
| [07 — Hardware: Mac vs PC (8 GB VRAM / 32 GB RAM)](07-hardware-mac-vs-pc-analysis.md) | Where the engine runs and why. |
| [08 — Model upgrade path: 1.5B → 7B → 8B](08-model-upgrade-path.md) | What bigger models buy on 16 GB unified memory. |
| [09 — Open questions & the experiment plan](09-open-questions-and-next-experiments.md) | Open threads at the time of writing. |
| [10 — Jev's published accuracy vs. our local 8B](10-jev-published-accuracy-vs-our-8b.md) | The apples-to-oranges comparison problem. |
| [11 — Head-to-head on TypeSafe's own questions (5 cases)](11-typesafe-eval-head-to-head.md) | First small eval — later shown to be misleading. |
| [12 — Full head-to-head on TypeSafe's public eval suite](12-full-head-to-head.md) | The 20-case / 373-pair result (the honest one). |
| [13 — Jev-style decisions across the app portfolio](13-app-integration-opportunities.md) | Integration ideas for real applications. |
| [14 — The GPU update: torch backend and what CUDA buys](14-gpu-torch-backend.md) | Historical CUDA backend analysis (backend since removed). |
