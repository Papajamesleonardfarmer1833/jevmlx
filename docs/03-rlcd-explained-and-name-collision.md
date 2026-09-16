# 03 — "RLCD": what it means here, and the name collision

## A. TypeSafe's RLCD (Sep 2026) — *Reinforcement Learning for Calibrated Decisions*

The training method behind Jev/System One models, per the launch blog:

- **Optimizes for calibrated decisions**: "answers with epistemically honest probabilities on System One tasks."
- Contrast with:
  - **RLHF** → optimizes *human preference* (chat responses raters like). Side effects TypeSafe calls out: mode dropping, overconfidence, unreliability, humans-in-the-loop required.
  - **RLVR** → optimizes *programmatically verifiable rewards* (math, code, benchmarks).
- Together with a **parallel sampler** and a "new architecture", it produces models that output typed decisions + confidence instead of strings.
- **Public details: essentially none.** No paper, no code, no weights. The blog's FAQ ("Why was a new training algorithm needed?", "Where does our training data come from?", "Is Jev just a smaller LLM?") is collapsed/elided in the rendered page — we could not extract answers.

**Status for us:** a black box. We cannot reproduce RLCD. What we *can* reproduce is the **inference shape** (parallel constrained decisions) on top of stock models — which is what the community artifact does, with **zero training**.

## B. The older RLCD (2023) — *Reinforcement Learning from Contrastive Distillation*

Completely unrelated, but guaranteed to pollute search results:

- Paper: "RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment" — Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, Yuandong Tian. arXiv **2307.12950** (Jul 2023), **ICLR 2024**.
- Code: https://github.com/facebookresearch/rlcd — alignment of **LLaMA-7B** on harmlessness/helpfulness/story-outlines. Pipeline: simulate preference pairs with positive/negative prompts → train a reward model → **PPO** to align the base LM.
- It is about *making chat models nicer*, and it predates TypeSafe by ~3 years. Zero relation to calibrated decisions.

> If you search "RLCD" you will mostly find the Meta/2023 work. Always qualify which RLCD you mean.

## C. Other RL-adjacent "parallel" research (for orientation)

Not required reading, but they show the neighborhood this idea lives in:

- **Parallel-R1** (ICLR 2026): RL to teach *parallel thinking* in reasoning (different concept — parallel reasoning paths, not parallel decision extraction).
- **Speculative / pipeline decoding** (e.g. "Speculative Pipeline Decoding", 2026): parallelize token generation, verified against the base model; a different speed trick — generate-then-verify rather than broadcast-and-slice.
- **Trie/grammar-constrained decoding** (e.g. xgrammar, llguidance, trie `LogitsProcessor` projects): guarantee *structure* during normal autoregressive generation by masking invalid tokens. This guarantees syntax but keeps you in the sequential regime, and for `response_format`-style schemas can even suppress tool-call emissions (see the "Constraint Tax" project). The TypeSafe-style parallel sampler sidesteps this class of issue entirely by not generating a token stream at all.

## D. About the "he trained it for only 30 steps" recollection

We could not verify any "30 RL steps" claim connected to the HF artifact:

- The X thread is login-walled; nothing in the fetched post or search results mentions 30 steps.
- The HF repo contains **no training code and no weights whatsoever** (verified: `usedStorage: 0`, no `.safetensors` in the file list — see `04-...`). The engine loads a **stock** `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
- Numbers that *are* in the wild and could be confused with "30 steps":
  - Meta's 2023 RLCD repo: "40–80 PPO steps" usually reasonable; checkpoints every 20 steps (512 rollouts each).
  - Parallel-R1: released a **200-step** mid-training checkpoint.
  - TypeSafe's Doom demo: ~10 queries/second → at ~$0.000081/call ≈ **$7/hour**.

**Conclusion:** the artifact we can actually run required **zero training steps**. If the "30 steps" anecdote exists, it's about someone else's experiment (or a misremembering). We note it here so it doesn't quietly become a false "fact" in this project.

## Sources

- TypeSafe launch blog (RLCD definition): https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Meta RLCD: https://github.com/facebookresearch/rlcd · https://arxiv.org/abs/2307.12950
- Parallel-R1: https://zhengkid.github.io/Parallel_R1.github.io/
- Speculative Pipeline Decoding: https://github.com/yuyijiong/speculative_pipeline_decoding
