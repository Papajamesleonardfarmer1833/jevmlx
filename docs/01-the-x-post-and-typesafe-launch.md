# 01 — The X Post & the TypeSafe AI Launch

> Research date: 2026-09-16 · Source: https://x.com/completeskeptic/status/2099925682726002904 (fetched 2026-09-16)

## The post

**Author:** Diogo Almeida — [@CompleteSkeptic](https://x.com/CompleteSkeptic)
**Posted:** Sep 15, 2026, 6:17 PM · **Reach at fetch time:** ~14.2M views, tens of thousands of likes/reposts
**Format:** announcement + 6:17 video demo (side-by-side Jev vs GPT-5.6 Terra)

Post text (verbatim):

> After co-inventing ChatGPT, I kept asking myself: why have superhuman chat models not led to AGI? I've spent the last 2 years in stealth building a new way to train models (RLCD), and a new type of frontier AI model that we are releasing today: Jev
> • 20-200x faster
> • 40-400x cheaper (w/ output tokens free)
> • Frontier composable intelligence optimized for decisions
> AFAICT the shortest path to AI-based economic revolution

## Who / what is behind it

| | |
|---|---|
| Company | **TypeSafe AI** (typesafe.ai), San Francisco, founded 2024, emerged from stealth Sep 15, 2026 |
| Funding | **$40M seed led by DCVC** (BusinessWire, Sep 15, 2026) |
| Founder & CEO | **Diogo Almeida** — ex-OpenAI researcher; public bios describe him as a co-inventor of **RLHF** and **ChatGPT** |
| Co-founders | Erik Gafni, Sasha Sheng |
| Product | **Jev** — first public "System One Model", early access / waitlist |
| Training method | **RLCD = Reinforcement Learning for Calibrated Decisions** (their definition — see `03-rlcd-explained-and-name-collision.md`) |
| New stack claim | "a new model architecture, a parallel sampler for maximum efficiency, and training method RLCD" |

## The core claims (as published)

From the launch blog ["Introducing System One Models & Jev"](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [typesafe.ai](https://typesafe.ai/):

- **Strings → typed decisions.** Jev gives up string generation entirely; every output is a typed value (choice / score / boolean-like "noul") with a **probability distribution and confidence** attached, defined up front by a schema. "Type errors are mathematically impossible."
- **"Can't hallucinate."** The claim is about *type safety*, not correctness: it cannot invent keys or malformed output, and can never emit fabricated prose. It can still choose a wrong label (see caveats below).
- **Parallel sampler.** All questions are evaluated in **one query / one pass**, not token-by-token. Adding questions "barely changes the response time". This is where the 20–200x speedup vs. autoregressive LLMs comes from.
- **Latency 70–500 ms** end-to-end (vs. 3–329 s for frontier LLMs; their home page demo shows Jev 0.114 s vs GPT-5.6 Terra 8.566 s).
- **Price: $0.042 / MTok input (= $42 / billion tokens), output tokens FREE** ("too cheap to meter"). Stated as 238x cheaper input than Claude Fable 5.1.
- **Calibrated confidence**: "higher confidence means higher accuracy", "returns similar answers for similar inputs" — i.e. calibration is (claimed to be) *trained for*, not just softmax.

## Third-party coverage (useful reality check)

- **The Register** (2026-09-16): ["TypeSafe AI debuts model for machines that plays Doom"](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711)
  - Confirms $40M, the RLHF/ChatGPT pedigree, the primitives ([Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score), [Noul](https://docs.typesafe.ai/primitives/noul)), the pricing, and the Doom demo.
  - Notable skeptical framing: "**hallucination-free** … really isn't a fair comparison as its output is not natural language" — Jev can still be *incorrect*, it just cannot produce untyped/malformed output.
  - Notes the Jevons-paradox naming rationale.
- **BusinessWire** (2026-09-15): official launch press release — "less than 100 ms latency … process hundreds of outputs in parallel from a single prompt".
- **NewsBytes / LavX / GoKawiil** (2026-09-15/16): secondary coverage; consistent summary of the claims.

## The demos

1. **Side-by-side decision demo** (the 6:17 video) — same state/context into Jev and an LLM; Jev returns all field decisions with probabilities nearly instantly, the LLM streams JSON for seconds and can disagree/omit.
2. **Doom** — Jev receives structured game state (text/data structure, *not* images) and picks an action ~10x/second (~$7/hour of queries).
3. **Wikiracing** — choosing the next Wikipedia link among hundreds–thousands of candidates; used to show compounding benefit of zero type errors + high cardinality (255 max per choice).

## Fine print / caveats they themselves published

- Their speed/cost receipts come from **their own workflow evals** (https://evals.typesafe.ai/): "we expect these [193.6x faster, 444.6x cheaper] are on the higher end of real-world gains".
- Workflows were built by their own capabilities team → "some bias could exist".
- Reference answers for the evals = average of **GPT-6 Astra** and **Fable 5.1** → biases toward OpenAI/Anthropic.
- The side-by-side demo's input was short/dense "to emphasize the difference in sampling methodology" — paints their model in a favorable light.
- Pricing sustainability: "We can't prove it isn't subsidized."

## Why this matters for us (the practical takeaway)

Two separable things are being announced:

1. **A training method** (RLCD): train models to output *calibrated probabilities on structured decisions* instead of human-pleasing text. Not reproducible from the public artifact — Jev is a closed early-access API, no weights, no paper yet.
2. **An inference/sampling trick** (parallel evaluation of many constrained questions against one prefilled state): this **is** reproducible on a laptop, and it is exactly what the community artifact in `04-...` implements — using a stock Qwen model with **zero training**.

That distinction is the single most important thing to keep straight in this whole project:

> The Hugging Face repo `harshatheg/Qwen-2.5-1B-RLCD` reproduces the **shape of the output** (typed, parallel, calibrated-ish decisions) — **not** the RLCD training, and not Jev itself.

## Sources

- X post: https://x.com/completeskeptic/status/2099925682726002904 (fetch: 2026-09-16)
- Launch blog: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Home page: https://typesafe.ai/ · Manifesto: https://typesafe.ai/manifesto · Docs: https://docs.typesafe.ai/
- Evals: https://evals.typesafe.ai/ · Adapter: https://github.com/typesafe-ai/system-one-adapter-python
- The Register: https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711
- Press release: https://www.businesswire.com/news/home/20260915525333/en/
