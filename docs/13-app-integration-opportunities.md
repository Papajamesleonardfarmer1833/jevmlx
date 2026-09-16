# Jev-style decisions across the app portfolio

Research date: 2026-09-16. Scope: opportunities and proposed integration, not implementation. Four parallel research agents inspected iBeleg, Snipledger2, Recall, and a bounded portfolio sample. App code was not modified or built. Paths below are relative to `C:\Users\Richard\Documents\Projects`.

## Recommendation

Use a bounded decision layer **between extraction and application behavior**, not a replacement for OCR, free-text generation, calculations, or scheduling.

Prioritize:

1. **iBeleg:** receipt classification and review suggestions after extraction. **Snipledger2:** grounded field-candidate selection has the highest value; financial-statement page classification is the smallest pilot. Keep amounts, taxes, booking and export checks authoritative in ordinary code.
2. **Recall:** import-time flashcard quality suggestions first; semantic spoken-answer coverage second. The latter has especially strong product value because its keyword grader explicitly does not model negation.
3. **Quilly:** select and label contact candidates already found in OCR. Strong opportunity to share infrastructure with iBeleg.
4. **SnapTriage:** screenshot-content routing after OCR/QR detection. Good product fit, but its zero-network invariant makes an on-device runtime a prerequisite.
5. **HearthLog / incase:** suggest asset/document categories. Useful, but unlikely to justify a multi-gigabyte model alone.

These are hypotheses to evaluate, not measured app improvements. The unidentified “underscore App” remains unresolved; it should not be guessed from folder names.

## What Jev actually offers

There are three distinct deployment options:

| Option | Available evidence | Implication |
|---|---|---|
| Hosted TypeSafe Jev | Official docs describe Choice, Score and Noul questions through HTTP and Python/JS SDKs; early access | Needs provider access, network, server-side credentials and privacy review. No verified downloadable iOS Jev weights here. |
| Local `parallel-decisions` | Existing Python/MLX package using ordinary Qwen weights and constrained parallel decoding | Can support a Mac research service; it is not the actual Jev model or a drop-in Swift library. |
| Native iPhone decision implementation | Several apps already have Apple Foundation Models or LiteRT-LM integration | Reuse an existing runtime where possible, but structured generation is not proof of parallel decoding or Jev performance. A true port needs logits/KV-cache support and device measurements. |

Typed output means allowed keys/types/choices, not factual correctness. Repeatability under a pinned model/runtime is distinct from both semantic correctness and a cross-version determinism guarantee. Pin model, schema, preprocessing and candidate order; test repeat runs.

Hosted Jev Score is an ordered judgment primitive, not arbitrary numeric extraction. The local package supports boolean, enum and multi-select; it cannot write arbitrary text or numbers. Dates, names and amounts can nevertheless be handled by **candidate selection**: OCR/parser finds spans, the model chooses an existing span ID or `none`, and code copies and normalizes the original value. This prevents invented values, not selecting the wrong source value or inheriting OCR errors.

### Important update beyond this repo's roadmap

`parallel-decisions` is already v0.2.0 and includes calibration, multi-select, config, locking and an HTTP example. Several items still marked future work in `jev-on-a-laptop/ROADMAP.md` now exist. Do not implement them twice.

However, `parallel-decisions/CALIBRATION.md` reports on 277 labelled decisions:

- Most-confident 25%: 4/69 errors (5.8%; reported 95% upper bound 14.0%).
- A 1% error budget was not attainable on that dataset.
- Calibration changes confidence scale, not wrong answers.

Its README also records a large answer-position bias on a prior evaluation. Test shuffled option orders; do not hide this by placing expected answers first. These are warning signs for the local reproduction, not proof of hosted Jev's performance on these apps.

## iBeleg: start with topic suggestions before persistence

The newer local source is `iBeleg/iBeleg` (inspected revision `8ff682c`), ahead of `_ibeleg_source` (`bbb3f00`). This does not establish the installed app revision.

### Existing pipeline and integration seam

`iBeleg/iBeleg/Sources/iBeleg/Services/MLService.swift` orchestrates Vision OCR, optional Apple Foundation Models, optional Gemma 4 E2B through LiteRT-LM, and a deterministic fallback parser. Both model paths currently consume OCR text rather than image evidence. `Services/ResilientLLMOutputParser.swift` handles malformed generated JSON. `ViewModels/ReceiptViewModel.swift` persists the extraction.

Insert advisory decisions at the **post-OCR, pre-persistence** boundary. This matters because a missing extracted date becomes today's date during persistence, and the Apple provider can return empty `rawText`. Later stored records cannot necessarily recover missing-evidence status.

### Ranked features

1. **Topic suggestions:** select among the eight existing `ReceiptTopic` values in `Models/Receipt.swift`, with an explicit abstention result. Compare with current keywords and Apple/Gemma suggestions; never silently replace user edits. The experimental receipt schema in this report is not this app's production taxonomy.
2. **Review cues:** suggest fixed issue codes such as category conflict, possible hospitality or insufficient evidence. Supplement `ReviewCenterAnalyzer` in `Views/HomeView.swift` and the Gemma pre-export review; do not suppress mandatory checks.
3. **Payment-method normalization:** suggest the existing payment-method enum from explicit OCR wording. Generic “card” must not force debit versus credit. Use keyword rules first.
4. **Candidate-based total/date selection:** promising later, but the current parser does not expose candidate sets. This needs extra extraction/provenance work and is not the first pilot.

Preserve `Services/VATMathAuditor.swift`, `HospitalityValidator.swift` and `DuplicateMatch.swift`. A topic can affect hospitality review, so categorization is not entirely cosmetic. Do not infer attendees, signatures, business purpose, deductibility or compliance; do not change money/date fields or approve accountant exports.

### Feasibility and pilot

`Services/Gemma4Service.swift` uses an approximately 2.6 GB downloaded model, a GPU language backend and a 1,024-token configuration. Its conversation API does not establish access to the logits/KV operations required for this Python/MLX technique. Test typed decisions through an existing native runtime first; an exact parallel-decoding port is separate engineering.

Start with shadow-mode topic suggestions on 300–500 consented/de-identified receipts, with merchant-disjoint splits, all eight classes, German/English, mixed purchases and noisy/empty OCR. Compare existing rules, current app models, local constrained inference and hosted Jev only if approved. Require useful per-class/macro-F1 improvement, no hospitality-recall regression, fewer correction taps and practical whole-receipt latency. Measure actual phones, cold/warm performance, memory and energy; preserve capture on timeout or model absence.

Keep OCR and accounting details out of logs. Private iCloud accounting-data sync is not permission to send receipt text to a hosted model. Existing `ExtractionDiagnostics` timing needs care: `parserLatencyMs` may include failed-provider fallback time, not parser-only work.

## SnipLedger2: an Excel integration, not an iOS app

The newer `Snipledger2/src/SnipLedger.Web/README.md` and `docs/privacy-audit-2026-09.md` identify Office.js as the shipping product line; the root README still emphasizes the older Windows COM/native architecture. Prioritize the web line.

Current browser tools are deterministic: `src/SnipLedger.Web/src/features/support/appConfig.ts` sets `AI_CONFIGURED = false`; `src/host/OfficeHostService.ts` rejects `ai.extract` and `ai.ask`. A separate Python service exists but is not wired into this shipping browser flow. Adding inference will not be faster than these rules; the goal is semantic quality, with parallel decoding potentially cheaper than an equivalent conventional model call.

### Highest-value opportunity: grounded candidate selection

`src/SnipLedger.Web/src/features/extract/fieldExtract.ts` finds regex candidates and then keeps the first occurrence per field. Preserve candidates before deduplication, assign source IDs, and ask which candidate is invoice date / due date / subtotal / tax / total / amount due. Include `NOT_FOUND` and `AMBIGUOUS`.

Code copies the selected original span, parses locale-aware numbers/dates and checks consistency. The decision engine does not invent values or evidence rectangles. This improves the current extraction approach more directly than generic receipt categorization.

### Smallest pilot: financial-statement page classification

Add shadow suggestions alongside `detectStatements()` in `src/SnipLedger.Web/src/features/finstmt/fsDetect.ts`:

- presence of income statement / balance sheet / cash flow / equity / trial balance;
- page role: statement / contents / notes / other / unknown.

First strengthen the current alias/headline rules as a cheap baseline. Measure alternative headings, continuation pages and contents-page false positives. Keep the model advisory and workbook writes unchanged.

Other candidates: rerank evidence in `features/probe/docProbe.ts` with an explicit no-evidence result; assess ambiguous candidate pairs before `features/match/matchEngine.ts` assigns one-to-one matches; tag existing diffs from `features/compare/compareEngine.ts` without suppressing differences.

### Integration constraints

- `src/SnipLedger.Web/src/features/pdf/docText.ts` unifies PDF/OCR/Markdown, but currently drops OCR confidence; Markdown has only synthetic full-page geometry. Preserve honest source quality/provenance.
- Add explicit decision contracts through `Snipledger2/shared/rpc-contract.json`, not misleading extraction confidence fields.
- Maintain deterministic sums in `src/SnipLedger.Web/src/host/sumsEngine.ts` and user-controlled exports in `features/WebTaskPane.tsx`.
- Respect `Snipledger2/tools/check-egress.mjs` and existing CSP. Hosted inference changes the no-cloud proposition; do not embed provider keys or broadly relax egress rules.
- A local Mac model cannot run inside Office.js without a service or a separate runtime port. The native/Python branch is not a shortcut to the shipping browser product.

Start with roughly 200–300 documents / 1,000 labelled pages if practical, splitting by document/template. Compare oracle text against OCR text, class-level errors, abstention, reviewer time and actual-host latency. This larger page study is distinct from the small generic receipt-schema smoke test below.

## Recall and a separate flashcard app

### Verified current architecture

Recall is **“Recall: Speak to Remember”**, an Expo/React Native iOS study app with native Swift modules, not a passive memory recorder.

- `recall-app/PRODUCT.md`: spoken explanations and key-idea coverage, with flashcard review.
- `recall-app/src/voice/coverage.ts`: deterministic keyword/alias/stemming coverage; negation explicitly not modeled.
- `recall-app/src/srs.ts`: SM-2 variant, not FSRS.
- `recall-app/src/screens/SpokenSessionScreen.tsx`: spoken grades currently update scheduling immediately.
- `recall-app/modules/recall-voice/ios/VoiceEngine.swift`: on-device speech recognition; iOS 26 Foundation Models provide qualitative feedback.
- `recall-app/modules/recall-ai/`: Gemma/LiteRT-LM native generation module exists in current source.
- `recall-app/src/import/deckJson.ts`: existing shape/nonempty-field checks.

`recall-ondevice-ai` is an extraction of the generation module, not another consumer app. Its README and the parent README have drifted from current parent source.

### Most valuable feature: semantic spoken-answer coverage

For each supplied expected point and the learner's transcript:

`SUPPORTED | CONTRADICTED | NOT_MENTIONED | UNCERTAIN`

This can distinguish “X causes Y” from “X does not cause Y”, where keywords alone may reward both. Derive totals and suggested grades in code; avoid independent aggregate predictions that can contradict per-point decisions.

Begin with advisory disagreements only. Never let a late model result create a second scheduler update. Any authoritative grading change needs a single explicit commit and learner override.

### Best first pilot: flashcard quality suggestions

After parse, before saving imported/generated cards:

- scope: atomic / multi-concept / unclear;
- answerability: answerable / ambiguous / missing context;
- source support: supported / contradicted / absent / unknown, only when actual source text is supplied.

Code derives the review disposition. Keep empty-field, length, JSON and exact-duplicate checks deterministic. Do not delete or rewrite cards automatically.

Other good uses: taxonomy tagging and semantic duplicate suggestions after cheap retrieval narrows candidate pairs. Keep card generation/explanations generative. For a separate flashcard app using FSRS, keep FSRS authoritative; do not substitute an LLM for interval arithmetic. That separate app was not inspected.

**Privacy issue found in source:** `recall-app/modules/recall-ai/ios/RecallLiteRTEngine.swift` logs the `topic` argument publicly, and the UI can pass source material into that argument. The extracted module has the same pattern. Remove/redact content logging before private-data evaluation; no code was changed in this research.

## Strongest alternatives

| App | Concrete feature | Evidence / integration boundary |
|---|---|---|
| Quilly | Pick primary email/mobile/company from extracted candidates; label contact roles | `Quilly/README.md`, `Quilly/Sources/Quilly/Services/`: OCR → Foundation Models/Gemma → deterministic fallback already exists. Keep contact import/merge user-controlled and no-upload promise intact. |
| SnapTriage | Route OCR/QR evidence to receipt, credential, whiteboard, coupon or unknown workflow | `snaptriage/README.md`: zero network and extraction-before-deletion are explicit invariants. Never infer auto-deletion permission from classification. |
| HearthLog | Suggest asset type and attach scanned manual/receipt to an existing asset | `hearthlog/README.md`: 17 asset types, 46 templates, OCR document vault. Keep maintenance scheduling and manufacturer instructions authoritative. |
| incase | Suggest vault document/record category from title or OCR | `incase/README.md`: local-first sensitive-document vault. No cloud inference without a product/privacy decision. |

Lower priority: contact-less/habit/alarm workflows where rules already suffice. Perilog health logs might benefit from descriptive organization, but clinical severity/doctor triage is not an appropriate first classifier pilot. ValidUntil can use document-type suggestions, not AI authority over passport validity or travel eligibility. Framework choice alone (native vs Expo) does not rule out inference: Recall already has native modules.

## Shared implementation shape

```text
OCR / speech transcript / imported text
    -> deterministic parsing + limited candidate retrieval
    -> short, versioned decision request
    -> interchangeable provider (existing native model / Mac research service / hosted Jev)
    -> validate enum membership, candidate IDs and cross-field invariants
    -> code-owned review policy
    -> reversible suggestion in UI
```

- Start with 3–6 independent questions, not 40 fields over a whole PDF. Context prefill and fields × context memory can dominate.
- Share request/result contracts and evaluation fixtures, not necessarily a single runtime across Swift, TypeScript and Python.
- Include `unknown`/`none`; derive missing fields and hard review requirements from deterministic checks. Model confidence must never suppress mandatory checks.
- Store schema/provider/model version and source provenance alongside suggestions; never overwrite user-confirmed fields.
- Timeouts, cancellation and unavailable models return the existing fallback. No silent cloud fallback.
- Treat OCR/web/document content as untrusted data. Model outputs do not authorize actions or bypass privacy/permission checks.
- Hosted requests need a hardened backend proxy with authentication, TLS, request limits, quotas, redacted logs and server-held credentials. `parallel-decisions/examples/serve.py` is a localhost research example, not a production public service.
- iPhone-to-Mac is off-phone transmission even on a local network. It does not preserve a literal “never leaves your device” promise.
- Apple Foundation Models structured output can test product value without a low-level port. Do not call that Jev or claim parallel-logit speedups.

## Browser-use analogy

The same idea is to select among **actual available targets**, not invent a selector or script:

1. Code gathers a bounded list of eligible links/actions with stable IDs.
2. The decision engine chooses an ID or `none`.
3. Code checks origin, permission, freshness and side-effect policy.
4. The browser executor acts and observes the result.

Use a hierarchy if the candidate list exceeds supported cardinality. Page instructions are untrusted. Typed choices prevent nonexistent IDs, not wrong or malicious choices. Payments, account changes and deletion still need proper authorization. For these iOS apps, native routing is generally simpler than introducing browser automation.

## Pilot and acceptance plan

1. Build roughly 200 public/synthetic, independently labelled examples per chosen domain for initial screening. Cover German/English, noise, mixed categories, unknowns and rare positive cases. Keep private receipts/transcripts out of public repos and providers.
2. Compare current rules, existing app model, local constrained engine, and hosted Jev only if access/privacy permit. Run exactly the same held-out cases; split by document/source so variants do not leak across fit and test.
3. Measure macro-F1/per-class recall, false reassurance, false review rate, abstention and correction taps. Include majority/keyword baselines, shuffled choice orders and repeated calls.
4. Measure end-to-end p50/p95 latency including cold load/OCR/network as applicable; memory, battery and thermal behavior on representative iPhones. Mac results are not iPhone estimates.
5. Fit calibration on separate training/calibration data and freeze routing before final evaluation. Report coverage and error confidence intervals. A small pilot is not proof of a sub-1% error rate.
6. Start in shadow/advisory mode. Ship only if it beats the current approach on useful correction/quality metrics without unacceptable performance or privacy cost. Keep a feature flag and fallback.

Recommended first experiment: the same three-field receipt-routing schema on iBeleg and Snipledger-style sanitized OCR, plus Recall card QC as a separate education dataset. Pick one shipping integration after results, rather than adding a model dependency to every app.

## Artifact and verification

`docs/app-pilots/receipt-routing.schema.json` contains a proposed local-package schema with document kind, purchase topic and text condition. Its categories are an experimental taxonomy, **not** a claim about existing app enum names or a tax classification schema.

Executed on Windows against the actual sibling package, using `PYTHONPATH=../parallel-decisions/src`:

- `Schema.from_json(...)` succeeded.
- Three fields loaded.
- Serialization/deserialization preserved field names and allowed answers.
- Every field includes `unknown`.

**Not executed:** real-tokenizer compilation/collision lint, model inference, accuracy benchmarking, iOS builds, hosted calls. The smoke test verifies loading/validation, not predictions or model compatibility. No model weights were downloaded or app code modified.

## Primary sources

- [TypeSafe System One concepts](https://docs.typesafe.ai/concepts/system-one.md)
- [TypeSafe pre-parsed candidate extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md)
- [TypeSafe confidence semantics](https://docs.typesafe.ai/confidence.md)
- [TypeSafe launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — performance claims are vendor claims, not measurements for these apps.
- `parallel-decisions/README.md`, `CALIBRATION.md`, `src/parallel_decisions/schema.py`, `examples/serve.py`.
- App sources identified above; product status is source-level evidence, not verification of a shipped App Store binary.
