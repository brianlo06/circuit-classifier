# Circuit Classifier Roadmap

Last updated: 2026-04-08

## Scope

This roadmap assumes the project is targeting a narrow, defensible finish line:

- keep the fixture-style app as a stable demo path
- advance the symbol-style path into a bounded real-schematic analyzer
- make benchmark-backed claims about what is supported

This roadmap does not assume the broader goal of solving arbitrary real-world logic schematics across all visual styles.

## Current State

Current known-good status:

- fixture demo path is stable
- real crop classifier benchmark is strong
- supported symbol end-to-end benchmark is `6/6`
- proposal benchmark is `55/55` when evaluated with aggressive recovery enabled in the benchmark path

Current practical interpretation:

- the project is already close to a strong beta
- the main remaining engineering work is still in the symbol front-end
- the web app can be improved soon, but the real gating work is proposal/search integration and supported-case expansion

## Hard Gates

These should be treated as non-regression gates for symbol-style work:

1. Crop classifier benchmark:

```bash
cd /Users/brianlo/circuit-classifier
./.venv/bin/python -m topology.run_symbol_crop_benchmark \
  --benchmark benchmarks/symbol_gate_real_crop_benchmark.json \
  --json
```

Expected current result:

- `55/55`
- accuracy `1.0`

2. Supported symbol end-to-end benchmark:

```bash
cd /Users/brianlo/circuit-classifier
./.venv/bin/python -m topology.run_symbol_end_to_end_benchmark --json
```

Expected current result:

- `6/6`
- accuracy `1.0`

3. Proposal benchmark:

```bash
cd /Users/brianlo/circuit-classifier
./.venv/bin/python -m topology.run_symbol_proposal_benchmark --json
```

Expected current result:

- `55/55`
- recall `1.0`

Important note:

- the proposal benchmark currently evaluates the richer augmented candidate set
- the live symbol path now includes the guarded fallback/recovery behavior needed for the current supported set

## Beta

Definition:

- a credible web-app beta for a limited set of supported real symbol-style schematics

Required outcomes:

- fixture demo remains stable
- symbol beta remains stable on supported real cases
- web app clearly distinguishes stable fixture mode from symbol beta mode
- unsupported symbol inputs fail clearly and honestly
- supported examples are easy to reproduce locally

Required work:

1. Preserve the current guarded live symbol-search behavior without reopening risky runtime tuning.
2. Preserve the current `6/6` supported symbol benchmark and `16/16` supported + candidate sweep.
3. Expand the supported symbol end-to-end benchmark with additional real images only after they are stable.
4. Improve web app messaging around supported circuits, expected styles, and beta limitations.
5. Clean up docs so the current supported behavior is easy to understand and run.

Promotion criteria:

- supported symbol benchmark remains `100%`
- crop classifier benchmark remains `100%`
- fixture demo remains stable
- symbol beta UX is clear enough that unsupported inputs are not misleading

Estimate:

- roughly `1-2 weeks` part-time from current state

## Release Candidate

Definition:

- the symbol path is stable enough that remaining work is primarily hardening, benchmark expansion, and polish rather than core front-end invention

Required outcomes:

- supported symbol benchmark is larger and more visually diverse
- fallback proposal recovery is integrated without destabilizing ranking
- runtime is acceptable for demo usage
- app behavior is predictable and documented
- tests and docs cover the intended narrow scope

Required work:

1. Expand supported symbol families and source coverage cautiously.
2. Harden ranking and fallback search so stronger recovery improves recall without search pollution.
3. Add more regression tests around proposal search and web-app behavior.
4. Improve output explanations and warnings in the UI.
5. Finalize local run/deployment documentation.

Promotion criteria:

- larger supported end-to-end symbol benchmark passes consistently
- current hard gates remain green
- runtime remains acceptable on supported examples

Estimate:

- roughly `1-2 weeks` after Beta

## Done

Definition:

- a bounded, benchmark-backed digital logic analyzer for a defined subset of real symbol-style schematics, plus the stable fixture demo

Required outcomes:

- explicit supported circuit families
- explicit supported visual styles and source expectations
- benchmark-backed claims only
- polished web demo and local setup
- no ambiguous claim that the project solves arbitrary schematic analysis

Done means:

- the project is complete for the narrow scope
- the broader “general real-schematic analyzer” problem is intentionally out of scope

Estimate:

- roughly `2-4 weeks` total from the current repo state for the narrow finish line

## Recommended Next Step

The next highest-value task is:

1. freeze the current runtime/search hardening pass unless regressions appear
2. keep the current supported symbol benchmark and candidate sweep as hard promotion gates
3. expand supported real symbol benchmark coverage cautiously
4. finish docs and Beta presentation around the current guarded scope

This is the correct next step because the remaining issue is no longer basic recovery integration. The higher-value work now is scope discipline, benchmark preservation, and careful expansion.
