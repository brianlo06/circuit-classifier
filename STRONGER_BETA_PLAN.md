# Stronger Beta Plan

Last updated: 2026-04-08

## Goal

Finish a strong, defensible Beta for the circuit-classifier project:

- keep the fixture demo path stable
- keep the current supported symbol beta path stable
- convert the recent proposal-benchmark gains into a safer live symbol-search improvement
- present the project as a real working product with clear supported scope

The intended Beta claim is:

- the web app provides a stable fixture demo
- the web app also provides a bounded symbol-style beta for a supported set of real schematic images
- supported behavior is benchmark-backed, and unsupported inputs fail honestly

## What We Accomplished So Far

### Core pipeline status

- fixture-style end-to-end demo remains working
- isolated real-gate classifier remains strong
- real crop classifier benchmark remains `55/55`
- supported symbol end-to-end benchmark is now `6/6`
- supported + candidate end-to-end sweep remains `16/16`
- current candidate sweep runtime is now `24.206s` total with `8` explored candidates

### Proposal and benchmark work

- improved proposal benchmarking from partial recall to full benchmark recovery under the richer augmented candidate set
- current proposal benchmark result is `55/55`, recall `1.0`
- recovered difficult style/layout cases including ANSI/XNOR and NAND-heavy real schematics
- added regression coverage around the proposer hard cases

### Web app and product surface

- web app exposes both `Fixture Demo` and `Symbol Beta`
- added built-in supported symbol beta samples
- improved user-facing mode guidance and supported-scope messaging
- added clearer handling for symbol-beta `unknown` results
- kept web app tests green while doing the UI/product hardening

### Planning and repo artifacts

- added `ROADMAP.md`
- clarified Beta, RC, and Done scope around a narrow, defensible finish line

## Current Known-Good Gates

These are the non-regression gates for the stronger Beta push.

1. Real crop classifier benchmark:

```bash
cd /Users/brianlo/circuit-classifier
./.venv/bin/python -m topology.run_symbol_crop_benchmark \
  --benchmark benchmarks/symbol_gate_real_crop_benchmark.json \
  --json
```

Expected:

- `55/55`
- accuracy `1.0`

2. Supported symbol end-to-end benchmark:

```bash
cd /Users/brianlo/circuit-classifier
./.venv/bin/python -m topology.run_symbol_end_to_end_benchmark --json
```

Expected:

- `6/6`
- accuracy `1.0`

3. Proposal benchmark:

```bash
cd /Users/brianlo/circuit-classifier
./.venv/bin/python -m topology.run_symbol_proposal_benchmark --json
```

Expected:

- `55/55`
- recall `1.0`

4. Web app smoke tests:

```bash
cd /Users/brianlo/circuit-classifier
./.venv/bin/python -m unittest tests.test_webapp
```

Expected:

- pass

## Main Remaining Gap

The main unfinished Beta task is no longer live fallback integration. That controlled integration work is now in place.

The current gap is product finish and scope discipline:

- preserve the current `6/6` supported symbol benchmark and `16/16` candidate sweep
- avoid reopening risky runtime tuning now that the obvious search waste has been removed
- decide whether the next pass is product/docs hardening or cautious supported-family expansion

## Next Steps

### Step 1: Freeze the current runtime pass

Requirements:

- keep the current runtime/search hardening as the stopping point unless a new benchmark regression appears
- do not trade additional product complexity for marginal speed wins without a clear need
- keep fixture path untouched

Success criteria:

- supported symbol benchmark stays `6/6`
- candidate sweep stays `16/16`
- current runtime snapshot remains in roughly the current range

### Step 2: Expand the promoted symbol set carefully

After the current runtime pass is frozen:

- add a small number of additional promoted symbol-style cases
- only promote images that are stable in end-to-end analysis, not just proposal recovery
- keep the benchmark-backed supported set honest and bounded

Current note:

- `data/real_schematics/decoder_2x4_asic.png` is now promoted and works in the live symbol-search path as `decoder_2to4`
- `data/real_schematics/decoder_3x8_asic.png` remains candidate-only and should not be promoted from partial decoder-family subset matches

Success criteria:

- promoted symbol set is broader than the previous 5 cases
- new cases do not regress the existing ones

### Step 3: Final Beta product hardening

Complete the Beta surface around the now-stable behavior.

Requirements:

- docs explain how to run the app locally
- docs explain what `Fixture Demo` supports
- docs explain what `Symbol Beta` supports and what it does not
- app behavior for unsupported images is clear and non-misleading

Success criteria:

- local setup is straightforward
- supported sample flows are obvious
- unsupported flows are still user-friendly

### Step 4: Beta signoff sweep

Before calling Beta finished:

- rerun all hard gates
- do a manual smoke test through the app
- confirm the web app is clean enough to present as a working product

Success criteria:

- all benchmark gates are green
- app feels coherent for both fixture and symbol modes
- Beta scope can be described in one short, honest paragraph

## Recommended Implementation Order

1. Freeze the current runtime-hardening pass and document the current benchmark snapshot.
2. Re-run the current symbol and web guardrails.
3. Promote a few more symbol cases only after the supported set remains stable.
4. Do final Beta docs and signoff.

## Beta Finish Line

The stronger Beta is done when all of the following are true:

- fixture demo is still stable
- supported symbol benchmark is still green
- proposal benchmark is still green
- live symbol search includes the current guarded fallback/recovery behavior
- the app clearly communicates supported and unsupported behavior
- the Beta can be shown as a nice working product without overstating its scope

## Current Readiness Note

Current assessment:

- the project is ready for a narrow Beta cut

Why:

- fixture demo is stable
- symbol beta now has a benchmark-backed supported scope
- real crop benchmark is `55/55`
- supported symbol benchmark is `6/6`
- supported + candidate sweep is `16/16`
- current runtime remains in a reasonable local-demo range even though it is slower than the previous handoff snapshot

Recommended decision:

- do not resume open-ended runtime tuning by default
- freeze the current runtime-hardening pass unless a regression appears
- ship the current bounded Beta after final manual app walkthrough and release packaging
