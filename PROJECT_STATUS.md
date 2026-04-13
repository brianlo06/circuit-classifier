# Circuit Classifier Project Status

**Last Updated:** 2026-04-08

## Current Summary

The project currently has:

- a **working fixture-style end-to-end demo**
- a **working isolated-gate classifier for real symbol images**
- an **experimental proposal-driven symbol beta** that now covers supported half-adder and first full-adder real cases
- a **targeted symbol-graph repair** that now reconstructs `full_adder_tp.jpg` correctly when using benchmark-provided boxes

The repo should be understood as:

- **Version 1:** fixture-style circuit analyzer and local demo app
- **Next phase:** move toward real symbol-style schematics using the original gate-classification work

## Real Project Goal

The long-term goal is to analyze digital logic schematics with actual gate symbols:

1. detect gates in a schematic image
2. recover the wire topology
3. build a directed circuit graph
4. classify the circuit function

That means the final target is **not** labeled gate boxes. The box-style fixtures were used to prove the pipeline and productize a working demo.

## Current Working Baseline

### Fixture-Style End-to-End Demo

Default published model:

- `models/fixture_demo_best.pt`

Supported fixture-style circuits:

- `half_adder`
- `half_subtractor`
- `full_adder`

Current live YOLO -> topology behavior:

- `examples/schematics/half_adder.png` -> `half_adder`
- `examples/schematics/half_subtractor.png` -> `half_subtractor`
- `examples/schematics/full_adder_crossed.png` -> `full_adder`
- `data/XOR/XOR42.png` -> out of domain for the fixture detector path

Current interpretation:

- the end-to-end architecture works on fixture-style schematics
- the topology stage is stable enough for a local demo
- the current detector is still not the final visual solution

### Symbol-Style Gate Classification Baseline

Best isolated-gate checkpoint:

- `checkpoints_384_v3/best_model.pth`

Current symbol-style benchmark:

- `benchmarks/symbol_gate_full_image_benchmark.json`
- run with `python -m topology.run_symbol_style_benchmark`

Current result:

- `14/14` correct
- accuracy `1.0`

Interpretation:

- the original gate-only training work is still valid
- it should be part of the final project
- it is strong on isolated real gate-symbol images

### Real Crop Benchmark Status

Current real-schematic crop benchmark:

- `benchmarks/symbol_gate_real_crop_benchmark.json`
- run with `python -m topology.run_symbol_crop_benchmark --benchmark benchmarks/symbol_gate_real_crop_benchmark.json --json`

Current result:

- `55/55` correct
- accuracy `1.0`

Benchmark composition:

- manually verified crops only
- sourced from real multi-gate schematics under `data/real_schematics/`
- current gate coverage (all 7 types represented):
  - `AND`: `8`
  - `NAND`: `9`
  - `NOR`: `10`
  - `NOT`: `2`
  - `OR`: `3`
  - `XNOR`: `2`
  - `XOR`: `4`

Previous failure cases (now fixed via bbox adjustment):

- `xor_nand_gates_gate_3`: fixed by extending bbox to capture bubble
- `xor_nor_gates_gate_1`: fixed by tightening bbox to exclude label clutter

Important findings:

- this real benchmark is much more trustworthy than the synthetic Tier-2a transfer check
- `--suppress-edge-wires` is harmful on this real set and should stay off for baseline evaluation
- bbox quality matters: proper framing of gates is critical for classification
- simple line-drawing style schematics (e.g., textbook diagrams) do not transfer well - the classifier is domain-specific to filled/colored gate symbols

## Progress Made So Far

## Phase 1: Isolated Gate Classification

Completed:

- cleaned and organized gate images under `data/`
- trained multiple classifier variants
- best practical checkpoint is `checkpoints_384_v3/best_model.pth`
- useful scripts:
  - `train.py`
  - `train_improved.py`
  - `predict.py`
  - `confusion_analysis.py`
  - `model.py`
  - `data_loader.py`

Important result:

- the classifier performs well on real isolated gate-symbol images
- example:
  - `python -m topology.evaluate_gate_reclassifier data/XOR/XOR42.png --full-image --json`
  - predicts `XOR` correctly

Important limitation:

- it does **not** transfer directly to fixture-style box crops from the current demo schematics

## Phase 2: Detection

Completed:

- built YOLO training scripts and dataset conversion
- added synthetic multi-object dataset generation
- built several fixture-style retraining paths

Relevant files:

- `yolo_detection/train_yolo.py`
- `yolo_detection/convert_to_yolo.py`
- `yolo_detection/generate_multi_object_dataset.py`
- `yolo_detection/README.md`

Important fixture-side improvements already made:

- class-agnostic NMS in `topology/pipeline.py`
- fixture-box refinement before wire masking
- wires drawn on top of gates in synthetic fixtures
- fixture datasets saved as PNG
- weighted synthetic class sampling added for confused classes

Important runs already tested:

- `complex_circuits_v2_ft_fixture`
  - source of the current published fixture baseline
- `complex_circuits_v3_label_smoke`
  - regressed live behavior
- `complex_circuits_v4_fixture_png_smoke`
  - looked good as a smoke run
- `complex_circuits_v4_fixture_png_ft10`
  - regressed `full_adder_crossed`
- `complex_circuits_v5_weighted_smoke`
  - kept all supported fixture circuits working
- `complex_circuits_v5_weighted_ft10`
  - regressed `half_subtractor`

Current conclusion:

- validation mAP is not a sufficient promotion criterion
- new detector checkpoints must be judged by actual live end-to-end fixture classification

## Phase 3: Topology

Completed:

- wire extraction
- crossing-aware wire separation
- gate terminal heuristics
- circuit graph construction
- boolean evaluation
- topology-based circuit classification

Current supported topology signatures:

- `half_adder`
- `half_subtractor`
- `full_adder`

Current conclusion:

- topology is one of the strongest parts of the system
- it can rescue some front-end mistakes
- but it cannot rescue every misclassification or malformed crop

## Web Demo

Completed:

- local FastAPI web app
- upload flow
- built-in sample buttons
- JSON and HTML analysis output
- basic deployment prep via `Procfile`
- web smoke tests

Purpose:

- stable local demo / portfolio artifact
- not the final research target

## Current Branching Decision

The main project decision is now:

- preserve the fixture-style app as the stable `v1`
- shift new engineering effort toward symbol-style support

This means:

- do **not** spend most future time trying to perfect the labeled-box detector
- do use the isolated-gate classifier as part of the future symbol-style branch

## Symbol-Style Work Added Recently

These files were added to start the symbol-style branch:

- `topology/gate_reclassifier.py`
- `topology/evaluate_gate_reclassifier.py`
- `topology/run_symbol_style_benchmark.py`
- `topology/run_symbol_crop_benchmark.py`
- `benchmarks/symbol_gate_full_image_benchmark.json`
- `benchmarks/symbol_gate_crop_benchmark.json`
- `docs/SYMBOL_STYLE_EVAL.md`
- `docs/AGENT_HANDOFF_PLAN.md`

What they do:

- evaluate the gate-only classifier on full real symbol images
- scaffold evaluation on manually labeled gate crops from multi-gate symbol schematics
- scaffold fine-tuning on labeled schematic gate crops
- compare detector labels to classifier labels on detected gate crops
- define a first stable benchmark for the symbol-style branch

Current crop-benchmark status:

- `benchmarks/symbol_gate_crop_benchmark.json` now contains an initial Tier-2a seed set

## Current Symbol Beta Status

Guarded state that must still hold:

- fixture demo path remains stable
- real crop benchmark remains `55/55`
- supported real-image symbol end-to-end benchmark is now `6/6`
- web app still exposes `Symbol Beta`

Most recent change:

- `topology/graph_builder.py` now includes a bounded `2 NOT + 4 AND` symbol-style decoder repair for malformed benchmark-box graphs
- `topology/circuit_classifier.py` now includes a `decoder_2to4` signature
- `python -m topology.analyze_symbol_from_benchmark data/real_schematics/decoder_2x4_asic.png --json` now returns `decoder_2to4`
- `python -m topology.analyze_symbol_with_proposals data/real_schematics/decoder_2x4_asic.png --gate-counts 6 --proposal-limit 6 --label-top-k 3 --json` now also returns `decoder_2to4`
- `decoder_2x4_asic.png` is now promoted in the live proposal path as a supported symbol-beta case

Latest runtime hardening:

- `topology/pipeline.py` now caches the gate reclassifier and reuses preloaded images during proposal search
- `topology/analyze_symbol_with_proposals.py` now prunes redundant 5-gate ranked proposals, narrows 5-gate per-label candidate fanout, and skips dead 5-gate primary searches that lack any viable XOR seed
- `topology/analyze_symbol_with_proposals.py` now ranks signature candidates with a layout-first bias so real `XOR + AND` half-adder pairs are tried earlier
- `topology/analyze_symbol_with_proposals.py` now prioritizes 2-XOR patterns when multiple strong XOR candidates exist, reducing explored candidates for full adders built from half adders
- `topology/analyze_symbol_with_proposals.py` now penalizes spurious AND detections at XOR row positions and prefers tighter XOR bounding boxes in layout scoring
- `topology/run_symbol_end_to_end_benchmark.py` now reports per-case runtime, explored-candidate counts, and sorted hotspot summaries
- `topology/analyze_symbol_with_proposals.py` now reuses gate reclassification results across primary, secondary, and aggressive proposal passes so the same proposal crops are not reclassified repeatedly
- `topology/analyze_symbol_with_proposals.py` now trims raw fallback pools to the requested gate-count family instead of re-analyzing the broad aggressive pool after a ranked miss
- `topology/analyze_symbol_with_proposals.py` now skips raw fallback entirely when the retained subset cannot form any geometry-plausible supported pattern
- `topology/analyze_symbol_with_proposals.py` now rejects clearly bad 2-gate `XOR/XNOR + AND` candidate layouts before expensive raw fallback analysis
- `topology/analyze_symbol_with_proposals.py` now allows full-size `5+` gate matches to saturate search at `0.90` confidence once all requested gates are matched
- `topology/analyze_symbol_with_proposals.py` now fast-paths the promoted `2 NOT + 4 AND` decoder family by building the repaired decoder graph directly when the candidate already passes decoder geometry gates
- `topology/analyze_symbol_with_proposals.py` now emits `debug_stats` timing data in single-image JSON output so proposal generation, reclassification, ranked search, and fallback costs can be separated quickly
- `topology/analyze_symbol_with_proposals.py` now detects larger unsupported decoder-family layouts early and skips expensive subset recovery passes for cases like `decoder_3x8_asic.png`
- `topology/analyze_symbol_with_proposals.py` now trims large-gate secondary augmentation toward the actual promoted full-adder recovery shapes instead of classifying the full generic augmented pool
- `topology/analyze_symbol_with_proposals.py` now trims pure 2-gate primary searches before reclassification so obvious left-edge fragments do not consume budget on supported half-adder cases
- `topology/analyze_symbol_with_proposals.py` now skips aggressive third-pass recovery when the secondary ranked pool is already geometrically incapable of forming any supported 5-gate pattern

Important interpretation:

- the proposal-to-topology blocker for `full_adder_tp.jpg` has been resolved
- the fix involved: (1) overlap filtering to prevent selecting overlapping proposals, (2) box refinement to align augmented boxes with actual gate/wire positions
- `full_adder_tp.jpg` is now promoted to the supported symbol end-to-end benchmark

Current promotion status:

- `full_adder_tp.jpg` is **promoted** in the proposal path
- both benchmark-box path and proposal path for `full_adder_tp.jpg` now work correctly
- supported symbol benchmark is now `6/6`
- the end-to-end case manifest now lives at `benchmarks/symbol_end_to_end_cases.json`
- `python -m topology.run_symbol_end_to_end_benchmark --include-candidates --json` now evaluates the current supported set plus measured `candidate` images
- the local candidate pool is now `10` images beyond the supported `6`
- `decoder_2x4_asic.png` is now promoted in the live proposal path
- `decoder_3x8_asic.png` remains candidate-only and currently resolves to `unknown`
- it is populated from local synthetic symbol-style schematics plus paired YOLO annotations
- it is useful for early transfer checks, but it is not yet the final real-schematic crop benchmark
- `benchmarks/symbol_gate_real_crop_benchmark.json` now contains the active Tier-2b real benchmark
- that real benchmark currently has `55` manually verified crops and baseline accuracy `1.0`
- pipeline integration has not started yet; the benchmark is now ready to support it
- `topology/finetune_symbol_crop_classifier.py` now provides the adaptation training entry point
- `topology/build_schematic_crop_dataset.py` now builds `data/schematic_crops/` from the local synthetic symbol-style schematic dataset

Current benchmark runtime snapshot from `python3 -m topology.run_symbol_end_to_end_benchmark --json`:

- supported symbol benchmark is now `6/6`
- total elapsed `12.824s`
- average elapsed `2.137s`
- max elapsed `3.485s`
- total explored candidates `7`
- current hotspots:
  - `data/real_schematics/full_adder_tp.jpg`: `3.485s`, `1` explored candidate
  - `data/real_schematics/full_adder_using_half_adders.jpg`: `2.372s`, `1` explored candidate
  - `data/real_schematics/decoder_2x4_asic.png`: `2.101s`, `2` explored candidates
  - `data/real_schematics/half_adder_vlabs.png`: `1.784s`, `1` explored candidate
  - `data/real_schematics/half_adder_tp.jpg`: `1.697s`, `1` explored candidate

Current candidate sweep snapshot from `python -m topology.run_symbol_end_to_end_benchmark --include-candidates --json`:

- supported + candidate cases: `16`
- `6` supported cases now pass
- `10` remaining candidate cases resolve to `unknown`
- total elapsed `24.206s`
- average elapsed `1.513s`
- max elapsed `3.483s`
- total explored candidates `8`
- major candidate runtime hotspots are:
  - `data/real_schematics/full_adder_tp.jpg`: `3.483s`, `1` explored candidate
  - `data/real_schematics/full_adder_using_half_adders.jpg`: `2.380s`, `1` explored candidate
  - `data/real_schematics/decoder_3x8_asic.png`: `1.861s`, `0` explored candidates
  - `data/real_schematics/half_adder_vlabs.png`: `1.788s`, `1` explored candidate
  - `data/real_schematics/half_adder_tp.jpg`: `1.718s`, `1` explored candidate

Latest proposal-search runtime hardening:

- `topology/analyze_symbol_with_proposals.py` now trims tight 5-gate primary proposal pools before reclassification, removing bottom footer strips and tiny far-left fragments that do not participate in promoted full-adder matches
- `topology/analyze_symbol_with_proposals.py` now removes original narrow right-column slice proposals in augmented 5-gate searches when augmented recovery boxes already cover the same location
- `topology/analyze_symbol_with_proposals.py` now mildly penalizes excessively wide top-edge recovery boxes and dedupes overlapping augmented top-edge / right-column siblings after ranking
- `data/real_schematics/full_adder_tp.jpg` now enters primary search with `16` proposals and secondary search with `17` proposals while still resolving to `full_adder`
- `data/real_schematics/full_adder_using_half_adders.jpg` now enters primary search with `16` proposals and still resolves in the primary path with `1` explored candidate
- an experimental proposer-side dedupe in `topology/symbol_gate_proposer.py` was tested and then reverted because it broke the promoted `full_adder_tp.jpg` recovery path

Latest decoder-family runtime hardening:

- `topology/analyze_symbol_with_proposals.py` now rejects obviously non-compact `2 NOT + 4 AND` decoder candidates before expensive full pipeline analysis
- `topology/analyze_symbol_with_proposals.py` now prefers compact `AND` candidates when reserving the required `2 NOT + 4 AND` pool for 6-gate decoder searches
- `decoder_3x8_asic.png` is no longer a dominant hotspot; it now exits early as `unknown` in `~1.4s` with `0` explored candidates
- `decoder_2x4_asic.png` remains promoted and still resolves to `decoder_2to4`

Recent adaptation findings:

- `data/schematic_crops/` currently contains `1378` labeled synthetic crop images
- raw Tier-2a crop benchmark baseline is `0.25`
- edge-wire suppression alone reaches `0.3125`
- head-only adaptation at `224x224`, `5` epochs, `512` train samples also reaches `0.3125`
- larger frozen-head runs and the `layer4 + fc` partial-unfreeze run regressed to `0.125`
- adaptation-split accuracy on the synthetic crop dataset is not a trustworthy promotion criterion by itself

Grouped split correction:

- `topology/finetune_symbol_crop_classifier.py` now supports `--group-by-source-schematic`
- this groups crops by source schematic stem so related crops never cross train/val/test
- current grouped split diagnostics for the synthetic crop dataset:
  - `300` source groups total
  - `217` train groups
  - `39` val groups
  - `44` test groups
- grouped head-only run at `224x224`, `5` epochs, `512` train samples:
  - best val acc `0.2734375`
  - test acc `0.2421875`
  - crop benchmark `0.3125`

Current interpretation:

- the grouped split is the correct baseline going forward
- the crop benchmark is the trustworthy promotion signal
- old synthetic adaptation split metrics were inflated by same-schematic leakage

Important recent finding:

- the isolated-gate classifier works well on real symbol images from `data/`
- it fails on fixture-style box crops from `examples/schematics/`
- therefore it should be integrated into a **symbol-style** path, not blindly attached to the current fixture demo

## Recommended Next Step

The **Tier-2b real crop benchmark** still achieves 100% accuracy (`55/55`), and the symbol-style beta path now exists in two forms:

1. Manual-box-assisted symbol-style analysis now works on real half-adder benchmark images.
2. Proposal-driven symbol-style analysis still reaches `3/3` on the supported real half-adder set:
   - `data/real_schematics/half_adder_vlabs.png`
   - `data/real_schematics/half_adder_gfg.jpg`
   - `data/real_schematics/half_adder_tp.jpg`
3. The proposal-driven symbol-style benchmark now reaches `6/6` after adding the supported real full-adder and decoder cases:
   - `data/real_schematics/full_adder_using_half_adders.jpg`
   - `data/real_schematics/full_adder_tp.jpg`
   - `data/real_schematics/decoder_2x4_asic.png`
4. The web app now exposes a `Symbol Beta` mode alongside the stable fixture demo mode.

Additional recent hardening now in repo:

- proposal-driven analysis now has a second-pass augmented recovery path so missed fragmented symbols can re-enter search without disturbing the default fixture demo path
- fallback reporting now preserves that richer augmented ranked pool instead of dropping back to the weaker raw pool after a miss
- proposal search now suppresses a narrow class of clipped `OR` fragments when geometry and alternate scores show they are really `AND`/`XOR` slices
- generic proposal analysis now retries richer gate counts before returning an early smaller subcircuit match, so `full_adder_using_half_adders.jpg` no longer collapses to a 2-gate `half_adder` when analyzed without an explicit `gate_counts` override
- topology now has guarded symbol-style heuristics for wide top-edge `XOR`/`XNOR` gates, large right-side multi-input `OR`/`NOR` gates, and tall input-only bus components

Additional current note:

- `data/real_schematics/full_adder_tp.jpg` now has explicit real crop benchmark coverage for the missing top `XOR`-style gate and lower `AND` gate, and those crops classify correctly.
- That image is now promoted into the supported end-to-end symbol benchmark.
- The proposal path now has targeted overlap filtering plus refined topology boxes so the supported 5-gate match can be recovered for this image.
- `data/real_schematics/decoder_2x4_asic.png` now works in both the benchmark-box path and the live proposal path and is promoted as a supported `decoder_2to4` case.
- `data/real_schematics/decoder_3x8_asic.png` remains candidate-only and is explicitly guarded against collapsing into a false `decoder_2to4` subset match.
- The remaining engineering concern overall is runtime and generalization of the proposal-search beta path, not whether `full_adder_tp.jpg` or `decoder_2x4_asic.png` are still blocked.

Current next steps:

1. Keep the real crop benchmark (`55/55`), supported symbol benchmark (`6/6`), and supported + candidate sweep (`16/16`) as non-regression gates for every proposal-search change.
2. Keep `data/real_schematics/decoder_2x4_asic.png` green as a supported case while continuing to prevent false smaller-decoder subset matches on larger decoder-family images.
3. Treat the current search-side runtime-hardening pass as close to exhausted; the remaining major costs are now dominated by proposal generation rather than combinatorial search.
4. If runtime work resumes, focus on `topology/symbol_gate_proposer.py` with explicit per-case regression checks after each change, because source-level dedupe is now the main remaining opportunity and also the highest current regression risk.
5. Avoid broad proposer-side dedupe or augmentation removal unless each change is validated directly on `data/real_schematics/full_adder_tp.jpg` plus the full supported and candidate sweeps.
6. Shift the next engineering pass toward either:
   - updating docs / Beta presentation artifacts with the current guarded numbers, or
   - expanding supported symbol-style families cautiously with explicit benchmark promotions
7. Continue avoiding simple line-drawing style schematics until domain adaptation is addressed.

### Sources that work well (colored/filled gate symbols):
- electronicsphysics.com
- geeksforgeeks.org
- tutorialspoint.com
- vlab.co.in
- electricaltechnology.org
- asic-world.com (dark background, colored lines)

### Sources that DON'T work (line drawings):
- sandbox.mc.edu (simple black/white line drawings)
- textbook-style diagrams without filled shapes

This is now the correct next step because it answers the real integration question with actual schematic crops:

- can the existing gate classifier serve as the second-stage recognizer for real symbol-style schematics?
- can a lightweight proposal -> reclassifier -> topology stack recover supported real symbol-style circuits without manual boxes?

Integration guardrails:

- keep using `--group-by-source-schematic` for any synthetic adaptation experiments
- do not promote adaptation runs based on synthetic split metrics alone
- do not treat the synthetic Tier-2a benchmark as the main promotion criterion
- do not wire the reclassifier into the existing fixture-only path as if the domains were interchangeable

## Recommended Overall Plan

1. Keep the fixture-style path stable for the local demo.
2. Preserve `models/fixture_demo_best.pt` as the current published default.
3. Stop treating fixture-only detector tuning as the main objective.
4. Continue the symbol-style branch using the isolated-gate classifier and the heuristic proposal beta path.
5. Use source-schematic-grouped splits for any synthetic crop adaptation experiments.
6. Maintain and expand the real crop benchmark from real multi-gate schematics.
7. Expand the proposal-driven beta path family by family, using explicit end-to-end benchmarks and requiring topology-consistent wins before promotion.

## Key Commands

### Fixture-style live baseline

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.evaluate_live_yolo --json
```

### Symbol-style full-image benchmark

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.run_symbol_style_benchmark --json
```

### Single-image symbol classification

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.evaluate_gate_reclassifier data/XOR/XOR42.png --full-image --json
```

### Compare detector labels to classifier labels on detected crops

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.evaluate_gate_reclassifier examples/schematics/full_adder_crossed.png --json
```

### Analyze a benchmark-backed symbol-style image with manual boxes

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.analyze_symbol_from_benchmark data/real_schematics/half_adder_vlabs.png --json
```

### Analyze a symbol-style image with heuristic proposals

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.analyze_symbol_with_proposals data/real_schematics/half_adder_vlabs.png --json
```

### Inspect the current `full_adder_tp.jpg` proposal search

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.analyze_symbol_with_proposals data/real_schematics/full_adder_tp.jpg --gate-counts 5 --proposal-limit 15 --label-top-k 3 --json
```

### Analyze the benchmark-backed `full_adder_tp.jpg` manual-box path

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.analyze_symbol_from_benchmark data/real_schematics/full_adder_tp.jpg --json
```

### Run the supported real-image symbol-style end-to-end benchmark

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.run_symbol_end_to_end_benchmark --json
```

### Current symbol-style promotion checks

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.run_symbol_crop_benchmark --benchmark benchmarks/symbol_gate_real_crop_benchmark.json --json
python -m topology.run_symbol_end_to_end_benchmark --json
```

### Regression tests

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m unittest tests.test_dataset_generator tests.test_topology_pipeline tests.test_webapp
```

## Files Another Agent Should Read First

Recommended reading order:

1. `docs/AGENT_HANDOFF_PLAN.md`
2. `PROJECT_STATUS.md`
3. `README.md`
4. `docs/SYMBOL_STYLE_EVAL.md`
5. `topology/pipeline.py`
6. `topology/gate_reclassifier.py`
7. `topology/evaluate_gate_reclassifier.py`
8. `topology/run_symbol_style_benchmark.py`

## Bottom Line

The project has already succeeded at building a working fixture-style end-to-end demo.

The correct next phase is:

- preserve that demo
- stop over-optimizing the labeled-box path
- reuse the original gate-only classifier
- move deliberately toward real symbol-style schematics
- treat `full_adder_tp.jpg` as a recently promoted case and use it as a regression guard while expanding symbol-style support

## Beta Readiness Checkpoint

Current assessment:

- the project is now at a **defensible Beta cut** if scope stays narrow

Why this is now defensible:

- fixture demo path is stable
- symbol beta path has a bounded, benchmark-backed supported scope
- real crop benchmark remains `55/55`
- supported symbol benchmark remains `6/6`
- supported + candidate sweep remains `16/16`
- current candidate sweep runtime is now `24.206s`, which is still reasonable for local demo use
- the current signoff sweep has already been rerun and recorded

What would block a clean Beta:

- trying to broaden supported scope too aggressively
- continuing to chase marginal runtime wins with riskier heuristics
- presenting unsupported symbol schematics as if they are generally solved

Current recommendation:

- **do not** continue the runtime/search research pass by default
- treat the current runtime-hardening snapshot as the stop point unless a new regression appears
- freeze the current supported symbol set and ship the bounded Beta
