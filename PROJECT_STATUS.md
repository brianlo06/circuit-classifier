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
- supported real-image symbol end-to-end benchmark remains `5/5`
- web app still exposes `Symbol Beta`

Most recent change:

- `topology/analyze_symbol_with_proposals.py` now includes box refinement (`_refine_edge_boxes`) that corrects augmented proposal boxes before topology testing
- this refinement adjusts XOR/AND/OR boxes to align with actual gate and wire positions
- `python3 -m topology.analyze_symbol_with_proposals data/real_schematics/full_adder_tp.jpg --json` now returns `full_adder`
- `topology/analyze_symbol_with_proposals.py` also includes overlap filtering (`_proposals_overlap`) that prevents selecting overlapping proposals in the same candidate set

Latest runtime hardening:

- `topology/pipeline.py` now caches the gate reclassifier and reuses preloaded images during proposal search
- `topology/analyze_symbol_with_proposals.py` now prunes redundant 5-gate ranked proposals, narrows 5-gate per-label candidate fanout, and skips dead 5-gate primary searches that lack any viable XOR seed
- `topology/analyze_symbol_with_proposals.py` now ranks signature candidates with a layout-first bias so real `XOR + AND` half-adder pairs are tried earlier
- `topology/analyze_symbol_with_proposals.py` now prioritizes 2-XOR patterns when multiple strong XOR candidates exist, reducing explored candidates for full adders built from half adders
- `topology/analyze_symbol_with_proposals.py` now penalizes spurious AND detections at XOR row positions and prefers tighter XOR bounding boxes in layout scoring
- `topology/run_symbol_end_to_end_benchmark.py` now reports per-case runtime, explored-candidate counts, and sorted hotspot summaries

Important interpretation:

- the proposal-to-topology blocker for `full_adder_tp.jpg` has been resolved
- the fix involved: (1) overlap filtering to prevent selecting overlapping proposals, (2) box refinement to align augmented boxes with actual gate/wire positions
- `full_adder_tp.jpg` is now promoted to the supported symbol end-to-end benchmark

Current promotion status:

- `full_adder_tp.jpg` is **promoted** in the proposal path
- both benchmark-box path and proposal path for `full_adder_tp.jpg` now work correctly
- supported symbol benchmark is now `5/5`
- it is populated from local synthetic symbol-style schematics plus paired YOLO annotations
- it is useful for early transfer checks, but it is not yet the final real-schematic crop benchmark
- `benchmarks/symbol_gate_real_crop_benchmark.json` now contains the active Tier-2b real benchmark
- that real benchmark currently has `55` manually verified crops and baseline accuracy `1.0`
- pipeline integration has not started yet; the benchmark is now ready to support it
- `topology/finetune_symbol_crop_classifier.py` now provides the adaptation training entry point
- `topology/build_schematic_crop_dataset.py` now builds `data/schematic_crops/` from the local synthetic symbol-style schematic dataset

Current benchmark runtime snapshot from `python3 -m topology.run_symbol_end_to_end_benchmark --json`:

- supported symbol benchmark remains `5/5`
- total elapsed `~7.0s`
- average elapsed `~1.4s`
- max elapsed `~2.5s`
- total explored candidates `5`
- current hotspots:
  - `data/real_schematics/full_adder_tp.jpg`: `~2.5s`, `1` explored candidate
  - `data/real_schematics/full_adder_using_half_adders.jpg`: `~1.5s`, `1` explored candidate
  - `data/real_schematics/half_adder_vlabs.png`: `~1.2s`, `1` explored candidate
  - `data/real_schematics/half_adder_tp.jpg`: `~1.2s`, `1` explored candidate
  - `data/real_schematics/half_adder_gfg.jpg`: `~0.7s`, `1` explored candidate

All 5 cases now explore exactly 1 candidate each (theoretical minimum).

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
3. The proposal-driven symbol-style benchmark now reaches `5/5` after adding the supported real full-adder cases:
   - `data/real_schematics/full_adder_using_half_adders.jpg`
   - `data/real_schematics/full_adder_tp.jpg`
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
- The remaining engineering concern is runtime and generalization of the proposal-search beta path, not whether `full_adder_tp.jpg` is still blocked.

Current next steps:

1. Keep the real crop benchmark as the promotion gate for symbol-style changes.
2. Expand supported symbol-style families cautiously, using explicit end-to-end benchmark additions as the promotion gate.
3. Find more real schematic images from supported sources to expand benchmark coverage.
4. Consider adding new circuit types beyond half_adder/half_subtractor/full_adder.
5. Continue avoiding simple line-drawing style schematics until domain adaptation is addressed.

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
