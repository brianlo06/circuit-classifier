# Circuit Classifier Agent Handoff Plan

This file is intended to be handed directly to another coding agent so it can pick up the project without re-discovering the repo state.

## 1. Project Goal

The real goal is:

- analyze **digital logic circuit schematics**
- detect the gates in the schematic
- recover the wiring topology
- build a circuit graph
- classify the circuit function

The intended long-term input domain is **real symbol-style logic schematics with actual gate drawings**, not just synthetic boxes with text labels.

The current repo contains both:

- a **working fixture-style demo path**
- a **separate gate-only classifier** trained on real gate symbol images

Those two lines of work have not been fully merged yet.

## 2. Current Reality

The repo currently has a strong `v1` demo for **fixture-style schematics** and a separate strong baseline for **isolated symbol-style gate classification**.

The project is **not finished**.

What is done:

- fixture-style end-to-end pipeline works locally
- web demo works locally
- topology reasoning is solid on supported fixture circuits
- isolated-gate classifier works on real gate symbol images from `data/`

What is not done:

- symbol-style multi-gate schematics are not solved end to end
- the isolated-gate classifier is not integrated into the main pipeline
- the current YOLO fixture detector is not the final visual strategy

## 3. Recommended Framing

Treat the repo as having two tracks:

### Track A: Fixture-Style Demo

Purpose:

- maintain a stable, GitHub-ready end-to-end demo
- keep the app working
- preserve a known-good topology backend

### Track B: Symbol-Style Final Goal

Purpose:

- move toward real gate-symbol schematics
- reuse the original gate-only training work
- eventually replace the box-label shortcut with actual gate recognition

This split is important because over-optimizing the fixture path has diminishing returns.

## 4. Current Working Baselines

### A. Fixture-Style End-to-End Baseline

Default published model:

- `models/fixture_demo_best.pt`

Current fixture-style supported circuits:

- `half_adder`
- `half_subtractor`
- `full_adder`

Live evaluation command:

```bash
python3 -m topology.evaluate_live_yolo --json
```

Expected current behavior:

- `examples/schematics/half_adder.png` -> `half_adder`
- `examples/schematics/half_subtractor.png` -> `half_subtractor`
- `examples/schematics/full_adder_crossed.png` -> `full_adder`
- `data/XOR/XOR42.png` -> still out of domain / unsupported for this path

### B. Symbol-Style Isolated Gate Baseline

Best current isolated-gate checkpoint:

- `checkpoints_384_v3/best_model.pth`

Core result:

- `results.json` reports strong held-out performance
- the model still performs well on a curated real-symbol benchmark

Benchmark runner:

```bash
python -m topology.run_symbol_style_benchmark
```

Current benchmark result:

- `benchmarks/symbol_gate_full_image_benchmark.json`
- `14/14` correct
- accuracy `1.0`

Interpretation:

- the original gate-only training work is still valid
- it should absolutely be part of the final project

Current real-crop benchmark status:

- `benchmarks/symbol_gate_real_crop_benchmark.json`
- `55` manually verified real schematic crops
- current baseline: `55/55` correct, accuracy `1.0`
- all 7 gate types represented (AND, NAND, NOR, NOT, OR, XNOR, XOR)
- previous failures fixed via bbox adjustments
- images converted from .webp to .png for compatibility

Domain finding:

- the classifier works well on filled/colored gate symbol schematics
- simple line-drawing style schematics (textbook diagrams) do NOT transfer well

Important note:

- this benchmark exists specifically to ground future work in real data
- do **not** return to synthetic adaptation optimization before the real benchmark is being used as the main promotion check
- do use it as a promotion gate for the current symbol beta work

### Current Guarded State

These must not regress:

- fixture demo path is stable and should not be disturbed
- real crop benchmark is `55/55`
- supported real-image symbol end-to-end benchmark is `6/6`
- supported + candidate end-to-end sweep is `16/16`
- web app still exposes `Symbol Beta`

### Current `full_adder_tp.jpg` Status

This is now resolved:

- `full_adder_tp.jpg` is **promoted** to the supported symbol end-to-end benchmark
- both the benchmark-box path and proposal path now work correctly
- `python3 -m topology.analyze_symbol_with_proposals data/real_schematics/full_adder_tp.jpg --json` now returns `full_adder`

What was added to fix this:

- `topology/analyze_symbol_with_proposals.py` now includes `_refine_edge_boxes` which corrects augmented proposal boxes before topology testing
- this refinement adjusts XOR/AND/OR boxes to align with actual gate and wire positions
- `topology/analyze_symbol_with_proposals.py` also includes `_proposals_overlap` which prevents selecting overlapping proposals in the same candidate set
- the combination of overlap filtering + box refinement allows the correct 5-gate subset to be found and correctly wired

### Current Runtime Snapshot

Latest benchmark command:

- `python3 -m topology.run_symbol_end_to_end_benchmark --json`

Current result:

- supported real-image symbol end-to-end benchmark remains `6/6`
- total elapsed `12.824s`
- average elapsed `2.137s`
- max elapsed `3.485s`
- total explored candidates `7`

Current hotspots:

- `data/real_schematics/full_adder_tp.jpg`: `3.485s`, `1` explored candidate
- `data/real_schematics/full_adder_using_half_adders.jpg`: `2.372s`, `1` explored candidate
- `data/real_schematics/decoder_2x4_asic.png`: `2.101s`, `2` explored candidates

What was added recently to get here:

- `topology/pipeline.py` now caches the gate reclassifier and reuses preloaded images during proposal search
- `topology/analyze_symbol_with_proposals.py` now prunes redundant 5-gate ranked proposals and narrows 5-gate per-label candidate fanout
- `topology/analyze_symbol_with_proposals.py` now skips dead 5-gate primary searches when the ranked pool has no viable XOR seed
- `topology/analyze_symbol_with_proposals.py` now uses layout-first ranking for signature candidates so real half-adder and full-adder layouts are tried earlier
- `topology/run_symbol_end_to_end_benchmark.py` now reports per-case runtime, explored-candidate totals, and sorted hotspot summaries
- `topology/analyze_symbol_with_proposals.py` now rejects bad 2-gate `XOR/XNOR + AND` layouts before fallback
- `topology/analyze_symbol_with_proposals.py` now stops early on larger unsupported decoder-family layouts instead of spending time on false `decoder_2to4` subset recovery
- `topology/analyze_symbol_with_proposals.py` now trims 5-gate secondary augmentation toward the actual promoted full-adder recovery shapes
- `topology/analyze_symbol_with_proposals.py` now trims 2-gate primary proposal pools before reclassification
- `topology/analyze_symbol_with_proposals.py` now skips aggressive third-pass recovery when the secondary pool is already geometrically impossible for any supported pattern

Current candidate sweep snapshot from `python3 -m topology.run_symbol_end_to_end_benchmark --include-candidates --json`:

- `16/16` cases currently resolve as expected
- total elapsed `24.206s`
- average elapsed `1.513s`
- max elapsed `3.483s`
- total explored candidates `8`

Current recommendation:

- do not keep chasing smaller runtime wins by default
- the remaining dominant costs are now mostly proposal generation and gate reclassification rather than obviously wasteful search branches
- prefer shifting the next pass to Beta docs/product hardening or cautious supported-family expansion

### Latest Runtime Pickup Note

If you are resuming from the latest runtime pass, the current verified baseline is:

- `python -m topology.run_symbol_end_to_end_benchmark --json` -> `6/6`
- total elapsed `12.824s`
- total explored candidates `7`
- `python -m topology.run_symbol_end_to_end_benchmark --include-candidates --json` -> `16/16`
- total elapsed `24.206s`
- total explored candidates `8`

Current supported-case hotspots:

- `data/real_schematics/full_adder_tp.jpg`: `3.485s`, `1` explored candidate
- `data/real_schematics/full_adder_using_half_adders.jpg`: `2.372s`, `1` explored candidate
- `data/real_schematics/decoder_2x4_asic.png`: `2.101s`, `2` explored candidates

Recent accepted runtime changes now in repo:

- `topology/analyze_symbol_with_proposals.py` trims tight 5-gate primary pools before reclassification
- `topology/analyze_symbol_with_proposals.py` removes bottom footer strips and tiny far-left fragments from 5-gate searches
- `topology/analyze_symbol_with_proposals.py` removes original narrow right-column slice proposals when augmented recovery boxes already cover the same area
- `topology/analyze_symbol_with_proposals.py` mildly penalizes overly wide top-edge recovery boxes and dedupes overlapping augmented top-edge / right-column siblings after ranking
- `full_adder_tp.jpg` now reaches secondary search with `17` proposals and still resolves as `full_adder`
- `full_adder_using_half_adders.jpg` now reaches primary search with `16` proposals and still resolves in the primary path with `1` explored candidate

Recent rejected experiment:

- a proposer-side augmented-box dedupe was attempted in `topology/symbol_gate_proposer.py`
- it reduced raw augmented boxes too aggressively and broke the promoted `full_adder_tp.jpg` recovery path
- that change was reverted and should not be assumed to be safe

Recommended next target if runtime work continues:

- focus on `topology/symbol_gate_proposer.py`, not the candidate search
- the current bottleneck is proposal generation / recovery-box creation rather than combinatorial search
- any proposer change should be checked immediately against:
  - `python -m topology.analyze_symbol_with_proposals data/real_schematics/full_adder_tp.jpg --gate-counts 5 --proposal-limit 15 --label-top-k 3 --json`
  - `python -m topology.run_symbol_end_to_end_benchmark --json`
  - `python -m topology.run_symbol_end_to_end_benchmark --include-candidates --json`

Useful pickup command:

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.run_symbol_end_to_end_benchmark --include-candidates --json
```

## 5. Important Progress Already Made

### Phase 1: Isolated Gate Classification

Done:

- cleaned and reorganized the gate dataset under `data/`
- trained multiple classifier variants
- best useful checkpoint is `checkpoints_384_v3/best_model.pth`
- class set:
  - `AND`
  - `NAND`
  - `NOR`
  - `NOT`
  - `OR`
  - `XNOR`
  - `XOR`

Relevant files:

- `train.py`
- `train_improved.py`
- `predict.py`
- `model.py`
- `data_loader.py`
- `confusion_analysis.py`

Important finding:

- the classifier transfers well to **real isolated symbol images**
- it does **not** transfer directly to the current fixture-style gate-box crops

### Phase 2: YOLO Detection

Done:

- built YOLO training scripts and synthetic multi-object dataset generation
- trained multiple fixture-style runs
- fixed several detector-to-topology handoff issues

Important generator / handoff changes already in repo:

- class-agnostic NMS in the topology pipeline
- fixture-box refinement to shrink oversized YOLO boxes
- fixture generator now draws wires on top of gates
- fixture generator now writes fixture datasets as PNG
- weighted class sampling was added to synthetic fixture generation

Important runs already explored:

- `complex_circuits_v2_ft_fixture`
  - source of the current published fixture baseline
- `complex_circuits_v3_label_smoke`
  - over-emphasized text labels; regressed live behavior
- `complex_circuits_v4_fixture_png_smoke`
  - promising smoke run, but not stable enough after longer training
- `complex_circuits_v4_fixture_png_ft10`
  - regressed `full_adder_crossed`
- `complex_circuits_v5_weighted_smoke`
  - kept all three fixture circuits working
- `complex_circuits_v5_weighted_ft10`
  - regressed `half_subtractor`

Key lesson:

- validation mAP alone is not enough
- promotion must be gated by actual live end-to-end fixture classification

### Phase 3: Topology

Done:

- wire extraction
- crossing-aware wire separation
- gate terminal heuristics
- graph construction
- boolean evaluation
- classification for supported circuit signatures

Topology is currently the strongest part of the project.

Supported topology signatures:

- `half_adder`
- `half_subtractor`
- `full_adder`

Important conclusion:

- topology can rescue some detector class mistakes
- but it cannot rescue all of them
- the detector / classifier front end is still the main limitation

### Web Demo / Productization

Done:

- FastAPI local app
- upload flow
- built-in sample fixture buttons
- health endpoint
- smoke tests for web routes
- README and GitHub publication

Purpose of current app:

- portfolio/demo artifact
- not the final research outcome

## 5B. Recent Symbol-Crop Adaptation Work

Additional files now in the repo:

- `topology/build_schematic_crop_dataset.py`
  - builds `data/schematic_crops/` from the local synthetic symbol-style schematic dataset
- `topology/finetune_symbol_crop_classifier.py`
  - fine-tunes the gate classifier on schematic crops and evaluates against the crop benchmark
- `data/schematic_crops/README.md`
  - documents the adaptation dataset layout and build command

Current local adaptation dataset:

- `data/schematic_crops/`
- built from local synthetic symbol-style schematics
- current size: `1378` crop images

Important recent findings from actual runs:

- raw Tier-2a crop benchmark baseline: `0.25`
- edge-wire suppression only: `0.3125`
- head-only adaptation at `224x224`, `5` epochs, `512` train samples: `0.3125`
- larger frozen-head run regressed to `0.125`
- `layer4 + fc` partial-unfreeze run also regressed to `0.125` despite very strong adaptation-split metrics

Interpretation:

- adaptation-split accuracy on the synthetic crop dataset is not a trustworthy promotion criterion
- the crop benchmark is the real gatekeeper right now
- the synthetic crop path is useful for infrastructure and quick transfer checks, not as proof of real symbol-style readiness

## 5C. Split Protocol Correction

One important issue was discovered and fixed:

- the generic dataset split allowed crops from the same source schematic to land in different splits
- that inflated synthetic adaptation metrics

This is now corrected:

- crop filenames already encode the source schematic stem
- `topology/finetune_symbol_crop_classifier.py` now supports `--group-by-source-schematic`
- grouped split diagnostics are written into `split_info.json`

Current grouped split characteristics:

- `300` source groups total
- `217` train groups
- `39` val groups
- `44` test groups

Most important grouped-run result:

- grouped head-only run at `224x224`, `5` epochs, `512` train samples:
  - best val acc `0.2734375`
  - test acc `0.2421875`
  - crop benchmark `0.3125`

Interpretation:

- the grouped split is the correct baseline going forward
- the benchmark result stayed the same while the split metrics became harsher
- this confirms the benchmark was already the honest signal, while the old split metrics were inflated

## 6. Current Technical Conclusion

Do **not** continue spending large amounts of time trying to perfect the fixture-style detector as the main objective.

Reason:

- it was the right move to get a working end-to-end demo
- it proved the pipeline architecture
- but it is not the actual final visual domain

The next serious work should shift toward **symbol-style support**, while preserving the fixture-style app as the stable `v1` demo.

Additional updated conclusion:

- do **not** trust synthetic schematic-crop train/val/test accuracy alone
- do use source-schematic-grouped splits for future synthetic crop training
- do treat the Tier-2a crop benchmark as the promotion criterion until a real-symbol crop benchmark exists

## 7. Possible Paths From Here

## Path 1: Keep Improving Fixture-Style Only

Description:

- continue tuning the synthetic fixture detector
- add more fixture-style circuits
- improve the current app/demo within the same labeled-box domain

Pros:

- fastest path to a stronger short-term demo
- uses the existing topology stack with low risk
- easiest to test and maintain

Cons:

- moves away from the real long-term goal
- box-label detection is not the intended final domain
- returns are diminishing

When to choose:

- if the objective is portfolio polish only
- if demo stability matters more than research progress

Implementation:

- keep `models/fixture_demo_best.pt` as default until a new run clearly beats it
- continue using fixture regression tests as the promotion gate
- add more supported fixture circuits only if needed for the demo

Recommendation:

- not the main recommended path anymore

## Path 2: Mixed-Domain Single Detector

Description:

- train one detector to handle both fixture-style and symbol-style schematics

Pros:

- elegant if it works
- one model for both domains
- simpler deployment story later

Cons:

- high risk of negative transfer
- harder to debug
- fixture and symbol domains are visually very different
- likely to waste time before enough real schematic data exists

When to choose:

- only after a symbol-style benchmark set exists and is reasonably sized

Implementation:

- create mixed synthetic/real training corpora
- add explicit domain balancing
- validate separately on fixture and symbol-style benchmarks

Recommendation:

- not recommended yet

## Path 3: Separate Symbol-Style Detection Track

Description:

- keep the fixture-style demo unchanged
- create a separate symbol-style branch for the true goal
- use the isolated-gate classifier as part of that branch

Pros:

- lowest-risk way to move toward the real target
- preserves the working demo
- uses the original gate-only training work
- easier to reason about failures

Cons:

- temporarily maintains two parallel paths
- more project complexity

When to choose:

- now

Implementation:

- keep the current fixture demo as `v1`
- build a symbol-style benchmark
- evaluate detector proposals and isolated-gate classification separately
- only merge into the main pipeline once symbol-style crops are working

Recommendation:

- **this is the recommended path**

## Path 4: Detector + Second-Stage Gate Reclassifier

Description:

- detect candidate gate regions in a schematic
- crop each region
- classify the crop using the isolated-gate model
- use the classifier result to override or validate detector labels

Pros:

- makes the early gate-only training directly useful
- can reduce class confusion in full schematics
- architecturally matches the real goal well

Cons:

- only works if the detector gives usable crops
- needs a symbol-style crop benchmark first
- does not solve missed detections or bad boxes by itself

When to choose:

- once the symbol-style crop benchmark exists

Implementation:

- use `topology/gate_reclassifier.py`
- use `topology/evaluate_gate_reclassifier.py`
- first benchmark the classifier on:
  - isolated real gate images
  - then real gate crops from schematics
- only then integrate into the main pipeline

Recommendation:

- this should be the core mechanism for the symbol-style branch

## 8. Current Recommendation

The best plan is:

1. Keep the fixture-style end-to-end path as the stable demo baseline.
2. Stop treating fixture-style detector optimization as the main project objective.
3. Build the symbol-style branch around the isolated-gate classifier.
4. Add a real crop-based benchmark for symbol-style schematics.
5. Then integrate detector + reclassifier + topology.

## 9. What Was Just Added For The Symbol-Style Branch

These files were added specifically to start the symbol-style track:

- `topology/gate_reclassifier.py`
  - reusable wrapper for the isolated-gate checkpoint
- `topology/evaluate_gate_reclassifier.py`
  - benchmark harness for:
    - full-image symbol classification
    - crop-based reclassification on detected gates
- `topology/run_symbol_style_benchmark.py`
  - runner for the curated symbol benchmark
- `topology/run_symbol_crop_benchmark.py`
  - runner for manually labeled symbol-schematic gate crops
- `topology/build_tier2b_benchmark.py`
  - helper to propose detector boxes on real schematics and save review crops
- `benchmarks/symbol_gate_full_image_benchmark.json`
  - curated Tier-1 real-symbol benchmark
- `benchmarks/symbol_gate_crop_benchmark.json`
  - empty scaffold for the Tier-2 crop benchmark
- `benchmarks/symbol_gate_real_crop_benchmark.json`
  - real-schematic crop benchmark manifest in progress
- `data/real_schematics/`
  - externally sourced real schematic images plus crop-review artifacts
- `docs/SYMBOL_STYLE_EVAL.md`
  - documentation for this branch

Important current findings:

- `python -m topology.run_symbol_style_benchmark` is strong
- `python -m topology.evaluate_gate_reclassifier data/XOR/XOR42.png --full-image` works
- `python -m topology.evaluate_gate_reclassifier examples/schematics/full_adder_crossed.png --json` fails in the expected way, because fixture crops are not the correct domain for the classifier

Interpretation:

- the classifier is valid
- the missing piece is **symbol-style crops from multi-gate schematics**

Additional current real-benchmark progress:

- a first real benchmark manifest exists at `benchmarks/symbol_gate_real_crop_benchmark.json`
- it currently contains manually corrected entries for `data/real_schematics/xor_basic_gates.webp`
- a helper tool exists at `topology/build_tier2b_benchmark.py`
- that tool is useful for proposing candidate boxes and saving crops for review
- detector labels on real schematics are not trustworthy enough to use directly as benchmark labels

Interpretation of the new real-benchmark tooling:

- the tool is good enough for proposal generation
- benchmark labels still need manual review
- the highest-value work is no longer benchmark plumbing; it is proposal-search runtime and broader real-image reliability after the latest promoted case

## 10. Immediate Next Tasks

### Recommended Next Task

Harden the proposal-driven symbol-style beta path beyond `data/real_schematics/full_adder_tp.jpg`.

This is now the correct next task because:

- the real crop benchmark already exists and is stable at `55/55`
- the supported real-image symbol benchmark is already stable at `5/5`
- `full_adder_tp.jpg` now has crop coverage for the previously missing wide top `XOR`-style gate and lower `AND` gate
- `full_adder_tp.jpg` is now promoted, so the next work should target broader robustness rather than re-opening that case

Additional guidance from the current repo state:

- keep using the real crop benchmark and the supported symbol end-to-end benchmark as the promotion gates
- if synthetic crop experiments continue, use `--group-by-source-schematic`
- do not spend time redoing synthetic adaptation work unless it directly helps the current real symbol path
- do not disturb the fixture demo path while iterating on symbol beta

### Practical Implementation Focus

Focus the next changes on these areas only:

- proposal ranking and candidate selection on additional real symbol schematics while keeping `full_adder_tp.jpg` stable as a regression case
- topology consistency checks that reject fragment-driven wins even if they superficially match a supported circuit
- runtime and search breadth improvements for the proposal-driven beta path on supported and near-supported real images

Recommendation:

- do not add more benchmark plumbing unless a concrete symbol-style investigation is blocked on it

### Exact Next Actions

1. The benchmark is now at `55/55` (100%) accuracy and has crossed the 50+ sample target.
2. Keep avoiding simple line-drawing style schematics until domain adaptation is addressed.
3. The symbol-style integration work is now underway:
   - manual-box-assisted symbol path exists
   - proposal-driven beta path exists
   - supported real half-adder benchmark is now `3/3`
   - supported real-image symbol benchmark is now `5/5` after adding `data/real_schematics/full_adder_using_half_adders.jpg` and `data/real_schematics/full_adder_tp.jpg`
   - `data/real_schematics/full_adder_tp.jpg` now has explicit crop-benchmark coverage for its top `XOR`-style gate and lower `AND` gate, and those crops classify correctly
4. Recent symbol-beta hardening now in repo:
   - second-pass augmented proposal recovery exists in `topology/analyze_symbol_with_proposals.py`
   - fallback now preserves the augmented ranked pool instead of dropping back to the weaker raw pool
   - narrow OR-fragment suppression now keeps clipped `OR` slices from crowding out plausible `AND`/`XOR` candidates during search
   - generic proposal analysis now retries richer gate counts before returning an early smaller subcircuit match, so `full_adder_using_half_adders.jpg` no longer falls back to a 2-gate `half_adder` when analyzed without an explicit `gate_counts` override
   - guarded topology heuristics now exist for wide top-edge `XOR`/`XNOR`, large right-side multi-input `OR`/`NOR`, and very tall input-only bus components
5. Keep using `topology/build_tier2b_benchmark.py` only to propose boxes or save crops, then manually verify labels and boxes before appending future real samples.
6. Re-run `python -m topology.run_symbol_crop_benchmark --benchmark benchmarks/symbol_gate_real_crop_benchmark.json --json` after each benchmark update.
7. Treat the real crop benchmark and the supported real-image end-to-end symbol benchmark as the promotion gates for symbol-style changes.

Current focus to keep in mind:

- `full_adder_tp.jpg` is now a promoted end-to-end symbol case and should be treated as a regression target.
- The visible gate crops benchmark correctly, and the promoted proposal path now finds the supported 5-gate match for the image.
- `decoder_2x4_asic.png` now works in both the benchmark-box path and the live proposal path and classifies as `decoder_2to4`.
- The highest-value remaining work is improving runtime and robustness of the proposal-search beta path on additional real diagrams, especially keeping larger decoder-family candidates from collapsing into smaller supported subset matches.

What not to do next:

- do not wire the reclassifier into the current fixture-only path without a symbol-style entry point
- do not spend more time optimizing synthetic adaptation runs before the real benchmark is the main decision-maker
- do not accept auto-detected labels into the real benchmark without manual correction
- do not treat the current symbol beta as general-purpose support for arbitrary uploaded pictures yet

## 10.5 Next Session Message

Start from the current post-promotion symbol-beta state:

- guarded non-regressions are green:
  - real crop benchmark `55/55`
  - supported symbol end-to-end benchmark `6/6`
  - web app still exposes `Symbol Beta`
- the end-to-end case manifest lives at `benchmarks/symbol_end_to_end_cases.json`
- `python -m topology.run_symbol_end_to_end_benchmark --include-candidates --json` is the current candidate audit
- candidate sweep target is `16/16`:
  - `6` supported cases should pass
  - `10` candidate images should continue resolving to `unknown`
- `decoder_2x4_asic.png` is promoted and should stay green as `decoder_2to4`
- `decoder_3x8_asic.png` remains candidate-only and should stay `unknown`
- current major candidate-audit runtime hotspots are:
  - `decoder_3x8_asic.png`: about `23.3s`
  - `xnor_ansi.png`: about `7.6s`
  - `xnor_basic_gates_gfg.png`: about `6.9s`

The next task is no longer recovering `decoder_2x4_asic.png` or re-opening `full_adder_tp.jpg`. Both are already promoted. The current work is runtime and robustness of the proposal-search beta path while keeping larger candidate images from collapsing into smaller supported subset matches.

Pick up by focusing on this post-promotion state:

1. keep the supported `6/6` symbol benchmark green, especially `decoder_2x4_asic.png` and `full_adder_tp.jpg`
2. keep the `10` current candidate images resolving to `unknown` unless there is a deliberate promotion backed by stable live-path behavior
3. reduce candidate-audit runtime on the worst current hotspots, starting with `decoder_3x8_asic.png`, without creating false `decoder_2to4` subset matches
4. continue expanding supported symbol-style families cautiously, only promoting new cases after they stay stable in the live proposal path and the explicit end-to-end benchmark
5. avoid spending time on new synthetic adaptation work or fixture-only detector tuning unless it directly helps the real symbol path

Non-regression gates to rerun after changes:

- `python -m topology.run_symbol_crop_benchmark --benchmark benchmarks/symbol_gate_real_crop_benchmark.json --json`
- `python -m topology.run_symbol_proposal_benchmark --json`
- `python -m topology.run_symbol_end_to_end_benchmark --json`
- `python -m topology.run_symbol_end_to_end_benchmark --include-candidates --json`
- `python -m unittest tests.test_topology_pipeline tests.test_webapp`

## 11. Commands Another Agent Can Run Immediately

### Run the fixture-style live baseline

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.evaluate_live_yolo --json
```

### Run the isolated-gate symbol benchmark

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.run_symbol_style_benchmark --json
```

### Run full-image symbol classification on one image

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.evaluate_gate_reclassifier data/XOR/XOR42.png --full-image --json
```

### Compare detector labels vs gate classifier on schematic crops

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.evaluate_gate_reclassifier examples/schematics/full_adder_crossed.png --json
```

### Propose boxes and save review crops for a real schematic

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.build_tier2b_benchmark data/real_schematics/half_adder_tp.jpg --save-crops data/real_schematics/crops_review
```

### Run the current regression suites

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m unittest tests.test_dataset_generator tests.test_topology_pipeline tests.test_webapp
```

### Run the symbol-style proposal benchmark

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.run_symbol_proposal_benchmark --json
```

### Run manual-box-assisted symbol-style analysis

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.analyze_symbol_from_benchmark data/real_schematics/half_adder_vlabs.png --json
```

### Run proposal-driven symbol-style analysis

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.analyze_symbol_with_proposals data/real_schematics/half_adder_vlabs.png --json
```

### Inspect the current `full_adder_tp.jpg` proposal search directly

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.analyze_symbol_with_proposals data/real_schematics/full_adder_tp.jpg --gate-counts 5 --proposal-limit 15 --label-top-k 3 --json
```

### Inspect the manual-box-assisted benchmark-backed path for `full_adder_tp.jpg`

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

### Run the symbol promotion gates

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m topology.run_symbol_crop_benchmark --benchmark benchmarks/symbol_gate_real_crop_benchmark.json --json
python -m topology.run_symbol_end_to_end_benchmark --json
```

## 12. Files Another Agent Should Read First

Suggested reading order:

1. `docs/AGENT_HANDOFF_PLAN.md`
2. `PROJECT_STATUS.md`
3. `README.md`
4. `topology/pipeline.py`
5. `topology/gate_reclassifier.py`
6. `topology/evaluate_gate_reclassifier.py`
7. `topology/run_symbol_style_benchmark.py`
8. `docs/SYMBOL_STYLE_EVAL.md`

## 13. Bottom Line

The project has already proven the architecture with a fixture-style demo.

The correct next phase is:

- preserve the working demo
- stop over-investing in labeled-box detection
- reuse the original isolated-gate classifier
- continue hardening the proposal-driven symbol-style beta path that gets the project back toward its real goal

If an agent is resuming from this file, the default recommendation is:

**use `full_adder_tp.jpg` as a regression guard next, not as the open blocker; focus new work on broader symbol-style proposal reliability and runtime rather than more fixture-only detector tuning.**
