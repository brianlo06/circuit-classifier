# Circuit Classifier

Circuit Classifier is a computer-vision and topology-analysis project for recognizing simple digital logic circuits from schematic images.

The current version is a working local demo for **clean fixture-style schematics** plus an experimental **symbol-style beta** path for clean real gate drawings.

## Current Demo Scope

Supported in the current demo:
- `half_adder`
- `half_subtractor`
- `full_adder`

Available as a beta path:
- a bounded proposal-driven symbol-style analyzer with `6` currently supported real cases:
- `half_adder_vlabs.png`
- `half_adder_gfg.jpg`
- `half_adder_tp.jpg`
- `full_adder_using_half_adders.jpg`
- `full_adder_tp.jpg`
- `decoder_2x4_asic.png`

Current limitation:
- real symbol-style schematics are still a narrow beta path with explicit benchmark-backed support, not a general input domain

In other words, this repo currently ships a stable fixture-style demo plus a guarded symbol-style beta for a small supported set of real schematics.

## How It Works

The pipeline is split into four stages:

1. **Gate detection**
   YOLO detects logic gates in a schematic image.
2. **Wire extraction**
   Classical CV extracts wire-like structure outside gate boxes.
3. **Graph construction**
   Detected gates and wires are converted into a directed circuit graph.
4. **Circuit classification**
   The recovered graph is evaluated and matched against known circuit signatures.

That means the project is not just doing object detection. It is trying to infer the actual circuit function from the schematic structure.

## Web Demo

A local FastAPI demo is included.

Install dependencies:

```bash
cd /Users/brianlo/circuit-classifier
python3 -m pip install -r requirements.txt
```

Start the app:

```bash
python3 -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The app supports:
- direct image upload
- a `Fixture Demo` vs `Symbol Beta` mode toggle
- built-in sample fixture buttons
- built-in supported symbol beta samples
- classification output
- detected gate listing
- overlay visualization
- debug visualization
- truth-table display when available

## Deployment Prep

The repo includes a simple `Procfile` for lightweight hosting platforms:

```text
web: python3 -m uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8000}
```

That means the project is already set up for basic platform-style deployment later.

If you want to deploy it, the minimum runtime expectation is:
- Python 3.9+
- enough disk space for `models/fixture_demo_best.pt`
- enough memory/CPU for local inference

The current app should still be treated as a demo deployment, not a production service.

## Example Inputs

Generated fixture schematics live in:
- `examples/schematics/`

Useful sample images:
- `examples/schematics/half_adder.png`
- `examples/schematics/half_subtractor.png`
- `examples/schematics/full_adder_crossed.png`

The web app can analyze these directly through the built-in sample buttons.

## CLI Usage

Run the live YOLO -> topology path on the default sample set:

```bash
python3 -m topology.evaluate_live_yolo --json
```

Analyze a single image directly:

```bash
python3 -m topology.main /path/to/image.png --json
```

Analyze a real symbol-style benchmark image with manual benchmark boxes:

```bash
python3 -m topology.analyze_symbol_from_benchmark data/real_schematics/half_adder_vlabs.png --json
```

Analyze a real symbol-style image with heuristic proposals:

```bash
python3 -m topology.analyze_symbol_with_proposals data/real_schematics/half_adder_vlabs.png --json
```

Run a fixture with ground-truth detections instead of YOLO:

```bash
python3 -m topology.main \
  examples/schematics/half_adder.png \
  --detections-json examples/detections/half_adder.json \
  --json
```

Generate the example fixtures:

```bash
python3 -m topology.generate_example_schematics
```

## Tests

Run the current regression suite:

```bash
python3 -m unittest tests.test_topology_pipeline
```

The test suite currently covers:
- fixture-based topology correctness
- default model configuration
- live YOLO fixture handoff
- fixture-box refinement behavior

## Model Baseline

The current default end-to-end demo model is:

```text
models/fixture_demo_best.pt
```

This is a stripped GitHub-safe copy of the best working fixture-style checkpoint. The heavier training-run artifacts are local development outputs, not the published demo baseline.

## Project Structure

Key directories:
- `topology/` — circuit graph, wire detection, classifier, visualization, CLI
- `yolo_detection/` — dataset generation, YOLO training, detection runs
- `examples/` — generated fixture schematics, detections, debug outputs
- `tests/` — regression tests for the current pipeline
- `webapp/` — local FastAPI demo app

## What This Project Does Well Right Now

- end-to-end analysis on clean fixture-style logic schematics
- bounded symbol-style beta analysis on a small supported real-schematic set
- detection + topology + classification in one pipeline
- local demo app for showing the system interactively
- reproducible synthetic fixtures for regression testing

## Current Limitations

- The detector still makes some gate-class mistakes on fixture images.
- The topology stage can sometimes recover the correct circuit label even when individual gate labels are wrong.
- Most real symbol-style schematics from the `data/` collection are still out of scope; unsupported images should be expected to return `unknown`.
- The web app is intended as a local demo, not a production deployment.

## Roadmap

Short term:
- preserve the current fixture baseline
- harden Beta docs and presentation around the current supported symbol scope
- expand supported symbol-style cases cautiously only when benchmark-backed

Longer term:
- expand the symbol-style beta family by family
- support more real logic-gate drawings within an explicitly bounded scope
- avoid claiming arbitrary schematic understanding outside the benchmarked domain

## Status

This repo should currently be read as:

**Version 1: a working logic-circuit reader for clean fixture-style schematics, plus a narrow symbol-style beta**

That remains the shipped scope. The fixture path is the stable default; the symbol path is real, but still intentionally narrow and benchmark-gated.
