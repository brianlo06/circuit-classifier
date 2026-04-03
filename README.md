# Circuit Classifier

Circuit Classifier is a computer-vision and topology-analysis project for recognizing simple digital logic circuits from schematic images.

The current version is a working local demo for **clean fixture-style schematics**. It can detect gates, recover wiring, build a circuit graph, and classify supported circuits end to end.

## Current Demo Scope

Supported in the current demo:
- `half_adder`
- `half_subtractor`
- `full_adder`

Current limitation:
- real symbol-style schematics are **not** the main supported input domain yet

In other words, this repo currently ships a strong proof-of-concept for clean synthetic logic schematics, with symbol-style support planned for a later version.

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
- built-in sample fixture buttons
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
- detection + topology + classification in one pipeline
- local demo app for showing the system interactively
- reproducible synthetic fixtures for regression testing

## Current Limitations

- The detector still makes some gate-class mistakes on fixture images.
- The topology stage can sometimes recover the correct circuit label even when individual gate labels are wrong.
- Real symbol-style schematics from the `data/` collection are still largely out of scope for the current demo path.
- The web app is intended as a local demo, not a production deployment.

## Roadmap

Short term:
- improve fixture-style detector class precision
- add more fixture-style circuit families
- improve demo polish and documentation

Longer term:
- build a separate symbol-style detection track
- support real logic-gate drawings instead of labeled fixture boxes
- eventually merge both paths into a broader schematic reader

## Status

This repo should currently be read as:

**Version 1: a working logic-circuit reader for clean fixture-style schematics**

That is the shipped scope. Real symbol-style schematic support is a future extension, not the current promise.
