# Gate Identifier Split Plan

This document describes how to extract the gate-vision portion of this project into a separate GitHub repository later without destabilizing the current `circuit-classifier` demo.

Recommended repo name:
- `logic-gate-identifier`

## Goal

Separate the project into:

1. `logic-gate-identifier`
   Vision-only project for gate classification and multi-object gate detection.

2. `circuit-classifier`
   Circuit-topology, graph reasoning, and end-to-end schematic classification project.

## Move To `logic-gate-identifier`

These files/folders are primarily gate-vision work:

- `predict.py`
- `train.py`
- `train_improved.py`
- `cross_validate.py`
- `confusion_analysis.py`
- `data_loader.py`
- `model.py`
- `training_common.py`
- `split_utils.py`
- `validate_data.py`
- `check_duplicates.py`
- `duplicate_audit.py`
- `augment_images.py`
- `generate_gates.py`
- `yolo_detection/`

## Keep In `circuit-classifier`

These are circuit-level reasoning and app files:

- `topology/`
- `webapp/`
- `tests/test_topology_pipeline.py`
- `tests/test_webapp.py`
- `examples/`
- `models/fixture_demo_best.pt`
- `README.md`
- `PROJECT_STATUS.md`

## Optional In Either Repo

These depend on how much history you want to expose:

- `requirements.txt`
  Could be split into repo-specific requirements later.

- `data/`
  Better left out of the public demo repos unless you intentionally want to publish the raw symbol collection.

- old checkpoints and training runs
  Better excluded from the public repos unless a specific stripped demo checkpoint is needed.

## Recommended Extraction Order

1. Leave `circuit-classifier` stable first.
2. Create a new repo named `logic-gate-identifier`.
3. Copy the gate-vision files into the new repo.
4. Add one stripped demo checkpoint if needed.
5. Write a detection-focused README.
6. Add a small CLI or web demo for gate-only detection.
7. Link the two repos to each other in their READMEs.

## Why Split

Benefits:

- cleaner portfolio story
- smaller repos with clearer scope
- easier README and demo messaging
- easier reuse of the vision component later

Tradeoff:

- some duplicated setup if both repos remain active

## Current Recommendation

Do not split immediately if it slows down shipping the current demo.

The safer path is:

1. keep `circuit-classifier` stable
2. ship it publicly
3. extract `logic-gate-identifier` afterward as a second portfolio project
