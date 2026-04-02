# Circuit Classifier Project Status

**Last Updated:** 2026-04-02

## Current State: End-to-End YOLO -> Topology Working On Multiple Fixture-Style Schematics

### Active Detection Artifacts
- **Published demo model:** `models/fixture_demo_best.pt`
- **Training run source:** `yolo_detection/runs/detect/complex_circuits_v2_ft_fixture/`
- **Checkpoints saved in training run:** `weights/last.pt`, `weights/best.pt`
- **Resume helper:** `python3 yolo_detection/resume_training.py`

### Phase 1: Gate Classification (COMPLETE)

Successfully trained a ResNet model to classify individual logic gate symbols.

**Best Model:** `checkpoints_384_v3/best_model.pth`
- **Accuracy:** 96.7% (11 errors out of 332 images)
- **Resolution:** 384x384
- **Classes:** AND, NAND, NOR, NOT, OR, XNOR, XOR

#### All Checkpoints

| Directory | Accuracy | Notes |
|-----------|----------|-------|
| `checkpoints_384_v3/` | 96.7% | **Current best** - cleaned dataset |
| `checkpoints_384_fixed/` | 96.7% | After first label fixes |
| `checkpoints_384/` | 96.1% | First 384x384 training |
| `checkpoints_diagnostic_latest/` | 93.48% | 224x224 baseline |

#### Dataset Cleaning Done
- Moved mislabeled images to correct classes (8 total)
- Removed problematic multi-gate diagrams (3 total)
- Final dataset: 332 images across 7 classes

### Phase 2: Object Detection (COMPLETE AS PROTOTYPE)

YOLOv8 project structure and first training run completed in `yolo_detection/`

**Files:**
- `yolo_detection/data.yaml` - Dataset config
- `yolo_detection/convert_to_yolo.py` - Conversion script
- `yolo_detection/train_yolo.py` - Training script
- `yolo_detection/README.md` - Documentation

**Dataset converted:** 265 train / 67 val images with auto-detected bounding boxes

New schematic-style synthetic dataset:
- `yolo_detection/datasets/complex_circuits/`
- 240 train / 60 val images
- 2 to 7 gates per image
- Average 4.48 gates per image
- Generated from real gate symbols with schematic-style wire routing and YOLO labels
- The retraining path is wired end-to-end and passed a 1-epoch smoke test on `datasets/complex_circuits_smoke/data.yaml`
- A real `yolov8m` training run on `datasets/complex_circuits/data.yaml` has been completed far enough to produce `weights/last.pt`, `weights/best.pt`, and validation metrics through at least epoch 21

**To train another run:**
```bash
pip install -r requirements.txt
cd yolo_detection
python3 train_yolo.py --epochs 100
```

### Phase 3: Circuit Topology (MVP COMPLETE FOR CLEAN FIXTURES)

Implemented scaffolding in `topology/` for:
1. Wire detection via classical CV
2. Gate terminal heuristics
3. Circuit graph data structures and boolean evaluation
4. Graph construction from wires + gates
5. Basic circuit classification for known topologies
6. End-to-end analysis CLI

Current progress:
- Added reproducible topology fixtures in `examples/schematics/` and matching detections in `examples/detections/`
- Added debug overlays in `examples/debug/` to inspect wire masks, terminals, and component matches
- The `half_adder` fixture now works end-to-end and classifies correctly
- The `full_adder` fixture now works end-to-end and classifies correctly
- Added harder synthetic fixtures: `half_adder_dense` and `full_adder_crossed`
- Added `unittest` regression coverage for the `half_adder` and `full_adder` fixture pipelines in `tests/test_topology_pipeline.py`
- Extended regression coverage to the harder synthetic fixtures as well
- Added `topology/evaluate_live_yolo.py` to measure the real YOLO -> topology handoff
- Wire extraction now uses a crossing-aware skeleton graph so orthogonal crossings can be separated from real junctions on clean schematics
- Terminal anchors are placed just outside gate boxes, which makes wire matching more robust after gate masking

Current handoff finding as of 2026-04-02:
- The paused fine-tuned weights in `complex_circuits_v2_ft_fixture` are now usable for fixture-style Phase 3 handoff
- On live YOLO inference, `examples/schematics/half_adder.png` detects 2 gates and classifies correctly as `half_adder`
- On live YOLO inference, `examples/schematics/half_subtractor.png` detects 3 gates and classifies correctly as `half_subtractor`
- On live YOLO inference, `examples/schematics/full_adder_crossed.png` detects 5 gates and classifies correctly as `full_adder`
- The live path now uses class-agnostic NMS plus a fixture-box refinement pass to shrink oversized YOLO boxes before wire masking
- On live YOLO inference, `data/XOR/XOR42.png` is still out-of-domain for the current fixture-style detector path
- That means the current end-to-end system works on fixture-style schematics, while symbol-style support remains a separate next problem

Phase 2 retraining and handoff work completed:
- Added `yolo_detection/generate_multi_object_dataset.py`
- Added a fixture-style rendering mode for `yolo_detection/datasets/complex_circuits/` to better match the topology fixtures
- Fine-tuned from `complex_circuits_v1/weights/best.pt` into `complex_circuits_v2_ft_fixture/`
- Added `topology/evaluate_live_yolo.py` so the YOLO -> topology handoff can be re-measured after retraining
- Added live fixture regression coverage in `tests/test_topology_pipeline.py`
- Added a third supported fixture-style circuit family: `half_subtractor`

What we tried:
- Recursive wire component grouping
- Endpoint/segment terminal matching
- Debug visualization overlays
- Better gate terminal placement
- Skeletonization of the wire mask
- Junction-aware splitting
- Skeleton path tracing
- Node clustering and edge/net consolidation
- Crossing-aware edge pairing at skeleton intersections
- Fixture routing cleanup around OR inputs to avoid masked-edge losses

Next plan:
1. Keep `complex_circuits_v2_ft_fixture` as the fixture-style baseline and extend regression coverage as needed
2. Improve detector class accuracy on fixture-style gates, especially reducing OR/AND confusion when topology still recovers the right circuit
3. Add more fixture-style circuit families beyond half-adder, half-subtractor, and full-adder
4. Start a separate symbol-style detection track for real gate-symbol schematics like `data/XOR/XOR42.png`
5. Decide whether symbol-style support should be a separate detector or a mixed-domain training setup
6. Mix in hand-labeled real schematics once they are available

## Key Files

```
circuit-classifier/
├── data/                          # Cleaned dataset (332 images)
├── checkpoints_384_v3/            # Best classifier model
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── history.json
│   └── results.json
├── confusion_analysis.py          # Evaluation script
├── train_improved.py              # Training script (384x384 support)
├── predict.py                     # Inference script
├── topology/                      # Phase 3 topology analysis
├── examples/                      # Phase 3 schematic fixtures + debug outputs
└── yolo_detection/                # Object detection setup
    ├── data.yaml
    ├── convert_to_yolo.py
    ├── train_yolo.py
    └── datasets/circuit_components/
```

## Resume Commands

**Run classifier on new image:**
```bash
python predict.py /path/to/image.png --checkpoint checkpoints_384_v3/best_model.pth
```

**Evaluate classifier:**
```bash
python confusion_analysis.py checkpoints_384_v3/best_model.pth
```

**Run topology analysis:**
```bash
python3 -m topology.main /path/to/schematic.png --json
```

**Generate fixture debug overlay:**
```bash
python3 -m topology.main \
  examples/schematics/half_adder.png \
  --detections-json examples/detections/half_adder.json \
  --save-debug-vis examples/debug/half_adder_debug.png \
  --json
```

**Start YOLO training:**
```bash
cd yolo_detection
pip install -r ../requirements.txt
python3 train_yolo.py
```
