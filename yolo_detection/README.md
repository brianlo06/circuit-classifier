# Circuit Component Detection with YOLOv8

Object detection model to find logic gates in schematic images.

## Setup

```bash
pip install ultralytics
```

## Dataset Conversion

Convert the classification dataset to YOLO format:

```bash
python convert_to_yolo.py --visualize
```

This will:
- Convert images from `../data/` to YOLO format
- Auto-detect bounding boxes around gate symbols
- Split into train/val sets
- Generate sample visualizations

## Multi-Object Synthetic Dataset

Generate a larger schematic-style detection dataset that contains multiple gates per image:

```bash
python3 generate_multi_object_dataset.py \
  --output datasets/complex_circuits \
  --train-count 240 \
  --val-count 60 \
  --gate-style fixture
```

This produces:
- `datasets/complex_circuits/images/{train,val}/`
- `datasets/complex_circuits/labels/{train,val}/`
- `datasets/complex_circuits/data.yaml`
- `datasets/complex_circuits/manifest.json`

The generated images use real gate symbols from `../data/`, place 2-7 gates per canvas, add schematic-style wires, and write one YOLO box per gate.

The default recommended mode for the current topology handoff is `--gate-style fixture`, which renders labeled gate boxes and orthogonal wiring closer to the synthetic fixtures in `examples/schematics/`. The fixture renderer intentionally uses large, high-contrast gate labels because the current detector relies heavily on text inside the gate box to separate classes such as `AND` vs `OR`. Use `--gate-style symbol` if you specifically want cropped single-symbol assets pasted into the canvases.

## Training

```bash
# Quick training with small model
python3 train_yolo.py --epochs 100

# Better accuracy with medium model
python3 train_yolo.py --model yolov8m.pt --epochs 150

# Full training with large model
python3 train_yolo.py --model yolov8l.pt --epochs 200 --batch 8
```

For a new multi-object schematic dataset, keep the same YOLO directory layout and point training at its `data.yaml`:

```bash
python3 train_yolo.py \
  --data datasets/complex_circuits/data.yaml \
  --model yolov8m.pt \
  --epochs 200 \
  --imgsz 1024 \
  --batch 8 \
  --name complex_circuits_v1
```

To fine-tune from an earlier run:

```bash
python3 train_yolo.py \
  --data datasets/complex_circuits/data.yaml \
  --weights runs/detect/complex_circuits_v1/weights/best.pt \
  --epochs 200 \
  --name complex_circuits_ft_v1
```

To resume an interrupted run:

```bash
python3 resume_training.py
python3 resume_training.py --checkpoint runs/detect/complex_circuits_v1/weights/last.pt --data datasets/complex_circuits/data.yaml
```

### Model Options
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (good balance)
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (most accurate, slowest)

## Inference

After training, run detection on new images:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/circuit_components/weights/best.pt")

# Predict
results = model.predict("path/to/schematic.png", save=True)

# Results include bounding boxes, class names, confidences
for r in results:
    for box in r.boxes:
        print(f"Found {r.names[int(box.cls[0])]} at {box.xyxy[0]}")
```

## Project Structure

```
yolo_detection/
├── data.yaml                 # Dataset config
├── convert_to_yolo.py        # Conversion script
├── generate_multi_object_dataset.py
├── train_yolo.py             # Training script
├── datasets/
│   └── circuit_components/
│       ├── images/
│       │   ├── train/        # Training images
│       │   └── val/          # Validation images
│       ├── labels/
│       │   ├── train/        # Training labels (YOLO format)
│       │   └── val/          # Validation labels
│       └── visualizations/   # Sample visualizations
└── runs/                     # Training outputs
```

## Next Steps

1. **Collect multi-component schematics** - Images with multiple gates, wires, and labels
2. **Annotate every object** - Each gate or component needs its own box
3. **Add more component types** - Resistors, capacitors, flip-flops, input/output pins
4. **Separate detection from reasoning** - YOLO finds components; a later graph step analyzes the circuit
