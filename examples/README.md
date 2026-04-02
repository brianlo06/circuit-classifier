# Example Topology Fixtures

Generated schematic fixtures for Phase 3 live under:

- `examples/schematics/`
- `examples/detections/`
- `examples/debug/`

Current generated fixtures:

- `half_adder.png`: clean baseline half-adder
- `half_adder_dense.png`: tighter half-adder routing and spacing
- `half_subtractor.png`: clean half-subtractor with XOR difference and borrow logic
- `full_adder.png`: clean baseline full-adder
- `full_adder_crossed.png`: compact full-adder with deliberate orthogonal crossings

Regenerate them with:

```bash
python3 -m topology.generate_example_schematics
```

Run the topology pipeline against a fixture without YOLO:

```bash
python3 -m topology.main \
  examples/schematics/half_adder.png \
  --detections-json examples/detections/half_adder.json \
  --json
```

Generate a debug overlay that shows wire components, terminals, and component-to-terminal matches:

```bash
python3 -m topology.main \
  examples/schematics/half_adder.png \
  --detections-json examples/detections/half_adder.json \
  --save-debug-vis examples/debug/half_adder_debug.png \
  --json
```

Run the harder crossed full-adder fixture:

```bash
python3 -m topology.main \
  examples/schematics/full_adder_crossed.png \
  --detections-json examples/detections/full_adder_crossed.json \
  --save-debug-vis examples/debug/full_adder_crossed_debug.png \
  --json
```

Measure the live YOLO -> topology handoff without fixture detections:

```bash
python3 -m topology.evaluate_live_yolo --json
```

At the current project state, the default `complex_circuits_v2_ft_fixture` YOLO weights should classify the clean fixture-style schematics successfully. Symbol-style images are still out of scope for the current demo path.
