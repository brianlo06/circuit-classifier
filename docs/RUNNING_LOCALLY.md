# Running Locally With `uv`

This project now has a local `uv` virtual environment at:

```text
/Users/brianlo/circuit-classifier/.venv
```

These steps assume you use two Terminal windows:

- Terminal 1: project setup and optional tests
- Terminal 2: run the web app

## Terminal 1: Setup

From the project root:

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
```

If you ever need to recreate the environment:

```bash
cd /Users/brianlo/circuit-classifier
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Optional sanity checks:

```bash
python -m unittest tests.test_topology_pipeline tests.test_webapp tests.test_dataset_generator
python -m topology.evaluate_live_yolo --json
```

## Terminal 2: Run The Web App

Open a separate Terminal window and run:

```bash
cd /Users/brianlo/circuit-classifier
source .venv/bin/activate
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Built-In Demo Samples

You do not need your own images to try the app.

Use the built-in sample buttons for:

- `half_adder`
- `half_subtractor`
- `full_adder`

## Stop The App

In the terminal running Uvicorn, press:

```text
Ctrl+C
```

## Notes

- The current stable demo is for clean fixture-style schematics.
- The web app now exposes a `Symbol Beta` mode for clean symbol-style gate drawings.
- The symbol-style beta path is still experimental and currently much narrower than the fixture demo path.
- The default published model is `models/fixture_demo_best.pt`.
