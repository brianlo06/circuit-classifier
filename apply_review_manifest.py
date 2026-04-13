"""
Apply duplicate-audit review decisions from a CSV manifest.
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


VALID_ACTIONS = {"pending", "keep", "quarantine", "move"}


def unique_destination(path: Path) -> Path:
    """Avoid overwriting an existing file."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def apply_manifest(
    manifest_path: str,
    data_dir: str = "data",
    quarantine_dir: str = "quarantine/duplicate_review",
):
    manifest = Path(manifest_path)
    data_root = Path(data_dir).resolve()
    quarantine_root = Path(quarantine_dir).resolve()
    quarantine_root.mkdir(parents=True, exist_ok=True)

    summary = Counter()
    actions_taken = []

    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        action = row["action"].strip().lower()
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{row['action']}' for {row['path']}")
        if action in {"pending", "keep"}:
            summary[action] += 1
            continue

        source = Path(row["path"]).resolve()
        if not source.exists():
            summary["missing"] += 1
            continue

        if action == "quarantine":
            rel = source.relative_to(data_root)
            destination = unique_destination(quarantine_root / rel)
            destination.parent.mkdir(parents=True, exist_ok=True)
        elif action == "move":
            target_class = row["target_class"].strip()
            if not target_class:
                raise ValueError(f"Missing target_class for move action on {row['path']}")
            destination = unique_destination(data_root / target_class / source.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            continue

        shutil.move(str(source), str(destination))
        actions_taken.append({
            "source": str(source),
            "destination": str(destination),
            "action": action,
        })
        summary[action] += 1

    print("Review actions applied")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    if actions_taken:
        print(f"  changed_files: {len(actions_taken)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply duplicate review manifest")
    parser.add_argument("--manifest", type=str, default="audit/duplicate_audit/review_manifest.csv")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--quarantine-dir", type=str, default="quarantine/duplicate_review")
    args = parser.parse_args()

    apply_manifest(
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        quarantine_dir=args.quarantine_dir,
    )
