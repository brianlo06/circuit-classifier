"""
Evaluate the proposal -> reclassifier -> topology beta path on supported real symbol-style schematics.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import List

from .analyze_symbol_with_proposals import analyze_with_proposals


DEFAULT_CASES_PATH = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "symbol_end_to_end_cases.json"
)


def load_cases(cases_path: Path, statuses: List[str]) -> List[dict]:
    payload = json.loads(cases_path.read_text())
    allowed_statuses = set(statuses)
    return [item for item in payload if item.get("status", "supported") in allowed_statuses]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the symbol-style proposal beta path on supported real schematics"
    )
    parser.add_argument("--cases", type=str, default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--proposal-limit", type=int, default=6)
    parser.add_argument("--label-top-k", type=int, default=2)
    parser.add_argument(
        "--status",
        action="append",
        choices=["supported", "candidate"],
        help="Case status to include. Defaults to supported only.",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Include candidate real images alongside supported promotion-gate cases.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent.parent
    statuses = list(args.status or [])
    if not statuses:
        statuses = ["supported"]
        if args.include_candidates:
            statuses.append("candidate")
    cases = load_cases(Path(args.cases), statuses)
    results = []
    correct = 0
    total_elapsed_seconds = 0.0

    for case in cases:
        start = perf_counter()
        error = None
        try:
            search = analyze_with_proposals(
                image_path=root / case["image"],
                proposal_limit=case.get("proposal_limit", args.proposal_limit),
                label_top_k=case.get("label_top_k", args.label_top_k),
                gate_counts=case.get("gate_counts", [2]),
                topology_bbox_expand_ratio=case.get("topology_bbox_expand_ratio", 0.10),
            )
            predicted = search.result.classification.label
            confidence = search.result.classification.confidence
            selected_gate_ids = search.selected_gate_ids
            explored_candidates = search.explored_candidates
            warnings = search.result.warnings
        except ValueError as exc:
            error = str(exc)
            predicted = "unknown"
            confidence = 0.0
            selected_gate_ids = []
            explored_candidates = 0
            warnings = [error]
        elapsed_seconds = perf_counter() - start
        total_elapsed_seconds += elapsed_seconds
        if predicted == case["expected"]:
            correct += 1
        results.append(
            {
                "image": case["image"],
                "expected": case["expected"],
                "status": case.get("status", "supported"),
                "predicted": predicted,
                "confidence": confidence,
                "selected_gate_ids": selected_gate_ids,
                "explored_candidates": explored_candidates,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "warnings": warnings,
                "error": error,
            }
        )

    payload = {
        "cases_path": str(Path(args.cases)),
        "statuses": statuses,
        "case_count": len(cases),
        "correct": correct,
        "accuracy": correct / max(len(cases), 1),
        "total_elapsed_seconds": round(total_elapsed_seconds, 3),
        "average_elapsed_seconds": round(total_elapsed_seconds / max(len(cases), 1), 3),
        "max_elapsed_seconds": round(max((item["elapsed_seconds"] for item in results), default=0.0), 3),
        "total_explored_candidates": sum(item["explored_candidates"] for item in results),
        "hotspots": sorted(
            [
                {
                    "image": item["image"],
                    "elapsed_seconds": item["elapsed_seconds"],
                    "explored_candidates": item["explored_candidates"],
                    "predicted": item["predicted"],
                    "expected": item["expected"],
                }
                for item in results
            ],
            key=lambda item: (item["elapsed_seconds"], item["explored_candidates"]),
            reverse=True,
        ),
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Cases: {payload['case_count']}")
    print(f"Correct: {payload['correct']}")
    print(f"Accuracy: {payload['accuracy']:.3f}")
    print(f"Total elapsed: {payload['total_elapsed_seconds']:.3f}s")
    print(f"Average elapsed: {payload['average_elapsed_seconds']:.3f}s")
    print(f"Max elapsed: {payload['max_elapsed_seconds']:.3f}s")
    print(f"Total explored candidates: {payload['total_explored_candidates']}")
    print("Hotspots:")
    for hotspot in payload["hotspots"][:3]:
        print(
            f"- {hotspot['image']}: {hotspot['elapsed_seconds']:.3f}s, "
            f"explored {hotspot['explored_candidates']}, "
            f"predicted {hotspot['predicted']}"
        )
    for result in results:
        marker = "OK" if result["predicted"] == result["expected"] else "FAIL"
        print(
            f"- [{marker}] {result['image']}: expected {result['expected']}, "
            f"status {result['status']}, "
            f"predicted {result['predicted']} ({result['confidence']:.2f}), "
            f"{result['elapsed_seconds']:.3f}s, explored {result['explored_candidates']}"
        )


if __name__ == "__main__":
    main()
