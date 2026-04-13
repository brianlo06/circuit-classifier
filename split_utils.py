"""
Utilities for duplicate-aware dataset splitting.
"""

from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence, Tuple

from check_duplicates import find_duplicates


class UnionFind:
    """Small union-find for grouping duplicate samples."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_grouped_split(
    samples: Sequence[Tuple[Path, int]],
    class_names: Sequence[str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    duplicate_threshold: int = 10,
) -> Tuple[Dict[str, List[int]], Dict]:
    """
    Split samples while keeping perceptual duplicate groups together.

    Returns:
        split_indices: {"train": [...], "val": [...], "test": [...]}
        diagnostics: serializable split summary
    """
    if train_split <= 0 or val_split < 0 or train_split + val_split >= 1:
        raise ValueError("Invalid train/val split fractions")

    num_samples = len(samples)
    uf = UnionFind(num_samples)
    path_to_index = {str(path): idx for idx, (path, _) in enumerate(samples)}

    duplicate_groups = find_duplicates(
        str(Path(samples[0][0]).parents[1]),
        similarity_threshold=duplicate_threshold,
    ) if samples else []

    for group in duplicate_groups:
        indices = [path_to_index[str(path)] for path in group if str(path) in path_to_index]
        for idx in indices[1:]:
            uf.union(indices[0], idx)

    grouped_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(num_samples):
        grouped_indices[uf.find(idx)].append(idx)

    total_size = num_samples
    split_names = ["train", "val", "test"]
    split_targets = {
        "train": int(round(total_size * train_split)),
        "val": int(round(total_size * val_split)),
    }
    split_targets["test"] = total_size - split_targets["train"] - split_targets["val"]

    total_class_counts = Counter(label for _, label in samples)
    class_targets = {
        split: {
            class_idx: total_class_counts[class_idx] * split_targets[split] / max(total_size, 1)
            for class_idx in total_class_counts
        }
        for split in split_names
    }

    groups = []
    for root, indices in grouped_indices.items():
        label_counts = Counter(samples[idx][1] for idx in indices)
        groups.append({
            "root": root,
            "indices": sorted(indices),
            "size": len(indices),
            "label_counts": label_counts,
        })

    rng = Random(seed)
    rng.shuffle(groups)
    groups.sort(
        key=lambda g: (
            -g["size"],
            -max(g["label_counts"].values()),
            tuple(sorted(g["label_counts"].items())),
        )
    )

    assigned = {name: [] for name in split_names}
    split_sizes = {name: 0 for name in split_names}
    split_class_counts = {name: Counter() for name in split_names}

    def global_score(candidate_split: str = None, candidate_group: Dict = None) -> float:
        score = 0.0
        for split_name in split_names:
            size = split_sizes[split_name]
            counts = split_class_counts[split_name]
            if split_name == candidate_split and candidate_group is not None:
                size += candidate_group["size"]
                counts = counts + candidate_group["label_counts"]

            score += abs(size - split_targets[split_name]) * 2.0

            for class_idx, total_count in total_class_counts.items():
                score += abs(counts[class_idx] - class_targets[split_name][class_idx])

            overshoot = max(0, size - split_targets[split_name])
            score += overshoot * 3.0

        return score

    for group in groups:
        best_split = min(split_names, key=lambda name: global_score(name, group))
        assigned[best_split].extend(group["indices"])
        split_sizes[best_split] += group["size"]
        split_class_counts[best_split].update(group["label_counts"])

    split_indices = {name: sorted(indices) for name, indices in assigned.items()}
    index_to_split = {}
    for split_name, indices in split_indices.items():
        for idx in indices:
            index_to_split[idx] = split_name

    duplicate_group_summaries = []
    cross_split_duplicate_groups = 0
    for group in duplicate_groups:
        members = []
        split_presence = set()
        for path in group:
            idx = path_to_index.get(str(path))
            if idx is None:
                continue
            split_name = index_to_split[idx]
            split_presence.add(split_name)
            members.append({
                "path": str(path),
                "class_name": class_names[samples[idx][1]],
                "split": split_name,
            })
        if len(split_presence) > 1:
            cross_split_duplicate_groups += 1
        duplicate_group_summaries.append(members)

    diagnostics = {
        "num_samples": total_size,
        "num_duplicate_groups": len(duplicate_groups),
        "num_grouped_components": len(groups),
        "cross_split_duplicate_groups": cross_split_duplicate_groups,
        "duplicate_groups": duplicate_group_summaries,
        "split_sizes": split_sizes,
        "split_class_distribution": {
            split: {class_names[idx]: count for idx, count in sorted(counts.items())}
            for split, counts in split_class_counts.items()
        },
        "target_split_sizes": split_targets,
    }

    return split_indices, diagnostics


def build_grouped_folds(
    samples: Sequence[Tuple[Path, int]],
    class_names: Sequence[str],
    num_folds: int = 5,
    seed: int = 42,
    duplicate_threshold: int = 10,
) -> Tuple[List[List[int]], Dict]:
    """
    Build duplicate-aware cross-validation folds.

    Returns:
        folds: list of sample-index lists, one per fold
        diagnostics: serializable fold summary
    """
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    num_samples = len(samples)
    uf = UnionFind(num_samples)
    path_to_index = {str(path): idx for idx, (path, _) in enumerate(samples)}

    duplicate_groups = find_duplicates(
        str(Path(samples[0][0]).parents[1]),
        similarity_threshold=duplicate_threshold,
    ) if samples else []

    for group in duplicate_groups:
        indices = [path_to_index[str(path)] for path in group if str(path) in path_to_index]
        for idx in indices[1:]:
            uf.union(indices[0], idx)

    grouped_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(num_samples):
        grouped_indices[uf.find(idx)].append(idx)

    total_class_counts = Counter(label for _, label in samples)
    target_fold_size = num_samples / num_folds
    class_targets = {
        class_idx: total_class_counts[class_idx] / num_folds
        for class_idx in total_class_counts
    }

    groups = []
    for root, indices in grouped_indices.items():
        label_counts = Counter(samples[idx][1] for idx in indices)
        groups.append({
            "root": root,
            "indices": sorted(indices),
            "size": len(indices),
            "label_counts": label_counts,
        })

    rng = Random(seed)
    rng.shuffle(groups)
    groups.sort(
        key=lambda g: (
            -g["size"],
            -max(g["label_counts"].values()),
            tuple(sorted(g["label_counts"].items())),
        )
    )

    folds: List[List[int]] = [[] for _ in range(num_folds)]
    fold_sizes = [0] * num_folds
    fold_class_counts = [Counter() for _ in range(num_folds)]

    def score_fold(fold_idx: int, group: Dict):
        size_after = fold_sizes[fold_idx] + group["size"]
        class_error = 0.0
        class_overshoot = 0.0
        for class_idx, total_count in total_class_counts.items():
            after = fold_class_counts[fold_idx][class_idx] + group["label_counts"].get(class_idx, 0)
            target = class_targets[class_idx]
            class_error += abs(after - target)
            class_overshoot += max(0.0, after - target)

        size_error = abs(size_after - target_fold_size)
        size_overshoot = max(0.0, size_after - target_fold_size)
        return (
            class_overshoot,
            class_error,
            size_overshoot,
            size_error,
            fold_sizes[fold_idx],
            fold_idx,
        )

    for group_idx, group in enumerate(groups):
        remaining_groups = len(groups) - group_idx
        empty_folds = [idx for idx in range(num_folds) if fold_sizes[idx] == 0]
        if empty_folds and remaining_groups == len(empty_folds):
            best_fold = empty_folds[0]
        else:
            best_fold = min(range(num_folds), key=lambda idx: score_fold(idx, group))
        folds[best_fold].extend(group["indices"])
        fold_sizes[best_fold] += group["size"]
        fold_class_counts[best_fold].update(group["label_counts"])

    folds = [sorted(indices) for indices in folds]
    index_to_fold = {}
    for fold_idx, indices in enumerate(folds):
        for idx in indices:
            index_to_fold[idx] = fold_idx

    duplicate_group_summaries = []
    cross_fold_duplicate_groups = 0
    for group in duplicate_groups:
        members = []
        fold_presence = set()
        for path in group:
            idx = path_to_index.get(str(path))
            if idx is None:
                continue
            fold_idx = index_to_fold[idx]
            fold_presence.add(fold_idx)
            members.append({
                "path": str(path),
                "class_name": class_names[samples[idx][1]],
                "fold": fold_idx,
            })
        if len(fold_presence) > 1:
            cross_fold_duplicate_groups += 1
        duplicate_group_summaries.append(members)

    diagnostics = {
        "num_samples": num_samples,
        "num_folds": num_folds,
        "num_duplicate_groups": len(duplicate_groups),
        "num_grouped_components": len(groups),
        "cross_fold_duplicate_groups": cross_fold_duplicate_groups,
        "duplicate_groups": duplicate_group_summaries,
        "fold_sizes": fold_sizes,
        "fold_class_distribution": [
            {class_names[idx]: count for idx, count in sorted(counts.items())}
            for counts in fold_class_counts
        ],
    }

    return folds, diagnostics
