"""
Shared training helpers.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from data_loader import CircuitDataset, get_transforms
from split_utils import build_grouped_folds, build_grouped_split


def prepare_datasets(
    data_dir: str,
    image_size: int,
    train_split: float,
    val_split: float,
    seed: int,
    batch_size: int,
    duplicate_threshold: int = 10,
    drop_last_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict]:
    """Build duplicate-aware train/val/test loaders."""
    target_size = (image_size, image_size)
    train_transform = get_transforms(target_size, augment=True)
    eval_transform = get_transforms(target_size, augment=False)

    full_dataset = CircuitDataset(data_dir, transform=eval_transform, target_size=target_size)
    split_indices, split_info = build_grouped_split(
        full_dataset.samples,
        full_dataset.classes,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        duplicate_threshold=duplicate_threshold,
    )

    train_dataset = CircuitDataset(data_dir, transform=train_transform, target_size=target_size)
    val_dataset = CircuitDataset(data_dir, transform=eval_transform, target_size=target_size)
    test_dataset = CircuitDataset(data_dir, transform=eval_transform, target_size=target_size)

    train_subset = Subset(train_dataset, split_indices["train"])
    val_subset = Subset(val_dataset, split_indices["val"])
    test_subset = Subset(test_dataset, split_indices["test"])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    split_info["class_names"] = full_dataset.classes
    split_info["class_distribution"] = full_dataset.get_class_distribution()

    return train_loader, val_loader, test_loader, full_dataset.classes, split_info


def create_loaders_from_indices(
    data_dir: str,
    image_size: int,
    batch_size: int,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int] = None,
    drop_last_train: bool = False,
):
    """Create loaders from explicit index lists."""
    target_size = (image_size, image_size)
    train_transform = get_transforms(target_size, augment=True)
    eval_transform = get_transforms(target_size, augment=False)

    train_dataset = CircuitDataset(data_dir, transform=train_transform, target_size=target_size)
    val_dataset = CircuitDataset(data_dir, transform=eval_transform, target_size=target_size)
    test_dataset = CircuitDataset(data_dir, transform=eval_transform, target_size=target_size)

    train_loader = DataLoader(
        Subset(train_dataset, sorted(train_indices)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        Subset(val_dataset, sorted(val_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = None
    if test_indices is not None:
        test_loader = DataLoader(
            Subset(test_dataset, sorted(test_indices)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    return train_loader, val_loader, test_loader


def prepare_crossval_folds(
    data_dir: str,
    num_folds: int,
    seed: int,
    duplicate_threshold: int = 10,
):
    """Build duplicate-aware cross-validation folds and metadata."""
    dataset = CircuitDataset(data_dir)
    folds, diagnostics = build_grouped_folds(
        dataset.samples,
        dataset.classes,
        num_folds=num_folds,
        seed=seed,
        duplicate_threshold=duplicate_threshold,
    )
    diagnostics["class_names"] = dataset.classes
    diagnostics["class_distribution"] = dataset.get_class_distribution()
    return dataset.classes, folds, diagnostics


def save_split_info(output_dir: Path, split_info: Dict):
    """Persist split diagnostics."""
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
