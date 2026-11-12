"""Separate PyTorch DataLoader construction for all three dataset partitions."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from dataset import FraudSplits


@dataclass
class FraudLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_dataloaders(
    splits: FraudSplits,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> FraudLoaders:
    """Call once per model with the SAME seed for the SAME shuffle order."""
    generator = torch.Generator().manual_seed(seed)
    options = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return FraudLoaders(
        train=DataLoader(splits.train, shuffle=True, generator=generator, **options),
        val=DataLoader(splits.val, shuffle=False, **options),
        test=DataLoader(splits.test, shuffle=False, **options),
    )
