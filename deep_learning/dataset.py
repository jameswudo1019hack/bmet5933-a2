"""PyTorch Dataset wrapper around a stratified split CSV.

Reads the fixed split, loads images through shared.preprocessing.load_image
(grayscale 256×256), then applies CNN-specific transforms: resize to
`image_size`, channel replication (1→3), ImageNet normalisation, and
train-time augmentation.

Defaults: primary split.csv, 224×224 input (EfficientNet-B0 Phase 2).
Override image_size and split_csv for the Sprint 2 ConvNeXt V2 run
(384×384 input, split_full.csv).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from deep_learning.config import (
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_HFLIP_PROB,
    AUG_ROTATION_DEG,
    AUG_ZOOM_RANGE,
    CNN_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from shared.preprocessing import load_image, load_split


def _to_3channel_tensor(arr: np.ndarray) -> torch.Tensor:
    # shared.preprocessing returns uint8 (H, W) grayscale. Expand to (3, H, W) float in [0,1].
    # torch.tensor() (not from_numpy) because PIL-backed arrays are read-only and torch warns.
    t = torch.tensor(arr, dtype=torch.float32) / 255.0
    return t.unsqueeze(0).repeat(3, 1, 1)


def build_transforms(
    train: bool,
    image_size: tuple[int, int] | int = CNN_IMAGE_SIZE,
) -> Callable[[np.ndarray], torch.Tensor]:
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    zoom = AUG_ZOOM_RANGE
    if train:
        tfm = transforms.Compose(
            [
                transforms.Lambda(_to_3channel_tensor),
                transforms.RandomHorizontalFlip(p=AUG_HFLIP_PROB),
                transforms.RandomAffine(degrees=AUG_ROTATION_DEG),
                transforms.RandomResizedCrop(
                    size=image_size, scale=(1 - zoom, 1.0), antialias=True
                ),
                transforms.ColorJitter(brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        tfm = transforms.Compose(
            [
                transforms.Lambda(_to_3channel_tensor),
                transforms.Resize(image_size, antialias=True),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return tfm


class KidneyCTDataset(Dataset):
    """PyTorch Dataset over one split ('train' | 'val' | 'test')."""

    def __init__(
        self,
        split_name: str,
        train: bool | None = None,
        split_csv: str | Path | None = None,
        dataset_root: str | Path | None = None,
        image_size: tuple[int, int] | int = CNN_IMAGE_SIZE,
    ):
        self.df = load_split(split_name, split_csv=split_csv, dataset_root=dataset_root)
        self._train = (split_name == "train") if train is None else train
        self.transform = build_transforms(self._train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        arr = load_image(row["abs_path"])
        x = self.transform(arr)
        y = int(row["class_idx"])
        return x, y
