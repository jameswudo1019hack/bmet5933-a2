"""EfficientNet-B0 transfer-learning model and training utilities.

Backbone = torchvision EfficientNet-B0 with ImageNet weights.
Head is replaced with a dropout + linear layer mapping 1280-dim
features to NUM_CLASSES logits.

Two-stage training is configured via freeze/unfreeze helpers rather
than duplicating the model definition.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from deep_learning.config import NUM_CLASSES
from shared.config import CLASSES, CLASS_TO_IDX
from shared.preprocessing import load_split


def build_model() -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # torchvision's EfficientNet classifier is Sequential(Dropout, Linear).
    # Preserve the Dropout and replace the Linear with our 4-way head.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    # model.features is a Sequential of 9 stages (indices 0..8) for B0.
    # n_blocks=2 unfreezes stages 7 and 8 (the last two blocks before classifier).
    stages = list(model.features.children())
    for i, stage in enumerate(stages):
        requires_grad = i >= len(stages) - n_blocks
        for p in stage.parameters():
            p.requires_grad = requires_grad
    for p in model.classifier.parameters():
        p.requires_grad = True


def compute_class_weights() -> torch.Tensor:
    """Inverse-frequency class weights computed from the train split."""
    train_df = load_split("train")
    counts = Counter(train_df["class"])
    # Order weights by CLASSES tuple so indices match model output
    freq = np.array([counts[cls] for cls in CLASSES], dtype=np.float64)
    inv = 1.0 / freq
    weights = inv / inv.sum() * len(CLASSES)  # normalise so mean ≈ 1
    return torch.tensor(weights, dtype=torch.float32)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_model()
    print(f"Total params:     {sum(p.numel() for p in model.parameters()):,}")
    freeze_backbone(model)
    print(f"Stage-1 trainable: {count_trainable(model):,}  (head only)")
    unfreeze_last_blocks(model, n_blocks=2)
    print(f"Stage-2 trainable: {count_trainable(model):,}  (+ last 2 blocks)")
    print(f"Class weights:    {compute_class_weights().tolist()}")
    print(f"Class order:      {list(CLASSES)}")
