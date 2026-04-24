"""Deep-learning model builders and transfer-learning utilities.

Primary model: EfficientNet-B0 (torchvision, ImageNet-1k). Used for the
main Phase 2 comparison on the medium dataset.

Supplementary model: ConvNeXt V2 Base (timm, ImageNet-22k→1k @ 384). Used
for the Sprint 2 scale-validation experiment on the full dataset — see
Planning/experiments/Sprint2_ConvNeXtV2_on_full.md.

Two-stage training is configured via freeze/unfreeze helpers that auto-
detect the backbone type, so the training loop in deep_learning.train
works identically for both architectures.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from deep_learning.config import NUM_CLASSES
from shared.config import CLASSES
from shared.preprocessing import load_split


# ─── EfficientNet-B0 (primary) ─────────────────────────────────────────────

def build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─── ConvNeXt V2 Base (supplementary) ──────────────────────────────────────

def build_convnextv2_base(
    num_classes: int = NUM_CLASSES,
    image_size: int = 384,
    drop_path_rate: float = 0.3,
    drop_rate: float = 0.3,
) -> nn.Module:
    """Build ConvNeXt V2 Base via timm with ImageNet-22k→1k weights.

    At 384 resolution we use the FCMAE-pretrained, 22k-fine-tuned, then
    1k-fine-tuned checkpoint ("fcmae_ft_in22k_in1k_384"). At 224 we fall
    back to the 224-resolution variant.

    drop_path_rate: stochastic depth — key regularisation for an 89M-param
      model on ~8,700 training images.
    drop_rate: dropout applied before the classifier head.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for ConvNeXt V2. "
            "Install with: pip install 'timm>=1.0.9'"
        ) from e

    if image_size >= 384:
        model_name = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    else:
        model_name = "convnextv2_base.fcmae_ft_in22k_in1k"
    return timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )


# ─── Dispatcher ────────────────────────────────────────────────────────────

def build_model(
    name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    image_size: int = 224,
) -> nn.Module:
    if name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes)
    if name == "convnextv2_base":
        return build_convnextv2_base(num_classes=num_classes, image_size=image_size)
    raise ValueError(f"Unknown model: {name!r}. Options: efficientnet_b0, convnextv2_base")


# ─── Freeze / unfreeze (backbone-type-aware) ───────────────────────────────

def _backbone_stages(model: nn.Module) -> tuple[str, list[nn.Module]]:
    """Return ("efficientnet" | "convnextv2", list of sequential stage modules)."""
    if hasattr(model, "features"):  # torchvision EfficientNet
        return "efficientnet", list(model.features.children())
    if hasattr(model, "stages"):   # timm ConvNeXt V2
        return "convnextv2", list(model.stages.children())
    raise ValueError(f"Unknown backbone structure in model of type {type(model).__name__}")


def _head_parameters(model: nn.Module) -> list[nn.Parameter]:
    if hasattr(model, "classifier"):
        return list(model.classifier.parameters())
    if hasattr(model, "head"):
        return list(model.head.parameters())
    raise ValueError(f"Unknown head on model of type {type(model).__name__}")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone params; keep classifier head trainable."""
    arch, stages = _backbone_stages(model)
    for stage in stages:
        for p in stage.parameters():
            p.requires_grad = False
    # For ConvNeXt V2, also freeze the stem
    if arch == "convnextv2" and hasattr(model, "stem"):
        for p in model.stem.parameters():
            p.requires_grad = False
    for p in _head_parameters(model):
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    """Unfreeze the last n_blocks backbone stages and the head. Stem stays frozen."""
    arch, stages = _backbone_stages(model)
    for i, stage in enumerate(stages):
        requires_grad = i >= len(stages) - n_blocks
        for p in stage.parameters():
            p.requires_grad = requires_grad
    # Stem always stays frozen during fine-tuning (low-level features transfer well)
    if arch == "convnextv2" and hasattr(model, "stem"):
        for p in model.stem.parameters():
            p.requires_grad = False
    for p in _head_parameters(model):
        p.requires_grad = True


# ─── Utilities ─────────────────────────────────────────────────────────────

def compute_class_weights(split_csv: Optional[str] = None) -> torch.Tensor:
    """Inverse-frequency class weights from the train split of the given CSV.

    Defaults to the primary split.csv. Pass split_csv path to use a different
    split (e.g. split_full.csv for the full-dataset ConvNeXt V2 run).
    """
    train_df = load_split("train", split_csv=split_csv)
    counts = Counter(train_df["class"])
    freq = np.array([counts[cls] for cls in CLASSES], dtype=np.float64)
    inv = 1.0 / freq
    weights = inv / inv.sum() * len(CLASSES)
    return torch.tensor(weights, dtype=torch.float32)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "convnextv2_base"])
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    m = build_model(name=args.model, image_size=args.image_size)
    print(f"Model: {args.model}  image_size={args.image_size}")
    print(f"Total params:      {sum(p.numel() for p in m.parameters()):,}")
    freeze_backbone(m)
    print(f"Stage-1 trainable: {count_trainable(m):,}  (head only)")
    unfreeze_last_blocks(m, n_blocks=2)
    print(f"Stage-2 trainable: {count_trainable(m):,}  (+ last 2 blocks)")
    print(f"Class order:       {list(CLASSES)}")
