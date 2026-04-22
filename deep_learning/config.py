"""Deep-learning-specific hyperparameters for Person B's pipeline.

Lives separately from shared.config because these do not affect the
classical pipeline and should not be a source of ambiguity across
team members. All values are justified in the Phase 0 design
justification document, §4 (preprocessing) and the Phase 2 plan.
"""
from __future__ import annotations

# --- Architecture ---
MODEL_NAME: str = "efficientnet_b0"
NUM_CLASSES: int = 4

# --- Input ---
CNN_IMAGE_SIZE: tuple[int, int] = (224, 224)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# --- Training ---
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4  # override to 2 on Colab if I/O-bound

# Stage 1: head-only training with frozen backbone
STAGE1_EPOCHS: int = 5
STAGE1_LR: float = 1e-3

# Stage 2: fine-tune last blocks of backbone
STAGE2_EPOCHS: int = 30
STAGE2_LR: float = 1e-5
STAGE2_UNFREEZE_BLOCKS: int = 2
STAGE2_WEIGHT_DECAY: float = 1e-4

# Early stopping on validation macro-F1
EARLY_STOPPING_PATIENCE: int = 5

# --- Augmentation (train-time only) ---
AUG_HFLIP_PROB: float = 0.5
AUG_ROTATION_DEG: float = 15.0
AUG_ZOOM_RANGE: float = 0.1
AUG_BRIGHTNESS: float = 0.1
AUG_CONTRAST: float = 0.1

# --- Class imbalance handling ---
USE_CLASS_WEIGHTS: bool = True

# --- Checkpoint / results naming ---
CHECKPOINT_FILENAME: str = "best_model.pt"
RESULTS_FILENAME: str = "dl_results.json"
