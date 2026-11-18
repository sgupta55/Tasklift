"""Seeding, determinism, device and I/O helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_determinism(seed: int = 0) -> None:
    """Fixed seeds and deterministic kernels (TD Sec. 10.3)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())
