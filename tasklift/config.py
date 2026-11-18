"""Configuration dataclasses (defaults = paper / technical-document Appendix C)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    # Lattice / analysis
    image_size: int = 256          # N
    levels: int = 4                # J
    in_channels: int = 4           # C (RGB is zero-padded to 4)
    mix_channels: int = 4          # C'
    init_order: int = 4            # r: sum-rule order of the initial predictor stencil
    learn_kernels: bool = True     # False -> fixed canonical stencils (classical lifting control)
    # Features
    width: int = 24                # projection width for the y/x stages
    root_width: int = 16           # projection width for the root stage
    degree: int = 2                # polynomial degree of psi (1 or 2)
    # Compiler
    compile: bool = True           # False -> use prior means mu as readouts ("compilation off")
    lam_init: float = 1.0          # lambda = softplus(raw), raw initialised so lambda = lam_init
    fp64_cond_threshold: float = 1e8  # promote a stage solve to float64 above this condition estimate
    class_balance_eps: float = 1e-3
    checkpoint_gram: bool = False  # recompute feature rows in backward to save memory
    gram_row_chunk: int = 65536
    # Calibration
    calib_ridge: float = 1e-2      # eta_c
    calib_newton_steps: int = 10

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TrainConfig:
    protocol: str = "within"       # "within" | "cross"
    dataset: str = "isic"          # within: the dataset; cross: the held-out dataset
    data_root: str = "data/processed"
    out_dir: str = "runs/default"
    seed: int = 0
    device: str = "auto"
    # Episodes
    train_shots: tuple[int, ...] = (1, 5)
    n_query: int = 8
    distinct_groups: bool = True   # episode images come from distinct cases/patients
    # Optimisation (TD Sec. 9.3)
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    episode_batch: int = 4
    max_episodes: int = 60000
    eta1: float = 1e-2             # R1 (kernel l1 penalty)
    eta2: float = 1e-1             # R2 (sweep amplification penalty)
    eta3: float = 1e-4             # ||Theta||^2
    # Validation / early stopping
    val_every: int = 500           # in episodes
    val_episodes: int = 100
    patience: int = 10             # in evaluations
    log_every: int = 100           # in episodes
    num_workers: int = 0
    cache_in_memory: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["train_shots"] = list(self.train_shots)
        return d


def _coerce(value: str, target_type: Any, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        return tuple(int(v) for v in value.split(","))
    return value


def apply_overrides(cfg: Any, overrides: list[str] | None) -> Any:
    """Apply `key=value` overrides to a dataclass in place."""
    names = {f.name for f in fields(cfg)}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        if key not in names:
            continue
        setattr(cfg, key, _coerce(value, None, getattr(cfg, key)))
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None
                ) -> tuple[ModelConfig, TrainConfig]:
    """Load `model:` and `train:` sections from YAML, then apply `key=value` overrides
    (keys are matched against both dataclasses)."""
    model, train = ModelConfig(), TrainConfig()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for section, obj in (("model", model), ("train", train)):
            for k, v in (raw.get(section) or {}).items():
                if not hasattr(obj, k):
                    raise KeyError(f"unknown {section} key: {k}")
                cur = getattr(obj, k)
                if isinstance(cur, tuple):
                    v = tuple(int(x) for x in v)
                elif isinstance(v, str) and not isinstance(cur, str):
                    v = _coerce(v, None, cur)          # e.g. YAML 1.1 reads "1.0e8" as a string
                elif isinstance(cur, float) and isinstance(v, int) and not isinstance(v, bool):
                    v = float(v)
                setattr(obj, k, v)
    apply_overrides(model, overrides)
    apply_overrides(train, overrides)
    return model, train
