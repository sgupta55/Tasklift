"""Meta-training (TD Sec. 9) and evaluation (TD Sec. 11.4)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .baselines import mean_support_mask
from .config import ModelConfig, TrainConfig
from .data import EpisodeSpec, ItemStore, TrainSampler, sample_episodes
from .data.preprocess import DATASETS
from .losses import l2_penalty, r1_kernel_penalty, r2_amplification_penalty, soft_dice_bce
from .metrics import assd, bootstrap_ci, dice, iou
from .model import TaskLift
from .utils import resolve_device, save_json, set_determinism


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str | Path, model: TaskLift, train_cfg: TrainConfig, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_cfg": model.cfg.to_dict(), "train_cfg": train_cfg.to_dict(),
                "state_dict": model.state_dict(), **extra}, path)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[TaskLift, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TaskLift(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# ----------------------------------------------------------------------------- episodes
def source_datasets(train_cfg: TrainConfig) -> list[str]:
    if train_cfg.protocol == "within":
        return [train_cfg.dataset]
    if train_cfg.protocol == "cross":
        return [d for d in DATASETS if d != train_cfg.dataset]
    raise ValueError(f"unknown protocol {train_cfg.protocol}")


def load_episode(stores: dict[str, ItemStore], spec: EpisodeSpec, device):
    store = stores[spec.dataset]
    xs, ys = store.load(list(spec.support))
    xq, yq = store.load(list(spec.query))
    return xs.to(device), ys.to(device), xq.to(device), yq.to(device)


def validation_specs(stores: list[ItemStore], shots: Iterable[int], n_query: int, n_total: int,
                     seed: int, distinct_groups: bool) -> list[EpisodeSpec]:
    shots = list(shots)
    specs: list[EpisodeSpec] = []
    per = max(1, n_total // (len(stores) * len(shots)))
    for k, store in enumerate(stores):
        groups = len({store.group(i) for i in store.ids("val")}) if distinct_groups else len(store.ids("val"))
        for s in shots:
            nq = min(n_query, groups - s)
            if nq < 1:
                continue
            specs += sample_episodes(store, "val", s, nq, per, seed=seed + 97 * k + s,
                                     distinct_groups=distinct_groups)
    return specs


def episode_loss(model: TaskLift, xs, ys, xq, yq) -> tuple[torch.Tensor, torch.Tensor]:
    """Segmentation loss on the queries and R2 for one episode."""
    z, prob, ep = model(xs, ys, xq)
    seg = soft_dice_bce(prob, yq)
    r2 = r2_amplification_penalty(ep.label_l1(model.J)) if model.cfg.compile else seg.new_zeros(())
    return seg, r2


# ----------------------------------------------------------------------------- training
def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Path:
    set_determinism(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    out = Path(train_cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": model_cfg.to_dict(),
                                                 "train": train_cfg.to_dict()}, indent=2))

    names = source_datasets(train_cfg)
    stores = {n: ItemStore(train_cfg.data_root, n, cache=train_cfg.cache_in_memory) for n in names}
    sampler = TrainSampler(list(stores.values()), "train", tuple(train_cfg.train_shots),
                           train_cfg.n_query, seed=train_cfg.seed, distinct_groups=train_cfg.distinct_groups)
    val = validation_specs(list(stores.values()), train_cfg.train_shots, train_cfg.n_query,
                           train_cfg.val_episodes, train_cfg.seed, train_cfg.distinct_groups)

    model = TaskLift(model_cfg, seed=train_cfg.seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.999),
                            weight_decay=train_cfg.weight_decay)
    steps = math.ceil(train_cfg.max_episodes / train_cfg.episode_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=train_cfg.lr_min)

    best, bad, episodes = float("inf"), 0, 0
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    run_seg = run_r2 = 0.0
    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        seg_sum = r2_sum = 0.0
        for _ in range(train_cfg.episode_batch):
            xs, ys, xq, yq = load_episode(stores, next(sampler), device)
            seg, r2 = episode_loss(model, xs, ys, xq, yq)
            loss = (seg + train_cfg.eta2 * r2) / train_cfg.episode_batch
            loss.backward()
            seg_sum += float(seg.detach())
            r2_sum += float(r2.detach())
        reg = train_cfg.eta1 * r1_kernel_penalty(model.a, model.u) \
            + train_cfg.eta3 * l2_penalty(model.parameters())
        if reg.requires_grad:
            reg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        sched.step()
        episodes += train_cfg.episode_batch
        run_seg += seg_sum / train_cfg.episode_batch
        run_r2 += r2_sum / train_cfg.episode_batch

        if episodes % train_cfg.log_every < train_cfg.episode_batch:
            k = max(1, train_cfg.log_every // train_cfg.episode_batch)
            rec = {"episodes": episodes, "seg_loss": run_seg / k, "r2": run_r2 / k,
                   "lr": sched.get_last_lr()[0], "B": model.analysis_constant(),
                   "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec))
            run_seg = run_r2 = 0.0

        if episodes % train_cfg.val_every < train_cfg.episode_batch or step == steps - 1:
            vloss = validate(model, stores, val, device)
            improved = vloss < best
            log.write(json.dumps({"episodes": episodes, "val_loss": vloss, "best": improved}) + "\n")
            log.flush()
            print(f"[val] episodes={episodes} loss={vloss:.4f}{' *' if improved else ''}")
            if improved:
                best, bad = vloss, 0
                save_checkpoint(out / "best.pt", model, train_cfg, {"episodes": episodes, "val_loss": vloss})
            else:
                bad += 1
                if bad >= train_cfg.patience:
                    print(f"early stopping at {episodes} episodes")
                    break
    save_checkpoint(out / "last.pt", model, train_cfg, {"episodes": episodes})
    log.close()
    return out / "best.pt"


@torch.no_grad()
def validate(model: TaskLift, stores, specs: list[EpisodeSpec], device) -> float:
    model.eval()
    losses = []
    for spec in specs:
        xs, ys, xq, yq = load_episode(stores, spec, device)
        _, prob, _ = model(xs, ys, xq)
        losses.append(float(soft_dice_bce(prob, yq)))
    return float(np.mean(losses))


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: TaskLift | None, store: ItemStore, specs: list[EpisodeSpec], device,
             baseline: str | None = None, n_boot: int = 10000) -> dict:
    """Per-episode means of Dice / IoU / ASSD (hard masks at 1/2), with bootstrap CIs.

    `baseline="mean_mask"` evaluates the training-free mean-support-mask control."""
    stores = {store.dataset: store}
    ep_rows, q_rows = [], []
    for e, spec in enumerate(specs):
        xs, ys, xq, yq = load_episode(stores, spec, device)
        t = time.perf_counter()
        if baseline == "mean_mask":
            prob, amp = mean_support_mask(ys, xq.shape[0]), float("nan")
        else:
            _, prob, ep = model(xs, ys, xq)
            amp = ep.amplification(model.J)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        pred = (prob >= 0.5).cpu().numpy()[:, 0]
        gt = yq.cpu().numpy()[:, 0] > 0.5
        d = [dice(p, g) for p, g in zip(pred, gt)]
        i = [iou(p, g) for p, g in zip(pred, gt)]
        a = [assd(p, g) for p, g in zip(pred, gt)]
        for qid, dd, ii, aa in zip(spec.query, d, i, a):
            q_rows.append({"episode": e, "id": qid, "fg": store.fg(qid), "dice": dd, "iou": ii, "assd": aa})
        ep_rows.append({"episode": e, "support": list(spec.support), "dice": float(np.mean(d)),
                        "iou": float(np.mean(i)), "assd": float(np.mean(a)),
                        "amplification": amp, "seconds": dt})
    summary = {}
    for m in ("dice", "iou", "assd"):
        mean, lo, hi, hw = bootstrap_ci([r[m] for r in ep_rows], n_boot=n_boot)
        scale = 100.0 if m in ("dice", "iou") else 1.0
        summary[m] = {"mean": mean * scale, "ci_lo": lo * scale, "ci_hi": hi * scale, "half_width": hw * scale}
    summary["n_episodes"] = len(ep_rows)
    summary["n_query_predictions"] = len(q_rows)
    amps = [r["amplification"] for r in ep_rows if np.isfinite(r["amplification"])]
    if amps:
        summary["amplification"] = {"median": float(np.median(amps)), "p95": float(np.quantile(amps, 0.95)),
                                    "max": float(np.max(amps))}
    summary["seconds_per_episode"] = float(np.mean([r["seconds"] for r in ep_rows]))
    summary["tertiles"] = tertile_dice(q_rows)
    return {"summary": summary, "episodes": ep_rows, "queries": q_rows}


def tertile_dice(q_rows: list[dict]) -> dict:
    """Mean Dice (%) in equal-size tertiles of query target extent (foreground fraction)."""
    if len(q_rows) < 3:
        return {}
    rows = sorted(q_rows, key=lambda r: r["fg"])
    parts = np.array_split(np.arange(len(rows)), 3)
    names = ("low", "mid", "high")
    return {n: 100.0 * float(np.mean([rows[k]["dice"] for k in idx])) for n, idx in zip(names, parts)}


@torch.no_grad()
def support_variance(model: TaskLift, store: ItemStore, n_shot: int, device, n_groups: int = 50,
                     n_draws: int = 10, n_query: int = 8, seed: int = 0) -> dict:
    """SD of episode Dice across support draws with the queries held fixed, averaged over
    query groups. For CHASE_DB1 (R-CHASE) the SD over the enumerated draws is reported."""
    from .data import rchase_episodes
    stores = {store.dataset: store}
    if store.dataset == "chase":
        res = evaluate(model, store, rchase_episodes(store, n_shot), device, n_boot=10)
        vals = [100 * r["dice"] for r in res["episodes"]]
        return {"sd": float(np.std(vals, ddof=1)), "n_draws": len(vals)}
    rng = np.random.default_rng(seed + n_shot)
    pool = [i for i in store.ids("test") if store.fg(i) > 0]
    n_query = min(n_query, len(pool) - n_shot - 1)
    if n_query < 1:
        return {"sd": float("nan"), "note": "test pool too small for support resampling"}
    sds = []
    for _ in range(n_groups):
        q = list(rng.choice(pool, size=n_query, replace=False))
        rest = [i for i in pool if i not in q and store.group(i) not in {store.group(x) for x in q}]
        if len(rest) < n_shot + 1:
            continue
        vals = []
        for _ in range(n_draws):
            s = list(rng.choice(rest, size=n_shot, replace=False))
            spec = EpisodeSpec(store.dataset, tuple(s), tuple(q))
            xs, ys, xq, yq = load_episode(stores, spec, device)
            _, prob, _ = model(xs, ys, xq)
            pred = (prob >= 0.5).cpu().numpy()[:, 0]
            gt = yq.cpu().numpy()[:, 0] > 0.5
            vals.append(100 * np.mean([dice(p, g) for p, g in zip(pred, gt)]))
        sds.append(np.std(vals, ddof=1))
    if not sds:
        return {"sd": float("nan"), "note": "test pool too small for support resampling"}
    return {"sd": float(np.mean(sds)), "n_groups": len(sds), "n_draws": n_draws, "n_query": n_query}
