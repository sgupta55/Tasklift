#!/usr/bin/env python
"""Build the paper's tables from evaluation JSON files (layout produced by run_paper.sh).

  runs/W_<ds>/eval/<ds>_<s>shot.json        within-dataset   (Table I, Fig. 2, Table IV)
  runs/X_<ds>/eval/<ds>_<s>shot.json        leave-one-out    (Table I)
  runs/abl_<name>_<ds>/eval/<ds>_5shot.json ablations        (Table III)
  runs/baselines/mean_mask/<ds>_<s>shot.json mean-support-mask control

  python scripts/make_tables.py --runs runs --out tables
"""
import argparse
import json
from pathlib import Path

import numpy as np

DS = ["isic", "brats", "kvasir", "chase"]
NAMES = {"isic": "ISIC 2018", "brats": "BraTS 2021 WT", "kvasir": "Kvasir-SEG", "chase": r"CHASE\_DB1$^\ddagger$"}
ABLATIONS = [("compile_off", r"Compilation off (prior means $\mu$ as readouts)"),
             ("fixed_stencils", r"Fixed stencils $a^\star$ (no meta-learned taps)"),
             ("r2_off", r"Amplification penalty $R_2$ off"),
             ("J2", r"$J=2$"), ("J3", r"$J=3$"), ("J5", r"$J=5$"),
             ("r2", r"Initial sum-rule order $r=2$"), ("r3", r"Initial sum-rule order $r=3$"),
             ("degree1", r"Degree-1 features ($F=34$)"),
             ("width16", "Projection width 16"), ("width32", "Projection width 32"),
             ("cp1", r"$C'=1$"), ("cp8", r"$C'=8$"), ("mean_mask", "Mean support mask (no learning)")]


def load(p: Path):
    return json.loads(p.read_text())["summary"] if p.exists() else None


def cell(s, m, ci=True, digits=1):
    if s is None:
        return "--"
    v = s[m]
    return f"{v['mean']:.{digits}f}" + (f"$\\pm${v['half_width']:.{digits}f}" if ci else "")


def table1(runs: Path) -> str:
    out = [r"\begin{tabular}{llccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{Within (W)} & \multicolumn{2}{c}{Leave-one-out (X)}\\",
           r"Dataset & $n_s$ & Dice & IoU & ASSD & Dice & IoU\\", r"\midrule"]
    means = {k: {1: [], 5: []} for k in ("wd", "wi", "xd", "xi")}
    for d in DS:
        for s in (1, 5):
            w = load(runs / f"W_{d}" / "eval" / f"{d}_{s}shot.json")
            x = load(runs / f"X_{d}" / "eval" / f"{d}_{s}shot.json")
            ci = d != "chase"     # R-CHASE: intervals omitted (six support draws)
            name = NAMES[d] if s == 1 else ""
            out.append(f"{name} & {s} & {cell(w, 'dice', ci)} & {cell(w, 'iou', ci)} & {cell(w, 'assd', ci)}"
                       f" & {cell(x, 'dice', ci)} & {cell(x, 'iou', ci)}\\\\")
            for key, src, m in (("wd", w, "dice"), ("wi", w, "iou"), ("xd", x, "dice"), ("xi", x, "iou")):
                if src:
                    means[key][s].append(src[m]["mean"])
    out.append(r"\midrule")
    for s in (1, 5):
        f = lambda v: f"{np.mean(v):.1f}" if len(v) == 4 else "--"
        out.append(f"Mean & {s} & {f(means['wd'][s])} & {f(means['wi'][s])} & --- & "
                   f"{f(means['xd'][s])} & {f(means['xi'][s])}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def ablation_table(runs: Path) -> str:
    def mean5(prefix):
        vals = []
        for d in DS:
            p = (runs / "baselines" / "mean_mask" / f"{d}_5shot.json") if prefix == "mean_mask" else \
                (runs / (f"W_{d}" if prefix == "full" else f"abl_{prefix}_{d}") / "eval" / f"{d}_5shot.json")
            s = load(p)
            if s:
                vals.append(s["dice"]["mean"])
        return float(np.mean(vals)) if len(vals) == 4 else None
    full = mean5("full")
    out = [r"\begin{tabular}{lcc}", r"\toprule", r"Variant & Dice & $\Delta$\\", r"\midrule",
           f"Full model & {full:.1f} & ---\\\\" if full is not None else r"Full model & -- & ---\\"]
    for key, label in ABLATIONS:
        v = mean5(key)
        if v is None:
            continue
        delta = f"{v - full:+.1f}" if full is not None else "--"
        out.append(f"{label} & {v:.1f} & ${delta}$\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tertile_table(runs: Path) -> str:
    out = [r"\begin{tabular}{lccc}", r"\toprule", r"Dataset & Low & Mid & High\\", r"\midrule"]
    for d in DS:
        s = load(runs / f"W_{d}" / "eval" / f"{d}_5shot.json")
        if s and s.get("tertiles"):
            t = s["tertiles"]
            out.append(f"{NAMES[d]} & {t['low']:.1f} & {t['mid']:.1f} & {t['high']:.1f}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def summary_md(runs: Path) -> str:
    lines = ["# Results summary", "", "## Shot scaling (within, Dice %)", "",
             "| Dataset | 1 | 2 | 3 | 5 | 10 |", "|---|---|---|---|---|---|"]
    for d in DS:
        row = [d]
        for s in (1, 2, 3, 5, 10):
            sm = load(runs / f"W_{d}" / "eval" / f"{d}_{s}shot.json")
            row.append(f"{sm['dice']['mean']:.1f}" if sm else "--")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Support-set SD of Dice (5-shot / 1-shot) and amplification", "",
              "| Dataset | SD 5 | SD 1 | median prod rho | p95 prod rho |", "|---|---|---|---|---|"]
    for d in DS:
        s5 = load(runs / f"W_{d}" / "eval" / f"{d}_5shot.json")
        s1 = load(runs / f"W_{d}" / "eval" / f"{d}_1shot.json")
        sv = lambda s: f"{s['support_variance']['sd']:.1f}" if s and "support_variance" in s else "--"
        amp = s5.get("amplification", {}) if s5 else {}
        lines.append(f"| {d} | {sv(s5)} | {sv(s1)} | {amp.get('median', float('nan')):.3f} | "
                     f"{amp.get('p95', float('nan')):.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table1_main.tex").write_text(table1(runs) + "\n")
    (out / "table3_ablations.tex").write_text(ablation_table(runs) + "\n")
    (out / "table4_tertiles.tex").write_text(tertile_table(runs) + "\n")
    (out / "summary.md").write_text(summary_md(runs))
    print(f"tables written to {out}/")


if __name__ == "__main__":
    main()
