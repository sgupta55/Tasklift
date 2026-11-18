#!/usr/bin/env python
"""Run the numerical audit (paper Table V and every theorem) and write a report.

  python scripts/verify.py --out runs/verification
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasklift.utils import save_json  # noqa: E402
from tasklift.verification import run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/verification")
    a = ap.parse_args()
    results = run_all()
    lines = ["| Check | Value | Threshold | Pass | Detail |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['value']:.3e} | {r['threshold']:.0e} | "
                     f"{'yes' if r['passed'] else 'NO'} | {r['detail']} |")
    report = "\n".join(lines)
    print(report)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.md").write_text(report + "\n")
    save_json(results, out / "verification.json")
    failed = [r["name"] for r in results if not r["passed"]]
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print(f"all {len(results)} checks passed")


if __name__ == "__main__":
    main()
