"""Preprocessing, splits and episode protocol on synthetic data in the real layouts."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tasklift.data import ItemStore, eval_episodes, rchase_episodes, sample_episodes
from tasklift.data.preprocess import prepare_brats, prepare_rgb, write_splits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_data.py"), "--out", str(tmp / "raw"),
                    "--n", "24"], check=True, capture_output=True)
    for d in ("isic", "kvasir", "chase"):
        prepare_rgb(d, tmp / "raw" / d, tmp / "proc", 64)
        write_splits(tmp / "proc", d)
    prepare_brats(tmp / "raw" / "brats", tmp / "proc", 64)
    write_splits(tmp / "proc", "brats")
    return tmp / "proc"


def test_items_and_splits(processed):
    for d in ("isic", "kvasir", "chase", "brats"):
        s = ItemStore(processed, d)
        x, y = s.load(s.ids("train")[:2])
        assert x.shape == (2, 4, 64, 64) and y.shape == (2, 1, 64, 64)
        assert set(np.unique(y.numpy())) <= {0.0, 1.0}
        parts = [set(s.ids(k)) for k in ("train", "val", "test")]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    chase = ItemStore(processed, "chase")
    assert [len(chase.ids(k)) for k in ("train", "val", "test")] == [16, 6, 6]


def test_brats_case_level_split(processed):
    s = ItemStore(processed, "brats")
    groups = [{s.group(i) for i in s.ids(k)} for k in ("train", "val", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_rchase_counts(processed):
    s = ItemStore(processed, "chase")
    e1, e5 = rchase_episodes(s, 1), rchase_episodes(s, 5)
    assert len(e1) == 6 and all(len(e.query) == 5 for e in e1)      # 30 one-shot predictions
    assert len(e5) == 6 and all(len(e.query) == 1 for e in e5)      # 6 five-shot predictions
    assert eval_episodes(s, 5) == e5
    with pytest.raises(ValueError):
        rchase_episodes(s, 6)


def test_sampled_episodes_deterministic(processed):
    s = ItemStore(processed, "isic")
    a = sample_episodes(s, "train", 2, 3, 5, seed=1)
    b = sample_episodes(s, "train", 2, 3, 5, seed=1)
    assert a == b
    for e in a:
        assert len(set(e.support + e.query)) == 5
