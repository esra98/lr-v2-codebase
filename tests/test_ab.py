import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import ab_test  # noqa: E402

from avsr.utils.ab import paired_comparison, summarize  # noqa: E402


def test_summarize():
    assert summarize([1.0, 2.0, 3.0]) == (2.0, 1.0)
    mean, std = summarize([5.0])
    assert mean == 5.0 and math.isnan(std)
    assert all(math.isnan(v) for v in summarize([]))


def test_paired_comparison_detects_a_consistent_gain():
    control = {1: 0.50, 2: 0.60, 3: 0.55}
    treatment = {1: 0.45, 2: 0.55, 3: 0.49}
    result = paired_comparison(control, treatment)
    assert result["n"] == 3 and result["verdict"] == "lower"
    assert result["mean_diff"] == pytest.approx(-0.0533, abs=1e-3)


def test_paired_comparison_is_inconclusive_for_noise():
    control = {1: 0.50, 2: 0.60, 3: 0.55}
    treatment = {1: 0.55, 2: 0.52, 3: 0.57}
    assert paired_comparison(control, treatment)["verdict"] == "inconclusive"


def test_paired_comparison_pairs_by_seed_not_by_mean():
    # Large seed-to-seed variance, small but consistent gain.
    control = {1: 0.30, 2: 0.60, 3: 0.90}
    treatment = {1: 0.28, 2: 0.58, 3: 0.875}
    assert paired_comparison(control, treatment)["verdict"] == "lower"


def test_paired_comparison_with_missing_runs():
    assert paired_comparison({1: 0.5}, {1: 0.4})["verdict"] == "inconclusive (one seed)"
    assert paired_comparison({1: 0.5}, {2: 0.4})["verdict"] == "no data"
    partial = paired_comparison({1: 0.5, 2: 0.6, 3: 0.7}, {1: 0.4, 3: 0.6})
    assert partial["n"] == 2


def _experiment(tmp_path, body):
    path = tmp_path / "exp.yaml"
    path.write_text(body)
    return str(path)


def test_load_experiment_validation(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        ab_test.load_experiment(_experiment(tmp_path, "name: x\narms: {a: []}\n"))
    with pytest.raises(ValueError, match="control"):
        ab_test.load_experiment(_experiment(tmp_path, "name: x\ncontrol: b\narms: {a: []}\n"))
    exp = ab_test.load_experiment(_experiment(tmp_path, "name: x\ncontrol: a\narms: {a: [], b: [seed=1]}\n"))
    assert exp["seeds"] == [1, 2, 3] and exp["metrics"] == ["val/cer", "val/loss"]


def test_report_from_finished_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_test, "PROJECT_ROOT", str(tmp_path))
    exp = ab_test.load_experiment(
        _experiment(tmp_path, "name: demo\ncontrol: base\nseeds: [1, 2, 3]\narms: {base: [], new: []}\n")
    )
    values = {"base": [0.50, 0.60, 0.55], "new": [0.45, 0.55, 0.49]}
    for arm, cers in values.items():
        for seed, cer in zip(exp["seeds"], cers):
            if arm == "new" and seed == 3:
                continue  # an unfinished run
            directory = ab_test.run_dir(exp, arm, seed)
            os.makedirs(os.path.join(directory, "csv"))
            with open(os.path.join(directory, "metrics.json"), "w") as f:
                json.dump({"val/cer": cer + 0.1, "train_minutes": 2.0, "gpu_peak_gb": 3.0, "oom_batches": 0}, f)
            with open(os.path.join(directory, "csv", "metrics.csv"), "w") as f:
                f.write(f"epoch,val/cer,val/loss,train/loss\n0,{cer + 0.2},9.0,\n0,,,5.0\n1,{cer},8.0,\n")
    results = ab_test.collect(exp)
    assert len(results) == 5
    assert results[("base", 1)]["best"] == {"val/cer": 0.50, "val/loss": 8.0}
    text = ab_test.report(exp, results)
    assert "**Incomplete:** new seeds [3]" in text
    assert "| base | 3 | 0.5500 |" in text
    assert "| new | 2 | 0.5000 |" in text and "-0.0500" in text
