#!/usr/bin/env python
"""Run an A/B experiment: several configurations, each with several seeds.

Every run is a separate process, so a crash or an out-of-memory error ends
only that run. Finished runs are skipped, which makes the experiment resumable.

Usage:
    python scripts/ab_test.py configs/exp/ptm.yaml [--dry-run] [--report-only]
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train as train_script  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from avsr.config import PROJECT_ROOT, load_config  # noqa: E402
from avsr.utils.ab import paired_comparison, summarize  # noqa: E402


def load_experiment(path):
    exp = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    for key in ("name", "arms", "control"):
        if key not in exp:
            raise ValueError(f"{path}: missing '{key}'")
    if exp["control"] not in exp["arms"]:
        raise ValueError(f"{path}: control arm '{exp['control']}' is not one of the arms")
    exp.setdefault("seeds", [1, 2, 3])
    exp.setdefault("metrics", ["val/cer", "val/loss"])
    exp.setdefault("files", [])
    exp.setdefault("overrides", [])
    return exp


def run_dir(exp, arm, seed):
    return os.path.join(PROJECT_ROOT, "exp", "ab", exp["name"], arm, f"seed{seed}")


def run_overrides(exp, arm, seed):
    return [
        *exp["overrides"],
        *exp["arms"][arm],
        f"seed={seed}",
        "exp_dir=exp",
        f"exp_name=ab/{exp['name']}/{arm}/seed{seed}",
    ]


def best_values(directory, metrics):
    """Best (lowest) value of each metric over the validation epochs of a run."""
    best = {}
    for row in train_script.read_metrics(directory):
        for metric in metrics:
            value = row.get(metric)
            if value not in (None, ""):
                best[metric] = min(best.get(metric, float("inf")), float(value))
    return best


def execute(exp, arm, seed, timeout):
    directory = run_dir(exp, arm, seed)
    os.makedirs(directory, exist_ok=True)
    command = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "train.py")]
    for path in exp["files"]:
        command += ["-c", path]
    command += run_overrides(exp, arm, seed)
    started = time.time()
    with open(os.path.join(directory, "train.log"), "w") as log_file:
        try:
            code = subprocess.run(
                command, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    status = {"returncode": code, "minutes": (time.time() - started) / 60}
    with open(os.path.join(directory, "status.json"), "w") as f:
        json.dump(status, f)
    return code == 0


def collect(exp):
    """Read the results of every finished run."""
    results = {}
    for arm in exp["arms"]:
        for seed in exp["seeds"]:
            directory = run_dir(exp, arm, seed)
            path = os.path.join(directory, "metrics.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                final = json.load(f)
            best = best_values(directory, exp["metrics"])
            results[(arm, seed)] = {"final": final, "best": best}
    return results


def report(exp, results):
    lines = [f"# A/B experiment: {exp['name']}", ""]
    if exp.get("description"):
        lines += [exp["description"], ""]
    lines += [f"Control arm: `{exp['control']}`. Seeds: {exp['seeds']}.", ""]

    finished = {arm: [s for s in exp["seeds"] if (arm, s) in results] for arm in exp["arms"]}
    missing = {arm: [s for s in exp["seeds"] if s not in done] for arm, done in finished.items()}
    if any(missing.values()):
        lines += ["**Incomplete:** " + ", ".join(f"{a} seeds {s}" for a, s in missing.items() if s), ""]

    for metric in exp["metrics"]:
        lines += [f"## {metric} (best epoch of each run; lower is better)", ""]
        lines += ["| arm | runs | mean | std | diff vs control | 95 % CI | verdict |", "|---|---|---|---|---|---|---|"]
        values = {
            arm: {s: results[(arm, s)]["best"][metric] for s in finished[arm] if metric in results[(arm, s)]["best"]}
            for arm in exp["arms"]
        }
        for arm in exp["arms"]:
            mean, std = summarize(values[arm].values())
            if arm == exp["control"]:
                diff = ci = verdict = "—"
            else:
                cmp = paired_comparison(values[exp["control"]], values[arm])
                diff, ci, verdict = f"{cmp['mean_diff']:+.4f}", f"±{cmp['ci95']:.4f}", cmp["verdict"]
            lines.append(f"| {arm} | {len(values[arm])} | {mean:.4f} | {std:.4f} | {diff} | {ci} | {verdict} |")
        lines.append("")

    lines += ["## Cost", "", "| arm | minutes/run | GPU peak (GB) | OOM batches |", "|---|---|---|---|"]
    for arm in exp["arms"]:
        finals = [results[(arm, s)]["final"] for s in finished[arm]]
        if not finals:
            continue
        minutes = summarize(f.get("train_minutes", float("nan")) for f in finals)[0]
        peak = max(f.get("gpu_peak_gb", float("nan")) for f in finals)
        ooms = sum(f.get("oom_batches", 0) for f in finals)
        lines.append(f"| {arm} | {minutes:.1f} | {peak:.2f} | {ooms} |")
    lines += [
        "",
        "A verdict of *lower* or *higher* means the paired 95 % confidence interval of the "
        "difference excludes zero. With few seeds and a small validation set the intervals "
        "are wide, so *inconclusive* is the expected outcome for small effects.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    parser.add_argument("--dry-run", action="store_true", help="Validate every configuration without training")
    parser.add_argument("--report-only", action="store_true", help="Summarise finished runs")
    parser.add_argument("--timeout-minutes", type=float, default=0, help="Per-run limit (0 = none)")
    args = parser.parse_args()

    exp = load_experiment(args.experiment)
    runs = [(arm, seed) for seed in exp["seeds"] for arm in exp["arms"]]

    # Fail before any training if a configuration is invalid.
    for arm, seed in runs:
        load_config(exp["files"], run_overrides(exp, arm, seed))
    if args.dry_run:
        print(f"{len(runs)} runs validated: arms {list(exp['arms'])}, seeds {exp['seeds']}")
        return

    if not args.report_only:
        timeout = args.timeout_minutes * 60 or None
        for i, (arm, seed) in enumerate(runs, 1):
            if os.path.isfile(os.path.join(run_dir(exp, arm, seed), "metrics.json")):
                print(f"[{i}/{len(runs)}] {arm} seed {seed}: already done")
                continue
            print(f"[{i}/{len(runs)}] {arm} seed {seed}: running", flush=True)
            ok = execute(exp, arm, seed, timeout)
            if not ok:
                print(f"    failed; see {os.path.join(run_dir(exp, arm, seed), 'train.log')}")

    text = report(exp, collect(exp))
    out = os.path.join(PROJECT_ROOT, "exp", "ab", exp["name"], "report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
