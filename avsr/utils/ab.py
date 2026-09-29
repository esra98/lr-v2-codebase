"""Statistics for A/B comparisons between training configurations."""

import math
import statistics

# Two-sided 95 % critical values of Student's t, indexed by degrees of freedom.
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_critical(dof):
    if dof < 1:
        return float("inf")
    return _T_95.get(dof, 1.96 if dof > 30 else 2.1)


def summarize(values):
    """Mean and sample standard deviation; the latter is NaN for one value."""
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else float("nan")
    return mean, std


def paired_comparison(control, treatment):
    """Compare two arms run with the same seeds.

    Args:
        control, treatment: dicts mapping seed -> metric value.

    Returns a dict with the mean difference (treatment - control), its 95 %
    confidence half-width and a verdict. Pairing by seed removes the variance
    that both arms share through identical initialisation and data order.
    """
    seeds = sorted(set(control) & set(treatment))
    diffs = [treatment[s] - control[s] for s in seeds]
    result = {"n": len(diffs), "mean_diff": float("nan"), "ci95": float("nan"), "verdict": "no data"}
    if not diffs:
        return result
    result["mean_diff"] = statistics.fmean(diffs)
    if len(diffs) < 2:
        result["verdict"] = "inconclusive (one seed)"
        return result
    sem = statistics.stdev(diffs) / math.sqrt(len(diffs))
    result["ci95"] = t_critical(len(diffs) - 1) * sem
    if abs(result["mean_diff"]) > result["ci95"]:
        result["verdict"] = "lower" if result["mean_diff"] < 0 else "higher"
    else:
        result["verdict"] = "inconclusive"
    return result
