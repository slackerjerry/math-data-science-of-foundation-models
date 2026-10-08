"""Enumerate the full-support view used in the Chapter 4 body.

No fitted model is involved. The matching calculation sums all candidate
magnitude tuples; independent signs have already been marginalized out.
"""
from itertools import product
from math import comb, log
import json


def calculate(k=8):
    values = (-2, -1, 1, 2)
    conditional = {
        0: {-2: 1 / 8, -1: 3 / 8, 1: 3 / 8, 2: 1 / 8},
        1: {-2: 3 / 8, -1: 1 / 8, 1: 1 / 8, 2: 3 / 8},
    }
    marginal = {v: sum(conditional[x][v] / 2 for x in (0, 1)) for v in values}
    view_mean = {x: sum(v * conditional[x][v] for v in values) for x in (0, 1)}
    task_mean = {x: sum(v**2 * conditional[x][v] for v in values) for x in (0, 1)}
    unconditional_mean = sum(task_mean.values()) / 2
    input_risk = sum(
        conditional[x][v] * (v**2 - task_mean[x])**2 / 2
        for x in (0, 1) for v in values
    )
    constant_risk = sum(marginal[v] * (v**2 - unconditional_mean)**2 for v in values)
    ratios = {x: {v: conditional[x][v] / marginal[v] for v in values} for x in (0, 1)}
    lost_information = sum(
        conditional[x][v] * log(ratios[x][v]) / 2
        for x in (0, 1) for v in values
    )
    tuple_kl = sum(
        comb(k, j) / 2**k * (0.5 + j / k) * log(0.5 + j / k)
        for j in range(k + 1)
    )
    optimal_full_loss = 0.0
    for x in (0, 1):
        for magnitudes in product((1, 2), repeat=k):
            r = [ratios[x][v] for v in magnitudes]
            total = sum(r)
            entropy = -sum((a / total) * log(a / total) for a in r)
            optimal_full_loss += 0.5 * 2**(-k) * (total / k) * entropy
    direct_gap = log(k) - optimal_full_loss
    m = min(r for rv in ratios.values() for r in rv.values())
    M = max(r for rv in ratios.values() for r in rv.values())
    assert k > 2 * M / m
    lower_gap = (1 - 2 * M / (k * m)) * lost_information
    assert view_mean == {0: 0, 1: 0}
    assert abs((constant_risk - input_risk) - 9 / 16) < 1e-12
    assert abs(direct_gap - (lost_information - tuple_kl)) < 1e-12
    assert direct_gap >= lower_gap > 0
    return {
        "conditional_view_means": view_mean,
        "conditional_task_means": task_mean,
        "input_task_risk": input_risk,
        "constant_task_risk": constant_risk,
        "task_gap": constant_risk - input_risk,
        "ratio_bounds": [m, M],
        "K": k,
        "S_KL": lost_information,
        "tuple_KL": tuple_kl,
        "optimal_matching_excess_direct": direct_gap,
        "optimal_matching_excess_S_minus_T": lost_information - tuple_kl,
        "guaranteed_lower_excess": lower_gap,
    }


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2))
