"""Reproduce Ch6's interpolation comparisons and additive-task boundary.

Direct least-squares fits check the formulas; Monte Carlo errors are reported,
not interpreted as proofs. Requires NumPy. No external data or model calls.
"""
import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def calculate(repetitions=12000, seed=1001202606):
    rng = np.random.default_rng(seed)
    # Exhaust all noise signs at one full-row-rank design. The prediction
    # error is computed from a fitted coefficient, without using its formula.
    n, r, a, noise_var = 4, 7, 0.2, 0.45
    H = rng.normal(size=(n, r))
    theta = np.zeros(r)
    theta[0] = np.sqrt(1 - a)
    errors = []
    for signs in itertools.product([-1, 1], repeat=n):
        y = H @ theta + np.sqrt(noise_var) * np.array(signs)
        fitted = np.linalg.lstsq(H, y, rcond=None)[0]
        errors.append(a + np.sum((fitted - theta) ** 2))
    Q = H.T @ np.linalg.solve(H @ H.T, H)
    conditional = (a + np.sum(((np.eye(r) - Q) @ theta) ** 2)
                   + noise_var * np.trace(np.linalg.inv(H @ H.T)))
    discrepancy = abs(np.mean(errors) - conditional)
    assert discrepancy < 1e-10

    # P2(f): independent random representation; same fresh labels across the
    # three fitted heads in each replication. The target is fixed at e_1.
    d, n, sigma2 = 20, 10, 0.25
    target = np.eye(d)[:, 0]
    risks = {name: [] for name in ['random_rank2', 'aligned_rank2', 'raw_rank20']}
    projector_diag = np.zeros(d)
    for _ in range(repetitions):
        random_basis = np.linalg.qr(rng.normal(size=(d, 2)), mode='reduced')[0]
        projector_diag += np.sum(random_basis ** 2, axis=1)
        X = rng.normal(size=(n, d))
        y = X @ target + np.sqrt(sigma2) * rng.normal(size=n)
        for name, basis in [('random_rank2', random_basis),
                            ('aligned_rank2', np.eye(d)[:, :2]),
                            ('raw_rank20', np.eye(d))]:
            fitted = np.linalg.lstsq(X @ basis, y, rcond=None)[0]
            raw_coefficient = basis @ fitted
            risks[name].append(float(np.sum((raw_coefficient - target) ** 2)))
    theory = {'random_rank2': 43 / 35, 'aligned_rank2': 1 / 14, 'raw_rank20': 7 / 9}
    mc = {}
    for name, values in risks.items():
        values = np.array(values)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        mc[name] = {'fitted_risk_mean': mean, 'mc_se': se,
                    'theoretical_mean': theory[name],
                    'standardized_difference': (mean - theory[name]) / se}

    # The block's three observed equations identify additive coefficients.
    # A new interaction column is zero on every training row.
    design = np.array([[1, 0, 0], [1, 1, 0], [2, 0, 1]], dtype=float)
    labels = np.array([2, 5, 3], dtype=float)
    coefficients = np.linalg.solve(design, labels)
    interaction_column = design[:, 1] * design[:, 2]
    query = np.array([3, 1, 1], dtype=float)
    assert np.array_equal(interaction_column, np.zeros(3))
    assert np.allclose(coefficients, [2, 3, -1])
    joint_predictions = [float(query @ coefficients + lam * query[1] * query[2])
                         for lam in [0, 1]]
    table = {'2': str(Fraction(1, 14)), '8': str(Fraction(2)),
             '12': str(Fraction(1, 6) + Fraction(5, 2)),
             '20': str(Fraction(1, 2) + Fraction(5, 18))}
    return {'seed': seed, 'repetitions': repetitions,
            'conditional_noise_enumeration': {'observed_mean': float(np.mean(errors)),
                                             'formula': float(conditional),
                                             'absolute_difference': float(discrepancy)},
            'aligned_width_risk_table': table,
            'head_comparison': mc,
            'random_projector_mean_diagonal': (projector_diag / repetitions).tolist(),
            'composition': {'additive_coefficients': coefficients.tolist(),
                            'training_interaction_column': interaction_column.tolist(),
                            'joint_predictions_lambda_0_and_1': joint_predictions},
            'scope': 'Finite Gaussian heads and a declared additive block; no Transformer experiment.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=12000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = calculate(args.repetitions)
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)
