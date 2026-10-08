"""Fit a pretext representation, freeze it, then fit independent task heads.

No downstream labels enter pretraining. Risk is against the task conditional
mean. The analytic risk is averaged over pretraining corpora; independently
fitted OLS heads provide a separate Monte Carlo comparison.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def fit_rrr(x, v, rank):
    n = len(x)
    gram = x.T @ x / n
    eig, vec = np.linalg.eigh(gram)
    if eig[0] <= 1e-12 * eig[-1]:
        raise ValueError("This reference learner requires a full-rank design")
    inverse_root = (vec / np.sqrt(eig)) @ vec.T
    cross = v.T @ x / n
    c = cross @ inverse_root
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    fitted = ((u[:, :rank] * s[:rank]) @ vt[:rank]) @ inverse_root
    _, _, rows = np.linalg.svd(fitted, full_matrices=True)
    basis = rows[:rank].T
    return fitted, basis, gram, inverse_root


def mean_se(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=930202611)
    parser.add_argument("--repetitions", type=int, default=600)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "linear_results")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("at least two repetitions are needed for standard errors")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    d, m, rank, label_variance, view_variance = 6, 3, 2, 0.25, 0.1
    truth_basis, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    view_basis, _ = np.linalg.qr(rng.normal(size=(m, rank)))
    view_map = view_basis @ np.diag([1.0, 0.7])
    truth = view_map @ truth_basis.T
    task = truth_basis[:, 1]  # A fixed task in the weaker latent direction.
    rows = []
    checks = {"max_square_identity_error": 0., "max_noiseless_error": 0.,
              "min_matrix_bound_slack": float("inf"),
              "min_projection_bound_slack": float("inf")}
    for n_pre in [16, 64, 256, 1024, 4096]:
        pre_losses, pre_excesses, lost = [], [], []
        values = {(n, name): [] for n in [10, 100]
                  for name in ["learned", "random", "oracle", "raw"]}
        for _ in range(args.repetitions):
            x = rng.normal(size=(n_pre, d))
            noise = np.sqrt(view_variance) * rng.normal(size=(n_pre, m))
            v = x @ truth.T + noise
            fitted, basis, gram, inverse_root = fit_rrr(x, v, rank)
            random_basis, _ = np.linalg.qr(rng.normal(size=(d, rank)))
            error = np.linalg.norm(fitted-truth, "fro") ** 2
            k = noise.T @ x @ inverse_root / n_pre
            bound = 4*np.linalg.norm(k, "fro")**2 / np.linalg.eigvalsh(gram)[0]
            miss = np.linalg.norm(task-basis @ (basis.T @ task))**2
            checks["min_matrix_bound_slack"] = min(checks["min_matrix_bound_slack"], bound-error)
            checks["min_projection_bound_slack"] = min(checks["min_projection_bound_slack"], error/0.7**2-miss)
            ls = np.linalg.solve(gram, (v.T @ x/n_pre).T).T
            empirical = np.linalg.norm(v-x@fitted.T, "fro")**2/n_pre
            identity = np.linalg.norm(v-x@ls.T, "fro")**2/n_pre
            identity += np.trace((fitted-ls)@gram@(fitted-ls).T)
            checks["max_square_identity_error"] = max(checks["max_square_identity_error"], abs(empirical-identity))
            clean, _, _, _ = fit_rrr(x, x@truth.T, rank)
            checks["max_noiseless_error"] = max(checks["max_noiseless_error"], float(np.linalg.norm(clean-truth)))
            pre_losses.append(empirical)
            pre_excesses.append(error)
            lost.append(miss)
            # The same fresh labeled sample is paired across representations.
            for n_label in [10, 100]:
                x_label = rng.normal(size=(n_label, d))
                y_label = x_label @ task + np.sqrt(label_variance)*rng.normal(size=n_label)
                for name, rep in [("learned", basis), ("random", random_basis),
                                  ("oracle", truth_basis), ("raw", np.eye(d))]:
                    omitted = np.linalg.norm(task-rep@(rep.T@task))**2
                    width = rep.shape[1]
                    predicted = omitted + width*(label_variance+omitted)/(n_label-width-1)
                    head = np.linalg.lstsq(x_label@rep, y_label, rcond=None)[0]
                    actual = np.linalg.norm(rep@head-task)**2
                    values[(n_label, name)].append([predicted, actual, actual-predicted])
        for (n_label, name), samples in values.items():
            a = np.asarray(samples)
            analytic, analytic_se = mean_se(a[:, 0])
            fitted_mean, fitted_se = mean_se(a[:, 1])
            diff, diff_se = mean_se(a[:, 2])
            row = dict(pretraining_pairs=n_pre, labels=n_label, representation=name,
                       analytic_risk=analytic, analytic_se=analytic_se,
                       fitted_head_risk=fitted_mean, fitted_head_se=fitted_se,
                       difference=diff, difference_se=diff_se)
            if name == "learned":
                row.update(empirical_pretext_loss=mean_se(pre_losses)[0],
                           population_pretext_excess=mean_se(pre_excesses)[0],
                           missed_task_direction=mean_se(lost)[0])
            rows.append(row)
    assert checks["max_square_identity_error"] < 1e-10
    assert checks["max_noiseless_error"] < 1e-10
    assert checks["min_matrix_bound_slack"] > -1e-10
    assert checks["min_projection_bound_slack"] > -1e-10
    config = {"seed": args.seed, "repetitions": args.repetitions, "d": d, "m": m,
              "rank": rank, "view_singular_values": [1., .7], "view_noise_covariance": "0.1 I_3",
              "label_noise_variance": label_variance, "risk": "fresh conditional mean",
              "numpy_version": np.__version__, "no_intercept": True,
              "true_basis": truth_basis.tolist(), "view_map": view_map.tolist()}
    (out / "results.json").write_text(json.dumps({"configuration": config, "checks": checks, "rows": rows}, indent=2)+"\n")
    with (out / "risk_comparison.csv").open("w", newline="") as stream:
        names = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    colors = dict(learned="#245978", random="#ab602d", oracle="#2f785a", raw="#646464")
    for ax, labels in zip(axes, [10, 100]):
        for name in ["random", "raw", "learned", "oracle"]:
            selected = [r for r in rows if r["labels"] == labels and r["representation"] == name]
            ns = np.array([r["pretraining_pairs"] for r in selected])
            means = np.array([r["analytic_risk"] for r in selected])
            ses = np.array([r["analytic_se"] for r in selected])
            ax.plot(ns, means, label=name, color=colors[name], marker="o", ms=3)
            ax.fill_between(ns, np.maximum(means-2*ses, 1e-8), means+2*ses, color=colors[name], alpha=.13)
        ax.set(xscale="log", yscale="log", xlabel="Independent pretext pairs", title=f"{labels} task labels")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Conditional-mean prediction risk")
    axes[1].legend(fontsize=8)
    fig.savefig(out/"linear_transfer.pdf")
    fig.savefig(out/"linear_transfer.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"checks": checks, "N4096": [r for r in rows if r["pretraining_pairs"]==4096]}, indent=2))


if __name__ == "__main__":
    main()
