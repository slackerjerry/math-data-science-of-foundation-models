#!/usr/bin/env python3
"""Exact finite calculations for Chapter 5 and its solutions.

Python standard library only. Run from any working directory.
The assertions compare empirical enumeration, matrix recurrences, closed-form
solutions, and finite differences. They do not replace the proofs in the notes.
Use --json for a machine-readable record of the calculated quantities.
"""
import argparse
import itertools
import json
import math
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def transpose(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    return [[dot(row, col) for col in transpose(b)] for row in a]


def mv(a, x):
    return [dot(row, x) for row in a]


def outer(x, y):
    return [[u * v for v in y] for u in x]


def scale(a, c):
    return [[c * x for x in row] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def eye(d):
    return [[float(i == j) for j in range(d)] for i in range(d)]


def inv2(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) < 1e-14:
        raise ValueError("Singular 2-by-2 matrix")
    return scale([[a[1][1], -a[0][1]], [-a[1][0], a[0][0]]], 1.0 / det)


def eig2(a):
    assert abs(a[0][1] - a[1][0]) < 1e-10
    disc = math.sqrt((a[0][0] - a[1][1]) ** 2 + 4 * a[0][1] ** 2)
    mid = a[0][0] + a[1][1]
    return [(mid - disc) / 2, (mid + disc) / 2]


def flat(x):
    if isinstance(x, (tuple, list)):
        return [v for row in x for v in flat(row)]
    return [float(x)]


def close(a, b, tol=1e-9):
    aa, bb = flat(a), flat(b)
    assert len(aa) == len(bb), (a, b)
    assert all(abs(x - y) <= tol * max(1.0, abs(x), abs(y))
               for x, y in zip(aa, bb)), (a, b)


def moments(b, y):
    n = len(b)
    h = scale(mm(transpose(b), b), 1.0 / n)
    beta = [sum(row[j] * yi for row, yi in zip(b, y)) / n
            for j in range(len(b[0]))]
    return h, beta


def gradient(b, y, w):
    n, d = len(b), len(w)
    return [sum(row[j] * (dot(row, w) - yi)
                for row, yi in zip(b, y)) / n for j in range(d)]


def objective(b, y, w, rho=0.0, penalty=None):
    value = sum((dot(row, w) - yi) ** 2 for row, yi in zip(b, y)) / (2 * len(b))
    penalty = eye(len(w)) if penalty is None else penalty
    return value + rho * dot(w, mv(penalty, w)) / 2


def ridge(b, y, rho, penalty=None):
    h, beta = moments(b, y)
    penalty = eye(len(beta)) if penalty is None else penalty
    return mv(inv2(add(h, scale(penalty, rho))), beta)


def gd(b, y, w, eta, steps, preconditioner=None):
    w = list(w)
    for _ in range(steps):
        g = gradient(b, y, w)
        if preconditioner is not None:
            g = mv(preconditioner, g)
        w = [v - eta * u for v, u in zip(w, g)]
    return w


def sigmoid(a):
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    t = math.exp(a)
    return t / (1.0 + t)


def softplus(a):
    return max(a, 0.0) + math.log1p(math.exp(-abs(a)))


def logistic_loss(b, y, w):
    return sum(softplus(dot(row, w)) - yi * dot(row, w)
               for row, yi in zip(b, y)) / len(b)


def logistic_gradient(b, y, w):
    return [sum(row[j] * (sigmoid(dot(row, w)) - yi)
                for row, yi in zip(b, y)) / len(b) for j in range(len(w))]


def logistic_hessian(b, w):
    d, n = len(w), len(b)
    result = [[0.0] * d for _ in range(d)]
    for row in b:
        p = sigmoid(dot(row, w))
        result = add(result, scale(outer(row, row), p * (1.0 - p) / n))
    return result


def squared_distance(a, b, metric=None):
    diff = [u - v for u, v in zip(a, b)]
    metric = eye(len(diff)) if metric is None else metric
    return dot(diff, mv(metric, diff))


def cosine(a, b):
    return dot(a, b) / math.sqrt(dot(a, a) * dot(b, b))


def recurrence_matrix(eta, batch, without_replacement=False):
    if without_replacement:
        assert 1 <= batch <= 4
        multiplier = (4.0 - batch) / (3.0 * batch)
    else:
        multiplier = 1.0 / batch
    off = 9.0 * eta * eta * multiplier
    return [[(1.0 - eta) ** 2, off], [off, (1.0 - 9.0 * eta) ** 2]]


def apply_steps(matrix, vector, steps):
    result = list(vector)
    for _ in range(steps):
        result = mv(matrix, result)
    return result


def main():
    xs = list(itertools.product((-1.0, 1.0), repeat=2))
    b = [[x1, 3.0 * x2] for x1, x2 in xs]
    y1 = [x1 for x1, _ in xs]
    y2 = [x2 for _, x2 in xs]
    h, beta = moments(b, y1)
    close(h, [[1.0, 0.0], [0.0, 9.0]])
    close(beta, [1.0, 0.0])
    report = {"design_second_moment": h}

    diagonal = [[10.0, 0.0], [0.0, 1.0]]
    shear = [[1.0, 4.0], [0.0, 1.0]]
    comparisons = {}
    for name, a in [("diagonal", diagonal), ("shear", shear),
                    ("mixed", [[2.0, -1.0], [1.0, 1.0]])]:
        z = mm(b, a)
        ai = inv2(a)
        metric = mm(ai, transpose(ai))
        penalty = mm(transpose(a), a)
        w = [0.7, -0.4]
        v = mv(ai, w)
        close(mv(z, v), mv(b, w))
        close(objective(z, y1, v, 1.0, penalty), objective(b, y1, w, 1.0))
        for i, j in itertools.product(range(4), repeat=2):
            close(squared_distance(z[i], z[j], metric), squared_distance(b[i], b[j]))
        wr = ridge(b, y1, 1.0)
        vr = ridge(z, y1, 1.0)
        vt = ridge(z, y1, 1.0, penalty)
        close(mv(a, vt), wr)
        # Compare direct empirical-gradient iterations with transformed updates.
        for t in [1, 2, 5, 20]:
            original = gd(b, y1, w, 0.05, t)
            restored = gd(z, y1, v, 0.05, t, metric)
            close(mv(a, restored), original)
        comparisons[name] = {
            "metric": metric, "penalty": penalty,
            "isotropic_new_coefficient": vr,
            "isotropic_effective_original_coefficient": mv(a, vr),
            "transported_new_coefficient": vt,
            "new_hessian": moments(z, y1)[0],
        }
    close(comparisons["shear"]["isotropic_new_coefficient"], [5/18, 1/9])
    close(comparisons["shear"]["isotropic_effective_original_coefficient"], [13/18, 1/9])
    close(comparisons["diagonal"]["isotropic_effective_original_coefficient"], [100/101, 0])
    query, ca, cb = [1., 3.], [-1., 3.], [1., -3.]
    retrieval = {}
    for name, a in [("original", eye(2)), ("diagonal", diagonal), ("shear", shear)]:
        q, ra, rb = mm([query, ca, cb], a)
        retrieval[name] = {"squared_distances": [squared_distance(q, ra), squared_distance(q, rb)],
                           "cosines": [cosine(q, ra), cosine(q, rb)]}
    close(retrieval["shear"]["squared_distances"], [68, 36])
    close(retrieval["diagonal"]["cosines"], [-91/109, 91/109])
    report["coordinate_comparisons"] = comparisons
    report["retrieval"] = retrieval

    # The fitted-response matrix is computed directly, independently of its trace formula.
    smoothing = scale(mm(mm(b, inv2(add(h, eye(2)))), transpose(b)), 1 / len(b))
    effective = sum(smoothing[i][i] for i in range(4))
    close(effective, 1.4)
    report["ridge_effective_dimension_rho_1"] = effective
    for alpha in [0.0, 0.3, math.pi / 4, math.pi / 2]:
        y = [math.cos(alpha)*x1 + math.sin(alpha)*x2 for x1, x2 in xs]
        for rho in [0.2, 1.0, 3.0]:
            wr = ridge(b, y, rho)
            expected = (rho/(1+rho))**2*math.cos(alpha)**2 + (rho/(9+rho))**2*math.sin(alpha)**2
            close(2*objective(b, y, wr), expected)
        for t in [0, 1, 2, 10]:
            wt = gd(b, y, [0, 0], 0.1, t)
            expected = 0.9**(2*t)*math.cos(alpha)**2 + 0.1**(2*t)*math.sin(alpha)**2
            close(2*objective(b, y, wt), expected)

    # Rank deficiency, a nonzero null-space initialization, and the all-zero design.
    br, yr = [[1., 1.], [-1., -1.]], [1., -1.]
    close(gd(br, yr, [0, 0], 0.1, 200), [0.5, 0.5])
    close(gd(br, yr, [2, -1], 0.1, 200), [2, -1])
    close(gd(br, yr, [2, 0], 0.1, 200), [1.5, -0.5])
    ar = [[2., 0.], [0., 1.]]
    vr = gd(mm(br, ar), yr, [0, 0], 0.1, 200)
    close(vr, [0.4, 0.2])
    close(mv(ar, vr), [0.8, 0.2])
    close(gd([[0., 0.]], [1.], [2., 3.], 1., 5), [2., 3.])
    report["rank_deficient_limits"] = {"original": [0.5, 0.5], "transformed_effective": mv(ar, vr),
                                       "query_predictions": [0., 0.6]}
    close(gd(b, y2, [0, 0], 0.3, 2)[1] - 1/3, -(-1.7)**2/3)

    # Enumerate batches to verify the conditional second moments directly.
    # This uses the empirical gradient for each sampled batch, not T_b to generate the updates.
    e = [0.8, -0.3]
    w = [1.0 + e[0], e[1]]
    for replace in [True, False]:
        for batch in ([1, 2, 4] if replace else [1, 2, 3, 4]):
            batches = (itertools.product(range(4), repeat=batch) if replace
                       else itertools.combinations(range(4), batch))
            updates, grads = [], []
            for indices in batches:
                g = gradient([b[i] for i in indices], [y1[i] for i in indices], w)
                grads.append(g)
                updates.append([e[j] - 0.21*g[j] for j in range(2)])
            avg_g = [sum(g[j] for g in grads)/len(grads) for j in range(2)]
            close(avg_g, mv(h, e))
            avg_squares = [sum(u[j]**2 for u in updates)/len(updates) for j in range(2)]
            close(avg_squares, mv(recurrence_matrix(0.21, batch, not replace), [x*x for x in e]))
            cov = [[sum((g[i]-avg_g[i])*(g[j]-avg_g[j]) for g in grads)/len(grads)
                    for j in range(2)] for i in range(2)]
            factor = 1/batch if replace else (4-batch)/(3*batch)
            close(cov, scale([[e[1]**2, e[0]*e[1]], [e[0]*e[1], e[0]**2]], 9*factor))
    # Enumerate all effective sign paths for several steps, not a Monte Carlo estimate.
    path_errors = [[-1.0, 0.0]]
    for t in range(1, 11):
        next_errors = []
        for err in path_errors:
            for sign in [-1.0, 1.0]:
                next_errors.append([(1-.21)*err[0] - 3*.21*sign*err[1],
                                    -3*.21*sign*err[0] + (1-9*.21)*err[1]])
        path_errors = next_errors
        direct = [sum(err[j]**2 for err in path_errors)/len(path_errors) for j in range(2)]
        close(direct, apply_steps(recurrence_matrix(.21, 1), [1, 0], t))

    table = []
    for t in [0, 5, 20, 50]:
        vt = apply_steps(recurrence_matrix(.21, 1), [1, 0], t)
        table.append({"updates": t, "squared_mean_error": .79**(2*t),
                      "mean_squared_error": sum(vt), "expected_training_loss": (vt[0]+9*vt[1])/2})
    budget = []
    for batch in [1, 2, 4, 8]:
        steps = 40 // batch
        tmat = recurrence_matrix(.21, batch)
        vt = apply_steps(tmat, [1, 0], steps)
        budget.append({"batch": batch, "updates": steps, "expected_F0": (vt[0]+9*vt[1])/2,
                       "mean_squared_error": sum(vt), "moment_spectral_radius": eig2(tmat)[1]})
    assert eig2(recurrence_matrix(.21, 1))[1] > 1
    assert eig2(recurrence_matrix(.21, 2))[1] < 1
    close(sum(apply_steps(recurrence_matrix(.2, 1), [1, 0], 50)), 1.)
    assert sum(apply_steps(recurrence_matrix(.1, 1), [1, 0], 100)) < 1e-8
    report["mean_and_second_moment_table"] = table
    report["budget_40_gradients"] = budget
    report["best_batch_in_budget_comparison"] = min(budget, key=lambda x: x["expected_F0"])["batch"]
    report["sgd_moment_radii"] = {str(batch): eig2(recurrence_matrix(.21, batch))[1] for batch in [1, 2, 4, 8]}
    rng = random.Random(20260929)
    path, err = [], [-1., 0.]
    for t in range(20):
        sign = rng.choice([-1., 1.])
        err = [.79*err[0]-.63*sign*err[1], -.63*sign*err[0]-.89*err[1]]
        path.append(dot(err, err))
    report["one_sample_path_squared_error"] = path

    ylog = [(v+1)/2 for v in y1]
    curvature = {}
    for w in [[0., 0.], [1., 1.]]:
        hg = logistic_hessian(b, w)
        eps = 1e-5
        for j in range(2):
            up, down = w.copy(), w.copy()
            up[j] += eps
            down[j] -= eps
            fd = [(u-v)/(2*eps) for u, v in
                  zip(logistic_gradient(b, ylog, up), logistic_gradient(b, ylog, down))]
            close(fd, [hg[i][j] for i in range(2)], 1e-8)
            loss_fd = (logistic_loss(b, ylog, up)-logistic_loss(b, ylog, down))/(2*eps)
            close(loss_fd, logistic_gradient(b, ylog, w)[j], 1e-8)
        z = mm(b, shear)
        v = mv(inv2(shear), w)
        close(logistic_loss(z, ylog, v), logistic_loss(b, ylog, w))
        close(logistic_hessian(z, v), mm(mm(transpose(shear), hg), shear))
        assert eig2(hg)[0] >= -1e-12
        assert eig2(add(scale(h, .25), scale(hg, -1)))[0] >= -1e-12
        curvature[str(w)] = hg
    lo, hi = 0., .5
    for _ in range(70):
        mid = (lo+hi)/2
        if mid < sigmoid(-mid):
            lo = mid
        else:
            hi = mid
    log_opt = (lo+hi)/2
    close(logistic_gradient(b, ylog, [log_opt, 0]), [-log_opt, 0])
    report["logistic_hessians"] = curvature
    report["ridge_logistic_optimum"] = log_opt

    # Test the stochastic smoothness inequality on all four possible next gradients.
    w, eta, lipschitz = [.4, -.2], .1, 9/4
    full_g = logistic_gradient(b, ylog, w)
    singles = [logistic_gradient([row], [yi], w) for row, yi in zip(b, ylog)]
    noise = sum(dot([gi-gj for gi, gj in zip(g, full_g)],
                    [gi-gj for gi, gj in zip(g, full_g)]) for g in singles)/4
    expected_next = sum(logistic_loss(b, ylog, [v-eta*u for v, u in zip(w, g)]) for g in singles)/4
    upper = logistic_loss(b, ylog, w) - eta*(1-lipschitz*eta/2)*dot(full_g, full_g) + lipschitz*eta**2*noise/2
    assert expected_next <= upper + 1e-12
    report["smooth_descent_example"] = {"expected_next_loss": expected_next, "upper_bound": upper}

    def head_loss(wm, labels):
        total = 0.
        for row, yi in zip(b, labels):
            logits = mm([row], wm)[0]
            maximum = max(logits)
            total += maximum + math.log(sum(math.exp(v-maximum) for v in logits)) - logits[yi]
        return total/len(b)

    wm = [[.2, -.1, .4], [-.3, .5, -.2]]
    direction = [[.4, .2, -.3], [.1, -.2, .6]]
    labels = [0, 1, 2, 1]
    directional_curvature = 0.
    for row in b:
        logits = mm([row], wm)[0]
        terms = [math.exp(z-max(logits)) for z in logits]
        probs = [z/sum(terms) for z in terms]
        delta = mm([row], direction)[0]
        directional_curvature += (dot(probs, [v*v for v in delta])-dot(probs, delta)**2)/4
    eps = 1e-4
    direct_second = (head_loss(add(wm, scale(direction, eps)), labels)
                     - 2*head_loss(wm, labels)
                     + head_loss(add(wm, scale(direction, -eps)), labels))/eps**2
    close(direct_second, directional_curvature, 2e-6)
    shifted = add(wm, [[.7]*3, [-.4]*3])
    close(head_loss(shifted, labels), head_loss(wm, labels))
    nonlinear_second = 2*(softplus(-eps**2)-softplus(0))/eps**2
    close(nonlinear_second, -1., 2e-7)
    report["multiclass_directional_curvature"] = directional_curvature
    report["nonlinear_logit_second_derivative_at_zero"] = nonlinear_second
    report["checks"] = "All finite-calculation and derivative checks passed."
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = main()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["checks"])
        print("GD versus one-example SGD, step size 0.21:")
        print(" steps   squared mean error    mean squared error    expected F0")
        for row in result["mean_and_second_moment_table"]:
            print(f'{row["updates"]:6d}  {row["squared_mean_error"]:19.9g}'
                  f'  {row["mean_squared_error"]:20.9g}  {row["expected_training_loss"]:13.9g}')
        print("Budget: 40 example-gradient evaluations, sampling with replacement")
        print(" batch   updates    expected F0    mean squared coefficient error")
        for row in result["budget_40_gradients"]:
            print(f'{row["batch"]:6d}  {row["updates"]:8d}  {row["expected_F0"]:13.9g}'
                  f'  {row["mean_squared_error"]:30.9g}')
        print("Best batch in this comparison:", result["best_batch_in_budget_comparison"])
        print("Ridge logistic optimum:", result["ridge_logistic_optimum"])
