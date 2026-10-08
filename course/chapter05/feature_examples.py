"""Check exact directional learning and PT-dependent fine-tuning dynamics."""
import json
import math


def rk4(fun, y, dt, steps):
    y = list(y)
    for _ in range(steps):
        a = fun(y)
        b = fun([v + dt * d / 2 for v, d in zip(y, a)])
        c = fun([v + dt * d / 2 for v, d in zip(y, b)])
        d = fun([v + dt * e for v, e in zip(y, c)])
        y = [v + dt * (da + 2 * db + 2 * dc + dd) / 6
             for v, da, db, dc, dd in zip(y, a, b, c, d)]
    return y


def rotation_flow(y):
    a, b1, b2 = y
    return [b2 - a * (b1 * b1 + b2 * b2), -a * a * b1,
            a * (1 - a * b2)]


def exact_rotation(t):
    v = math.cosh(t) ** 2 / (1 + t + math.sinh(2 * t) / 2)
    a = math.sqrt(v)
    return [a, a / math.cosh(t), a * math.tanh(t)]


def task_risk(beta, target):
    return sum((sum(x * (b - t) for x, b, t in zip(xs, beta, target))) ** 2
               for xs in ((-1, -1), (-1, 1), (1, -1), (1, 1))) / 4


def main():
    rotation = []
    for t in (.1, 1, 3):
        state = rk4(rotation_flow, [1, 1, 0], .0005, round(t / .0005))
        exact = exact_rotation(t)
        error = max(abs(x - y) for x, y in zip(state, exact))
        assert error < 1e-10
        a, b1, b2 = exact
        assert abs(a * a - b1 * b1 - b2 * b2) < 1e-14
        norm = b1 * b1 + b2 * b2
        risks = []
        for j in range(2):
            w = (b1, b2)[j] / norm
            risks.append(task_risk([w * b1, w * b2], [int(i == j) for i in range(2)]))
        assert abs(risks[0] - math.tanh(t) ** 2) < 1e-14
        assert abs(risks[1] - 1 / math.cosh(t) ** 2) < 1e-14
        rotation.append(dict(t=t, parameter_error=error, readout_risks=risks))

    w0, v0, gamma, pretrained = 1.2, .7, .8, [2., 0.]
    c = w0 * w0 + v0 * v0
    lam = (w0 * w0 - v0 * v0) / c
    scales = []
    for target in pretrained:
        z = math.asinh(target / c) / 2
        wp, wm = (w0 * math.cosh(z) + v0 * math.sinh(z),
                  w0 * math.cosh(z) - v0 * math.sinh(z))
        vp, vm = (v0 * math.cosh(z) + w0 * math.sinh(z),
                  v0 * math.cosh(z) - w0 * math.sinh(z))
        assert abs(wp * vp - wm * vm - target) < 1e-14
        direct = (wp + wm) ** 2
        formula = (1 + lam) * (c + math.sqrt(c * c + target * target))
        assert abs(direct - formula) < 1e-14
        scales.append(math.sqrt(formula))
    kappas = [s * s + gamma * gamma for s in scales]
    state0 = [value for s in scales for value in (s, gamma, s, gamma)]

    def parameters(y):
        beta = [y[j] * y[j + 1] - y[j + 2] * y[j + 3] for j in (0, 4)]
        g = sum(beta) - 1
        return [v for j in (0, 4) for v in
                (-y[j + 1] * g, -y[j] * g, y[j + 3] * g, y[j + 2] * g)]

    final = rk4(parameters, state0, .0005, 10000)
    beta = [final[j] * final[j + 1] - final[j + 2] * final[j + 3] for j in (0, 4)]
    expected = [v / sum(kappas) for v in kappas]
    assert max(abs(x - y) for x, y in zip(beta, expected)) < 1e-10
    risks = [task_risk(beta, [1, 0]), task_risk(beta, [0, 1])]
    assert risks[0] < .5 < risks[1]
    # A nonzero reset tests the source's compact k expression against the flow.
    printed_kappa = [s * s + gamma * gamma / 2 for s in scales]
    printed_predictor = [v / sum(printed_kappa) for v in printed_kappa]
    discrepancy = max(abs(x - y) for x, y in zip(beta, printed_predictor))
    assert discrepancy > 1e-4
    print(json.dumps(dict(status='passed', rotation=rotation,
        fine_tuning=dict(kappas=kappas, parameter_flow_limit=beta,
                         derived_limit=expected, deployment_risks=risks,
                         compact_source_constant_discrepancy=discrepancy)), indent=2))


if __name__ == '__main__':
    main()
