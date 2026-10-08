"""Active-rank and frozen-Jacobian calculations; requires NumPy."""
import json
import numpy as np


def fit(x, v, weights):
    weights = np.asarray(weights) / np.sum(weights)
    s = x.T @ (weights[:, None] * x)
    ev, q = np.linalg.eigh(s)
    invroot = (q * ev**-0.5) @ q.T
    t = v.T @ (weights[:, None] * x)
    u, z, vt = np.linalg.svd(t @ invroot, full_matrices=False)
    return (u[:, :1] * z[:1]) @ vt[:1] @ invroot, s


def loss(x, v, f, weights):
    return float(np.asarray(weights) @ np.sum((v-x@f.T)**2, axis=1) / np.sum(weights))


def main():
    active, reweighted, tangent = [], [], []
    for s in [1., 3.9, 4., 4.1, 9.]:
        x = np.sqrt(2)*np.diag([1, np.sqrt(s)])
        v = np.sqrt(2)*np.diag([2, np.sqrt(s)])
        f, _ = fit(x, v, [1, 1])
        actual = loss(x, v, f, [1, 1])
        assert np.isclose(actual, min(s, 4))
        assert np.isclose(loss(x, v, np.diag([2., 0.]), [1, 1]), s)
        active.append(dict(s=s, ordinary_loss=s, weighted_loss=actual, fitted_matrix=f.tolist()))
    q = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    x = np.array([[1., 1.], [3., -3.]])
    v = np.sqrt(2)*np.diag([2., 3.])
    for omega in [0.1, 4/9, 0.6, 1., 3.]:
        f, s = fit(x, v, [1, omega])
        actual = loss(x, v, f, [1, omega])
        expected = min(8, 18*omega)/(1+omega)
        assert np.isclose(actual, expected)
        if not np.isclose(omega, 4/9):
            target = np.diag([2., 0.]) @ q.T if omega < 4/9 else np.diag([0., 1.]) @ q.T
            assert np.allclose(f, target)
        else:
            for theta in np.linspace(0, 2*np.pi, 37):
                u = np.array([np.cos(theta), np.sin(theta)])
                tied = np.outer(u, u) @ np.diag([2., 1.]) @ q.T
                assert np.isclose(loss(x, v, tied, [1, omega]), expected)
        reweighted.append(dict(omega=omega, minimum_loss=actual, S=s.tolist()))
    x = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    y = x[:, 1]
    for alpha in [0., 0.1, 1., 2.]:
        p0 = np.array([alpha, 1., 0.])
        def predictor(p):
            return p[0]*(x@p[1:])
        eps = 1e-5
        j = np.column_stack([(predictor(p0+eps*e)-predictor(p0-eps*e))/(2*eps) for e in np.eye(3)])
        assert np.allclose(j, np.column_stack([x[:, 0], alpha*x[:, 0], alpha*x[:, 1]]))
        ev, u = np.linalg.eigh(j@j.T/len(x))
        for time in [0., 0.3, 2., 10.]:
            residual = u @ (np.exp(-ev*time)*(u.T@(predictor(p0)-y)))
            expected = alpha**2*np.exp(-2*(1+alpha**2)*time)+np.exp(-2*alpha**2*time)
            assert np.isclose(np.mean(residual**2), expected)
            tangent.append(dict(alpha=alpha, time=time, risk=float(np.mean(residual**2))))
    print(json.dumps(dict(active_rank=active, reweighted=reweighted, tangent_flow=tangent), indent=2))


if __name__ == '__main__':
    main()
