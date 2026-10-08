#!/usr/bin/env python3
"""Finite calculations for Chapter 4 P6 and a supplementary binary matching example.
For the current finite-candidate P7, run finite_candidates_examples.py.

Run with --json for machine-readable output. Only the standard library is used.
This contains worked answers; try the problem first, then compare the calculation.
"""
import argparse
import itertools
import json
import math


def close(a, b, atol=2e-11):
    assert abs(a - b) <= atol * max(1.0, abs(a), abs(b)), (a, b)


def quadratic_check(joint, groups, score, target):
    nu, nv = len(joint), len(joint[0])
    pu = [sum(row) for row in joint]
    rho = [sum(joint[u][v] for u in range(nu)) for v in range(nv)]
    close(sum(pu), 1)
    assert min(pu + rho) > 0
    gh = sorted(set(groups))
    ph = {h: sum(pu[u] for u in range(nu) if groups[u] == h) for h in gh}
    rh = {h: [sum(joint[u][v] for u in range(nu) if groups[u] == h)
              / ph[h] / rho[v] for v in range(nv)] for h in gh}
    r = [[joint[u][v] / pu[u] / rho[v] for v in range(nv)] for u in range(nu)]
    mean_score = {h: sum(rho[v] * score[h][v] for v in range(nv)) for h in gh}
    quad, i2, i2h, suff, fit, paired_s = 0., 0., 0., 0., 0., 0.
    kl, klh, sk, decoder_kl, full_kl = 0., 0., 0., 0., 0.
    for u in range(nu):
        h = groups[u]
        normalizer = sum(rho[v] * math.exp(score[h][v]) for v in range(nv))
        for v in range(nv):
            w = pu[u] * rho[v]
            s = score[h][v]
            t = s - mean_score[h]
            quad += w * (.5 * t*t + s)
            paired_s += joint[u][v] * s
            i2 += .5*w*(r[u][v]-1)**2
            i2h += .5*w*(rh[h][v]-1)**2
            suff += .5*w*(r[u][v]-rh[h][v])**2
            fit += .5*w*(t-(rh[h][v]-1))**2
            q = rho[v]*math.exp(s)/normalizer
            p = joint[u][v]/pu[u]
            hp = rho[v]*rh[h][v]
            if p > 0:
                kl += joint[u][v]*math.log(r[u][v])
                sk += joint[u][v]*math.log(p/hp)
                full_kl += joint[u][v]*math.log(p/q)
            if hp > 0:
                klh += pu[u]*hp*math.log(rh[h][v])
                decoder_kl += pu[u]*hp*math.log(hp/q)
    quad -= paired_s
    close(quad+i2, suff+fit)
    close(i2-i2h, suff)
    close(kl-klh, sk)
    close(full_kl, sk+decoder_kl)
    means_u = [sum(joint[u][v]/pu[u]*target[v] for v in range(nv)) for u in range(nu)]
    means_h = {h: sum(rho[v]*rh[h][v]*target[v] for v in range(nv)) for h in gh}
    risk_u = sum(joint[u][v]*(target[v]-means_u[u])**2 for u in range(nu) for v in range(nv))
    risk_h = sum(joint[u][v]*(target[v]-means_h[groups[u]])**2 for u in range(nu) for v in range(nv))
    amean = sum(rho[v]*target[v] for v in range(nv))
    avar = sum(rho[v]*(target[v]-amean)**2 for v in range(nv))
    bound = 2*avar*suff
    assert risk_h-risk_u <= bound+2e-11
    return dict(quadratic_excess=quad+i2, sufficiency_chi2=suff, score_error=fit,
                sufficiency_kl=sk, task_risk_u=risk_u, task_risk_h=risk_h,
                task_bound=bound)


def info(alpha):
    return .5*((1+alpha)*math.log1p(alpha)+(1-alpha)*math.log1p(-alpha))


def finite_sum(alpha, k):
    q = (1+alpha)/2
    terms = []
    for n in range(k):
        mass = math.comb(k-1, n)/2**(k-1)
        terms.append(mass*(q*math.log1p(alpha*(2*(n+1)-k)/k)
                           +(1-q)*math.log1p(alpha*(2*n-k)/k)))
    j = math.fsum(terms)
    i = info(alpha)
    delta = i-j
    assert -2e-12 <= j <= alpha**2/k+2e-12
    assert delta <= i+2e-12
    if k == 2:
        close(delta, i/2)
    else:
        assert i <= k/(k-2)*delta+2e-12
    return dict(alpha=alpha, k=k, information=i, tuple_kl=j,
                contrastive_excess=delta, ratio=delta/i)


def tuple_enumeration(alpha, k):
    # Fix U=+1; explicitly average over the random positive index.
    loss, tuple_kl = 0., 0.
    base = 2**(-k)
    mass_total = 0.
    for vals in itertools.product((-1, 1), repeat=k):
        ratios = [1+alpha*v for v in vals]
        den = sum(ratios)
        masses = [base*ratios[j]/k for j in range(k)]
        total = sum(masses)
        mass_total += total
        for j in range(k):
            loss -= masses[j]*math.log(ratios[j]/den)
        tuple_kl += total*math.log(total/base)
    close(mass_total, 1)
    return math.log(k)-loss, tuple_kl


def run():
    # Nonuniform marginals, nontrivial compression, imperfect restricted score.
    joint = [[.08,.02,.10,.05],[.04,.16,.05,.10],[.06,.04,.12,.18]]
    q1 = quadratic_check(joint, [0,0,1], {0:[.2,-.4,.8,.1],1:[-.1,.5,.3,-.8]}, [-2,1,3,0])
    q2 = quadratic_check([[0,.25,.25,0],[.25,0,0,.25]], [0,0], {0:[0,0,0,0]}, [4,1,1,4])
    close(q2['sufficiency_chi2'], .5)
    close(q2['task_risk_u'], 0)
    close(q2['task_risk_h'], 2.25)
    close(q2['task_bound'], 2.25)
    rows = [finite_sum(a,k) for a in (.2,.5,.9) for k in (2,3,8,32)]
    enum_cases = 0
    for a in (.02,.2,.5,.9,.999):
        for k in (2,3,4,5,8):
            delta,j = tuple_enumeration(a,k)
            row = finite_sum(a,k)
            close(delta,row['contrastive_excess'])
            close(j,row['tuple_kl'])
            enum_cases += 1
    for n in range(-200,201):
        z = n/100
        assert abs(math.tanh(z)-z) <= abs(z)**3/3+2e-15
    return dict(status='passed', quadratic_nonuniform=q1, variance_task=q2,
                finite_negative_table=rows, exhaustive_tuple_settings=enum_cases,
                tanh_grid_points=401)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    out = run()
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print('All finite Chapter 4 research checks passed.')
        print('P7: alpha, K, contrastive excess / lost information')
        for row in out['finite_negative_table']:
            print(f"{row['alpha']:.1f} {row['k']:2d} {row['ratio']:.8f}")
