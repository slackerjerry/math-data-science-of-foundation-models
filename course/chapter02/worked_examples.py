"""Reproduce Chapter 2's finite calculations using only the standard library.

Rational weights are exact. Logarithms are numerical. The script evaluates the weighted tables used in the chapter and exercises.
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import argparse
import json
import math


def neglog(p):
    return math.inf if p == 0 else -math.log(float(p))


def kl(p, q):
    return sum(float(v)*(neglog(q.get(y, 0))-neglog(v))
               for y, v in p.items() if v > 0)


def entropy(p):
    return sum(float(v)*neglog(v) for v in p.values() if v > 0)


def normalize(weights):
    total = sum(weights.values())
    assert total > 0
    return {k: v/total for k, v in weights.items()}


def fit(counts, prior=None, strength=F(0)):
    """Binary row fitting; None marks an unidentified zero-count row."""
    fitted = {}
    for c, row in counts.items():
        n = sum(row.values())
        fitted[c] = None if n == 0 and strength == 0 else {
            y: (row.get(y, 0)+strength*(prior[c][y] if prior else 0))
            /(n+strength) for y in (0, 1)}
    return fitted


def log_risk(joint, q):
    return sum(float(mass)*neglog(q[c][y])
               for (c, y), mass in joint.items() if mass > 0)


def joint_from_counts(counts):
    return normalize({(c, y): n for c, row in counts.items() for y, n in row.items()})


def decision_risk(joint, q, fp=F(1), fn=F(1)):
    threshold = fp/(fp+fn)
    risk = F(0)
    for (c, y), mass in joint.items():
        action = int(q[c][1] > threshold)
        risk += mass*(fp if action == 1 and y == 0
                      else fn if action == 0 and y == 1 else 0)
    return risk


def close(a, b):
    assert abs(a-b) < 1e-12, (a, b)


def main(positive_weight=F(1)):
    if positive_weight <= 0:
        raise ValueError('positive_weight must be strictly positive')
    report = {}
    copy_joint = {(x, y): F(3 if x == y else 1, 8)
                  for x, y in product((0, 1), repeat=2)}
    copy_rows = {(): {0: F(1, 2), 1: F(1, 2)},
                 (0,): {0: F(3, 4), 1: F(1, 4)},
                 (1,): {0: F(1, 4), 1: F(3, 4)}}
    copy_loss = sum(float(p)*(neglog(copy_rows[()][x])
                    + neglog(copy_rows[(x,)][y]))/2
                    for (x, y), p in copy_joint.items())
    close(copy_loss, entropy(copy_joint)/2)
    close(copy_loss, (math.log(2)+entropy({0: F(3, 4), 1: F(1, 4)}))/2)
    report['copy_experiment'] = {'joint': list(map(float, copy_joint.values())),
                                 'average_position_loss': copy_loss}
    counts = {'a': {1: F(3), 0: F(1)}, 'b': {1: F(1), 0: F(3)}}
    joint = joint_from_counts(counts)
    q = fit(counts)
    assert (q['a'][1], q['b'][1]) == (F(3, 4), F(1, 4))
    loss = log_risk(joint, q)
    close(loss, entropy({1: F(3, 4), 0: F(1, 4)}))
    report['empirical_fit'] = {'a': .75, 'b': .25, 'log_loss': loss}
    pooled = {c: {0: F(1, 2), 1: F(1, 2)} for c in q}
    assert decision_risk(joint, q, fp=F(2)) == F(3, 8)
    assert decision_risk(joint, pooled, fp=F(2)) == F(1, 2)
    report['copy_decision_costs'] = {'conditional': '3/8', 'pooled': '1/2'}
    unseen = joint_from_counts({'c0': {1: F(3), 0: F(1)}})
    report['body_unseen_context_losses'] = [
        log_risk(unseen, {'c0': {1: v, 0: 1-v}}) for v in (F(1, 4), F(3, 4))]
    table = []
    for p1 in (F(1, 2), F(1, 5)):
        row = []
        for s1 in (F(1, 2), F(1, 5)):
            p, s = {1: p1, 0: 1-p1}, {1: s1, 0: 1-s1}
            value = sum(float(p[y])*neglog(s[y]) for y in p)
            close(value, entropy(p)+kl(p, s))
            row.append(value)
        table.append(row)
    report['document_and_token_evaluation'] = table
    heldout = joint_from_counts({'a': {1: F(1), 0: F(3)},
                                'b': {1: F(2), 0: F(2)},
                                'c0': {1: F(3), 0: F(1)}})
    variants = []
    for extra, error, cost in [(F(1, 4), F(2, 3), F(11, 12)),
                               (F(3, 4), F(1, 2), F(10, 12))]:
        model = dict(q, c0={1: extra, 0: 1-extra})
        close(log_risk(joint, model), loss)
        assert decision_risk(heldout, model) == error
        assert decision_risk(heldout, model, fp=F(2)) == cost
        variants.append({'c0_probability': float(extra),
                         'log_loss': log_risk(heldout, model),
                         'accuracy': float(1-error), 'cost': str(cost)})
    report['heldout_extensions'] = variants
    changed_counts = {c: {1: row[1]*positive_weight, 0: row[0]}
                      for c, row in counts.items()}
    changed_model = fit(changed_counts)
    # Keep the unobserved row fixed by an explicit extension rule.
    changed_model['c0'] = {0: F(1, 2), 1: F(1, 2)}
    report['weight_experiment'] = {
        'positive_weight': str(positive_weight),
        'fitted_positive_probabilities': {c: str(row[1]) for c, row in changed_model.items()},
        'changed_objective_log_loss': log_risk(joint_from_counts(changed_counts), changed_model),
        'common_original_train_log_loss': log_risk(joint, changed_model),
        'fixed_heldout_log_loss': log_risk(heldout, changed_model),
        'fixed_heldout_cost_fp2_fn1': str(decision_risk(heldout, changed_model, fp=F(2)))}
    report['calibration_table_log_losses'] = [
        -.75*math.log(s)-.25*math.log(1-s) for s in (.75, .99, .5)]
    # Nonuniform three-bit sequence distribution and a different model.
    seqs = list(product((0, 1), repeat=3))
    pseq = dict(zip(seqs, [F(n, 20) for n in [1, 2, 3, 4, 4, 3, 2, 1]]))
    pairs, context_mass = defaultdict(F), defaultdict(F)
    for x, mass in pseq.items():
        for t in range(3):
            pairs[(x[:t], x[t])] += mass/3
            context_mass[x[:t]] += mass/3
    target = {c: {y: pairs[(c, y)]/mass for y in (0, 1)}
              for c, mass in context_mass.items()}
    model = {c: {1: F(1+sum(c), 2+len(c)), 0: 1-F(1+sum(c), 2+len(c))}
             for c in context_mass}
    qseq = {x: math.prod(model[x[:t]][x[t]] for t in range(3)) for x in seqs}
    assert sum(qseq.values()) == 1
    delta = sum(float(mass)*kl(target[c], model[c]) for c, mass in context_mass.items())
    close(kl(pseq, qseq), 3*delta)
    close(log_risk(pairs, model)-log_risk(pairs, target), delta)
    report['sequence_chain'] = {'sequence_KL': kl(pseq, qseq), 'T_times_delta': 3*delta}
    full = defaultdict(F)
    for old, recent, noise in product((0, 1), repeat=3):
        full[((old, recent), old ^ noise)] += F(1, 4)*(F(1, 4) if noise else F(3, 4))
    full_counts = {c: {y: full[(c, y)] for y in (0, 1)} for c in product((0, 1), repeat=2)}
    full_q = fit(full_counts)
    coarse_q = {c: {0: F(1, 2), 1: F(1, 2)} for c in full_counts}
    gap = log_risk(full, coarse_q)-log_risk(full, full_q)
    close(gap, math.log(2)-entropy({0: F(3, 4), 1: F(1, 4)}))
    report['context_information_gap'] = gap
    prior_context, cond = {'a': F(1, 4), 'b': F(3, 4)}, {'a': F(9, 10), 'b': F(1, 10)}
    new_context = normalize({c: prior_context[c]*(9 if c == 'a' else 1) for c in cond})
    assert sum(prior_context[c]*cond[c] for c in cond) == F(3, 10)
    assert sum(new_context[c]*cond[c] for c in cond) == F(7, 10)
    n, strength, p = 4, F(2), F(3, 5)
    prior = {0: F(1, 4), 1: F(3, 4)}
    mle_mean, smooth_mean = 0., 0.
    bound = math.log(float((n+strength)/(strength*min(prior.values()))))
    for k in range(n+1):
        chance = F(math.comb(n, k))*p**k*(1-p)**(n-k)
        mle = {1: F(k, n), 0: F(n-k, n)}
        smooth = fit({'c': {1: F(k), 0: F(n-k)}}, {'c': prior}, strength)['c']
        mle_risk = float(p)*neglog(mle[1])+float(1-p)*neglog(mle[0])
        smooth_risk = float(p)*neglog(smooth[1])+float(1-p)*neglog(smooth[0])
        assert smooth_risk <= bound
        mle_mean += float(chance)*mle_risk
        smooth_mean += float(chance)*smooth_risk
    assert math.isinf(mle_mean) and math.isfinite(smooth_mean)
    report['smoothing'] = {'unsmoothed_expected_risk': 'infinity',
                            'smoothed_expected_risk': smooth_mean, 'uniform_bound': bound}
    for p, s in product([F(0), F(1, 5), F(1, 2), F(4, 5), F(1)], repeat=2):
        divergence = kl({1: p, 0: 1-p}, {1: s, 0: 1-s})
        assert divergence+1e-12 >= 2*float(p-s)**2
        for fp, fn in [(F(1), F(1)), (F(4), F(1))]:
            costs = [fn*p, fp*(1-p)]
            regret = costs[int(s > fp/(fp+fn))]-min(costs)
            B = float(max(fp, fn))
            assert float(regret) <= min(B, B*math.sqrt(2*divergence))+1e-12
    rank_losses = [-.6*math.log(float(s))-.4*math.log(float(1-s))
                   for s in (F(49, 100), F(99, 100))]
    assert rank_losses[0] < rank_losses[1]
    report['loss_ranking_counterexample'] = {'log_losses': rank_losses, 'errors': [.6, .4]}
    for paired in [{(0, 0): F(1, 2), (1, 1): F(1, 2)},
                   {(0, 1): F(1, 2), (1, 0): F(1, 2)}]:
        marginals = [{y: sum(mass for pair, mass in paired.items() if pair[j] == y)
                      for y in (0, 1)} for j in (0, 1)]
        close(sum(entropy(p) for p in marginals)-entropy(paired), math.log(2))
    report['masked_joint_gap'] = math.log(2)
    report['verification_scope'] = 'Finite calculations supplement the written proofs.'
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positive-weight', default='1',
                        help='positive rational weight, e.g. 2 or 3/2')
    args = parser.parse_args()
    main(F(args.positive_weight))
