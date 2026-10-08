"""Exact partial-compression contrastive calculation, with no Monte Carlo."""
import itertools
import json
import math


def binary_entropy(p):
    return -p * math.log(p) - (1 - p) * math.log1p(-p)


def calculate(alpha=.2, beta=.1, k=8):
    signs = (-1, 1)
    lost = binary_entropy((1 + alpha) / 2) - sum(
        binary_entropy((1 + alpha + b * beta) / 2) for b in signs) / 2
    retained = math.log(2) - binary_entropy((1 + alpha) / 2)
    tuple_kl = full_loss = encoded_loss = fitted_loss = fitting_kl = 0.0
    for a, b in itertools.product(signs, repeat=2):
        for vs in itertools.product(signs, repeat=k):
            r = [1 + (alpha * a + beta * b) * v for v in vs]
            rh = [1 + alpha * a * v for v in vs]
            gu, gh = sum(r) / k, sum(rh) / k
            base = .25 * 2 ** (-k)
            tuple_kl += base * gu * math.log(gu / gh)
            extra_scores = [math.log(x) + .3 * v for x, v in zip(rh, vs)]
            exp_scores = [math.exp(s) for s in extra_scores]
            score_sum = sum(exp_scores)
            for j in range(k):
                pu, ph = r[j] / (k * gu), rh[j] / (k * gh)
                ps = exp_scores[j] / score_sum
                mass = base * r[j] / k
                full_loss -= mass * math.log(pu)
                encoded_loss -= mass * math.log(ph)
                fitted_loss -= mass * math.log(ps)
                # Average over U; H's tuple distribution emerges after averaging B.
                fitting_kl += base * gh * ph * math.log(ph / ps)
    excess = encoded_loss - full_loss
    assert abs(excess - (lost - tuple_kl)) < 3e-13
    assert abs(fitted_loss - full_loss - excess - fitting_kl) < 3e-13
    m, big_m = 1 - alpha - beta, 1 + alpha + beta
    moment = beta * beta
    assert tuple_kl <= moment / (k * m) + 1e-12
    assert lost >= moment / (2 * big_m) - 1e-12
    factor = 1 / (1 - 2 * big_m / (k * m)) if k > 2 * big_m / m else None
    if factor is not None:
        assert lost <= factor * excess + 1e-12
    risk_full = sum((1-abs(alpha*a+beta*b))/2
                    for a,b in itertools.product(signs, repeat=2))/4
    risk_encoded = (1-abs(alpha))/2
    regret_bound = None if factor is None else min(1, math.sqrt(2*factor*excess))
    if regret_bound is not None:
        assert 0 <= risk_encoded-risk_full <= regret_bound+1e-12
    return dict(alpha=alpha, beta=beta, K=k, retained_information=retained,
                lost_information=lost, tuple_KL=tuple_kl,
                contrastive_excess=excess, full_optimal_loss=full_loss,
                encoded_optimal_loss=encoded_loss, score_fitting_KL=fitting_kl,
                multiplicative_factor=factor,
                upper_bound=None if factor is None else factor * excess,
                anchor_tuple_terms=4 * 2 ** k,
                task_V_full_risk=risk_full, task_V_encoded_risk=risk_encoded,
                task_V_regret=risk_encoded-risk_full, task_regret_bound=regret_bound)


def four_value_decision_implication():
    """Conditional guarantee for a stipulated population excess, not a training run."""
    m, big_m, k, excess_upper = .5, 1.5, 8, .005
    coefficient = 1-2*big_m/(k*m)
    regret_upper = math.sqrt(2*excess_upper/coefficient)
    full_risk = (.25+.25)/2  # Predict whether |V|=2, using U=0 or U=1.
    constant_risk = .5
    lost_constant = .25*math.log(.5)+.75*math.log(1.5)
    constant_excess_lower = coefficient*lost_constant
    assert abs(coefficient-.25)<1e-14
    assert abs(regret_upper-.2)<1e-14
    assert full_risk+regret_upper < constant_risk
    assert excess_upper < constant_excess_lower
    return dict(K=k, m=m, M=big_m, stipulated_population_excess_upper=excess_upper,
                correction_coefficient=coefficient, full_anchor_risk=full_risk,
                constant_decision_risk=constant_risk, regret_upper=regret_upper,
                encoded_risk_upper=full_risk+regret_upper,
                constant_encoder_excess_lower=constant_excess_lower,
                interpretation='Conditional population implication; no trained score is measured.')


if __name__ == '__main__':
    rows = [calculate(.2, .1, k) for k in (2, 4, 8)]
    rows += [calculate(.2, b, 8) for b in (.05, .01)]
    print(json.dumps({'status': 'passed', 'cases': rows,
                     'four_value_decision_implication': four_value_decision_implication()}, indent=2))
