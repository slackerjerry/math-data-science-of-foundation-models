"""Exact finite distributions behind Chapter 2's masking and decoding examples.

Run from the source root with: python3 chapter02/generation_examples.py
Fractions verify transition probabilities. Logarithms use floating point.
"""
from fractions import Fraction as F
from itertools import product
import json
import math

BITS = tuple(product((0, 1), repeat=2))


def full_conditionals(joint):
    result = []
    for j in (0, 1):
        row = {}
        for other in (0, 1):
            denominator = sum(p for x, p in joint.items() if x[1-j] == other)
            row[other] = sum(p for x, p in joint.items()
                             if x[1-j] == other and x[j] == 1) / denominator
        result.append(row)
    return result


def coordinate_update(joint, tables, j):
    out = {x: F(0) for x in BITS}
    for x, mass in joint.items():
        probability = tables[j][x[1-j]]
        for bit in (0, 1):
            y = list(x)
            y[j] = bit
            out[tuple(y)] += mass * (probability if bit else 1-probability)
    return out


def sweep(joint, tables, order):
    for j in order:
        joint = coordinate_update(joint, tables, j)
    return joint


def parallel_update(joint, tables):
    out = {x: F(0) for x in BITS}
    for old, mass in joint.items():
        p, q = tables[0][old[1]], tables[1][old[0]]
        for a, b in BITS:
            out[(a, b)] += mass*(p if a else 1-p)*(q if b else 1-q)
    return out


def conditional_entropy(records, context, target):
    counts = {}
    for record, mass in records.items():
        c, y = context(record), target(record)
        counts.setdefault(c, {})
        counts[c][y] = counts[c].get(y, F(0)) + mass
    return -sum(float(mass)*math.log(float(mass/sum(row.values())))
                for row in counts.values() for mass in row.values() if mass)


def close(a, b):
    assert abs(a-b) < 1e-12, (a, b)


def run():
    positive = dict(zip(BITS, map(F, ("0.45", "0.05", "0.05", "0.45"))))
    negative = dict(zip(BITS, map(F, ("0.05", "0.45", "0.45", "0.05"))))
    for joint in (positive, negative):
        assert sum(joint.values()) == 1
        assert all(sum(p for x, p in joint.items() if x[j]) == F(1, 2)
                   for j in (0, 1))
    compatible = full_conditionals(positive)
    assert compatible == [{0: F(1, 10), 1: F(9, 10)}]*2
    for order in ((0, 1), (1, 0)):
        assert sweep(positive, compatible, order) == positive
    parallel = parallel_update(positive, compatible)
    equality = parallel[(0, 0)] + parallel[(1, 1)]
    assert equality == F(189, 250)  # 0.756, not the original 0.9

    incompatible = [{0: F(1, 10), 1: F(9, 10)},
                    {0: F(1, 10), 1: F(1, 10)}]
    stationary_12 = dict(zip(BITS, map(F, ("0.738", "0.082", "0.162", "0.018"))))
    stationary_21 = dict(zip(BITS, map(F, ("0.81", "0.01", "0.09", "0.09"))))
    assert sweep(stationary_12, incompatible, (0, 1)) == stationary_12
    assert sweep(stationary_21, incompatible, (1, 0)) == stationary_21
    assert full_conditionals(stationary_12) != incompatible
    assert full_conditionals(stationary_21)[1] == {0: F(1, 82), 1: F(1, 2)}
    assert stationary_12 != stationary_21

    q = {(): {"a": F(3, 5), "b": F(2, 5)},
         ("a",): {"a": F(1, 2), "b": F(1, 2)},
         ("b",): {"a": F(99, 100), "b": F(1, 100)}}
    seqs = tuple(product("ab", repeat=2))
    joint = {x: q[()][x[0]]*q[(x[0],)][x[1]] for x in seqs}
    reference = dict(zip(seqs, map(F, ("0.3", "0.3", "0.396", "0.004"))))
    assert joint == reference
    powered = {x: p*p for x, p in joint.items()}
    global_normalizer = sum(powered.values())
    global_tempered = {x: v/global_normalizer for x, v in powered.items()}
    local_q = {c: {v: p*p/sum(z*z for z in row.values()) for v, p in row.items()}
               for c, row in q.items()}
    local_tempered = {x: local_q[()][x[0]]*local_q[(x[0],)][x[1]] for x in seqs}
    assert sum(local_tempered.values()) == sum(global_tempered.values()) == 1
    assert global_tempered[("b", "a")] > global_tempered[("a", "a")]
    assert local_tempered[("a", "a")] > local_tempered[("b", "a")]

    def continuation(prefix):
        if len(prefix) == 2:
            return F(1)
        return sum(p*p*continuation(prefix+(v,)) for v, p in q[prefix].items())

    assert continuation(()) == global_normalizer
    assert continuation(("a",)) == F(1, 2)
    assert continuation(("b",)) == F(4901, 5000)
    global_first_a = sum(p for x, p in global_tempered.items() if x[0] == "a")
    assert global_first_a == F(18, 100)/F("0.336832")
    assert local_q[()]["a"] == F(9, 13)
    for x in seqs:
        induced = F(1)
        for t in range(2):
            prefix = x[:t]
            induced *= q[prefix][x[t]]**2*continuation(x[:t+1])/continuation(prefix)
        assert induced == global_tempered[x]
    sampling_match = sum(p*p for p in joint.values())
    assert sampling_match == F("0.336832")
    hamming = {a: sum(p*sum(ai != xi for ai, xi in zip(a, x))
                      for x, p in joint.items()) for a in seqs}
    assert min(hamming, key=hamming.get) == ("a", "a")
    assert hamming[("a", "a")] == F("0.704")
    assert hamming[("b", "a")] == F("0.904")

    # Problem 6(a)--(c): evaluate criteria separately from their KL expressions.
    def entropy(distribution):
        return -sum(float(p)*math.log(float(p))
                    for p in distribution.values() if p)

    def kl(first, second):
        return sum(float(p)*math.log(float(p/second[x]))
                   for x, p in first.items() if p)

    def criterion(distribution):
        return sum(float(p)*math.log(float(joint[x]))
                   for x, p in distribution.items()) + 0.5*entropy(distribution)

    path_sums = {
        x: sum(math.log(float(sum(p*p for p in q[x[:t]].values())))
               for t in range(2)) for x in seqs}
    point_mass = {x: F(x == ("b", "a")) for x in seqs}
    uniform = {x: F(1, 4) for x in seqs}
    for candidate in (local_tempered, global_tempered, point_mass, uniform):
        close(criterion(candidate), 0.5*math.log(float(global_normalizer))
              - 0.5*kl(candidate, global_tempered))
        local_criterion = criterion(candidate) - 0.5*sum(
            float(p)*path_sums[x] for x, p in candidate.items())
        close(local_criterion, -0.5*kl(candidate, local_tempered))
    variational_gap = criterion(global_tempered)-criterion(local_tempered)
    close(variational_gap, 0.5*kl(local_tempered, global_tempered))
    assert variational_gap > 0

    # The printed source objective without log(epsilon) fails at the local rule.
    epsilon_aa = 1/(F("0.52")*F("0.5"))
    epsilon_ba = 1/(F("0.52")*F("0.9802"))
    assert epsilon_aa > epsilon_ba > 1
    incorrect_directional_derivative = float(epsilon_aa-epsilon_ba) - math.log(
        float(epsilon_aa/epsilon_ba))
    assert incorrect_directional_derivative > 0

    # Problem 6(e): local ties create unequal probabilities of greedy paths.
    varied_q = {(): {"a": F(1, 2), "b": F(1, 2)},
                ("a",): {"a": F(1, 2), "b": F(1, 2)},
                ("b",): {"a": F(3, 4), "b": F(1, 4)}}
    varied_joint = {x: varied_q[()][x[0]]*varied_q[x[:1]][x[1]] for x in seqs}
    local_limit = {}
    for x in seqs:
        mass = F(1)
        for t in range(2):
            row = varied_q[x[:t]]
            modes = [v for v, p in row.items() if p == max(row.values())]
            mass *= F(x[t] in modes, len(modes))
        local_limit[x] = mass
    seq_modes = [x for x, p in varied_joint.items() if p == max(varied_joint.values())]
    global_limit = {x: F(x in seq_modes, len(seq_modes)) for x in seqs}
    assert list(local_limit.values()) == [F(1, 4), F(1, 4), F(1, 2), F(0)]
    assert list(global_limit.values()) == [F(0), F(0), F(1), F(0)]

    def power_normalize(row, tau):
        scores = {v: math.log(float(p))/tau for v, p in row.items()}
        maximum = max(scores.values())
        weights = {v: math.exp(s-maximum) for v, s in scores.items()}
        total = sum(weights.values())
        return {v: w/total for v, w in weights.items()}

    small_temperature = {}
    for tau in (1.0, 0.5, 0.1, 0.02):
        rows = {c: power_normalize(row, tau) for c, row in varied_q.items()}
        local = {x: rows[()][x[0]]*rows[x[:1]][x[1]] for x in seqs}
        global_ = power_normalize(varied_joint, tau)
        small_temperature[str(tau)] = {
            "local": list(local.values()), "global": list(global_.values())}
        if tau == 1:
            for x in seqs:
                close(local[x], float(varied_joint[x]))
                close(global_[x], float(varied_joint[x]))
        if tau == 0.02:
            assert max(abs(local[x]-float(local_limit[x])) for x in seqs) < 1e-8
            assert max(abs(global_[x]-float(global_limit[x])) for x in seqs) < 1e-8

    # Problem 6(f): independent generated/reference sequences, evaluated directly.
    first_token_errors = {}
    for name, sampler in (("local", local_tempered), ("global", global_tempered)):
        error = sum(sa*px*F(a[0] != x[0])
                    for a, sa in sampler.items() for x, px in joint.items())
        s = sum(mass for x, mass in sampler.items() if x[0] == "a")
        assert error == F(3, 5)-F(1, 5)*s
        first_token_errors[name] = error
    assert first_token_errors["local"] == F(6, 13)
    assert first_token_errors["global"] > first_token_errors["local"]

    # Problem 4(e): a context bit survives only in target dependence.
    records = {(o, r, o, o ^ r): F(1, 4) for o, r in BITS}
    truncation = {}
    for name, context in (("full", lambda z: z[:2]),
                          ("recent", lambda z: z[1]), ("none", lambda z: ())):
        hj = conditional_entropy(records, context, lambda z: z[2:])
        hs = sum(conditional_entropy(records, context, lambda z, j=j: z[j])
                 for j in (2, 3))
        truncation[name] = {"joint": hj, "separate_sum": hs}
    for name, factors in {"full": (0, 0), "recent": (1, 2), "none": (2, 2)}.items():
        for value, factor in zip(truncation[name].values(), factors):
            close(value, factor*math.log(2))
    return {
        "parallel_update_equality_probability": float(equality),
        "ordered_sweep_stationary_distributions": {
            "1_then_2": [float(stationary_12[x]) for x in BITS],
            "2_then_1": [float(stationary_21[x]) for x in BITS]},
        "sequences": ["".join(x) for x in seqs],
        "joint": [float(joint[x]) for x in seqs],
        "local_temperature_half": [float(local_tempered[x]) for x in seqs],
        "global_temperature_half": [float(global_tempered[x]) for x in seqs],
        "first_token_a_at_temperature_half": {
            "local": float(local_q[()]["a"]), "global": float(global_first_a)},
        "variational_comparison": {
            "F_local": criterion(local_tempered),
            "F_global": criterion(global_tempered),
            "gap": variational_gap,
            "KL_local_global": kl(local_tempered, global_tempered),
            "printed_objective_directional_derivative": incorrect_directional_derivative},
        "modified_model_temperature_distributions": small_temperature,
        "modified_model_zero_temperature_limits": {
            "local": list(map(float, local_limit.values())),
            "global": list(map(float, global_limit.values()))},
        "independent_first_token_errors": {
            name: float(error) for name, error in first_token_errors.items()},
        "independent_sampling_match_probability": float(sampling_match),
        "tokenwise_risks": {"".join(x): float(v) for x, v in hamming.items()},
        "context_truncation_optima": truncation}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
