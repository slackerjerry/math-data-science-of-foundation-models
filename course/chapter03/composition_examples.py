"""Finite-score causal pointer composition; Python standard library only."""
import itertools
import json
import math


def read(register, addresses, margin, score_error=0.0, return_error=0.0):
    n = len(register)
    result = []
    for i, address in enumerate(addresses):
        assert address <= i
        scores = [margin - score_error if j == address else score_error
                  for j in range(i + 1)]
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        den = sum(weights)
        output = [0.0] * n
        for j, w in enumerate(weights):
            output[register[j]] += w / den
        intended = register[address]
        output = [x + (-return_error if j == intended else return_error)
                  for j, x in enumerate(output)]
        clean = [2 * max(x - .25, 0) - 2 * max(x - .75, 0)
                 for x in output]
        assert max(abs(x - (j == intended)) for j, x in enumerate(clean)) < 2e-12
        result.append(max(range(n), key=clean.__getitem__))
    return result


def compose(f, k, score_error=.1, return_error=.08):
    n = len(f)
    margin = 2 * score_error + math.log((n - 1) * (.75 + return_error)
                                      / (.25 - return_error))
    p, r = list(f), list(range(n))
    trace = [(p[-1] + 1, r[-1] + 1)]
    for ell in range(k.bit_length()):
        next_p = read(p, p, margin, score_error, return_error)
        next_r = (read(p, r, margin, score_error, return_error)
                  if (k >> ell) & 1 else r)
        p, r = next_p, next_r
        trace.append((p[-1] + 1, r[-1] + 1))
    expected = list(range(n))
    for _ in range(k):
        expected = [f[j] for j in expected]
    assert r == expected
    return trace


def main():
    cases = 0
    for n in range(2, 7):
        for f in itertools.product(*(range(i + 1) for i in range(n))):
            for k in range(1, 10):
                compose(f, k)
                cases += 1
    trace = compose([max(0, i - 1) for i in range(8)], 6)
    assert trace == [(7, 8), (6, 8), (4, 6), (1, 2)]
    print(json.dumps({'exhaustive_map_hop_cases': cases,
                      'tested_lengths': [2, 3, 4, 5, 6],
                      'hops': [1, 9], 'score_error': .1,
                      'return_coordinate_error': .08,
                      'n8_k6_trace': trace, 'status': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
