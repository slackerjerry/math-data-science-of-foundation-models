"""Finite-score token routing, checked against direct prefix searches.

Standard library only. No target pointer enters the matching attention scores.
From the source root: python3 chapter03/token_routing_examples.py
"""
import itertools
import json
import math


def onehot(i, size):
    return [float(j == i) for j in range(size)]


def psi(x):
    return 2 * max(x - 0.25, 0) - 2 * max(x - 0.75, 0)


def attend(query, keys, values):
    scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query))
              for key in keys]
    raw = [math.exp(s - max(scores)) for s in scores]
    weights = [w / sum(raw) for w in raw]
    out = [sum(w * v[d] for w, v in zip(weights, values))
           for d in range(len(values[0]))]
    return [psi(z) for z in out], weights, scores


def read_identifier(address, values, receiving, scale):
    width = len(values)
    query = [scale * math.sqrt(width) * z for z in onehot(address, width)]
    keys = [onehot(j, width) for j in range(receiving + 1)]
    out, _, _ = attend(query, keys, values[:receiving + 1])
    return out


def decode(row):
    assert sum(abs(v - round(v)) for v in row) < 1e-10, row
    assert abs(sum(row) - 1) < 1e-10, row
    return max(range(len(row)), key=row.__getitem__)


def direct_pointers(tokens):
    x = [None] + list(tokens)
    return [0] + [max([0] + [j for j in range(1, i + 1) if x[j - 1] == x[i]])
                  for i in range(1, len(x))]


def construct_pointers(tokens, alphabet):
    n = len(tokens)
    token_ids = [len(alphabet)] + [alphabet.index(x) for x in tokens]
    token_rows = [onehot(i, len(alphabet) + 1) for i in token_ids]
    # A little extra margin prevents boundary roundoff from obscuring exact cleanup.
    scale = math.log(3 * n) + 0.1
    previous = [read_identifier(max(0, i - 1), token_rows, i, scale)
                for i in range(n + 1)]
    assert [decode(p) for p in previous] == [token_ids[max(0, i - 1)] for i in range(n + 1)]
    key_width = len(alphabet) + 1
    keys = [[0.] * len(alphabet) + [float(n + 1)]]
    keys += [previous[j][:-1] + [float(j)] for j in range(1, n + 1)]
    values = [onehot(j, n + 1) for j in range(n + 1)]
    pointers = []
    for i in range(n + 1):
        qbase = [2 * (n + 1) * z for z in token_rows[i][:-1]] + [1.]
        query = [scale * math.sqrt(key_width) * z for z in qbase]
        out, weights, scores = attend(query, keys[:i + 1], values[:i + 1])
        pointers.append(decode(out))
        if i:
            ordered = sorted(scores, reverse=True)
            assert ordered[0] - ordered[1] >= scale - 1e-10
            assert max(weights) >= .75 - 1e-12
    return pointers, token_rows, scale


def composed_answer(pointers, token_rows, k, scale):
    n = len(pointers) - 1
    p, r = list(pointers), list(range(n + 1))
    for ell in range(k.bit_length()):
        old_values = [onehot(j, n + 1) for j in p]
        p_next = [decode(read_identifier(p[i], old_values, i, scale)) for i in range(n + 1)]
        r_next = ([decode(read_identifier(r[i], old_values, i, scale)) for i in range(n + 1)]
                  if (k >> ell) & 1 else r[:])
        p, r = p_next, r_next
    result = [decode(read_identifier(r[i], token_rows, i, scale)) for i in range(n + 1)]
    return r, result


def main():
    sequences = 0
    hop_checks = 0
    for n in range(1, 8):
        for tokens in itertools.product('ab', repeat=n):
            pointers, values, scale = construct_pointers(tokens, 'ab')
            expected = direct_pointers(tokens)
            assert pointers == expected
            sequences += 1
            for k in (1, 2, 3, 4, 6, 9):
                final, answer = composed_answer(pointers, values, k, scale)
                for i in range(n + 1):
                    target = i
                    for _ in range(k):
                        target = expected[target]
                    assert final[i] == target
                    assert answer[i] == decode(values[target])
                    hop_checks += 1
    example = 'acdbcaba'
    p, vals, scale = construct_pointers(example, 'abcd')
    assert p == [0, 0, 0, 0, 0, 3, 2, 5, 7]
    answers = [composed_answer(p, vals, k, scale)[1][-1] for k in range(1, 5)]
    assert answers == [1, 2, 3, 4]  # b, c, d, sentinel
    # Prefix recovery must survive arbitrary suffixes, using the same length parameters.
    pa, _, _ = construct_pointers('acdbcaba', 'abcd')
    pb, _, _ = construct_pointers('acdbdddd', 'abcd')
    assert pa[:5] == pb[:5]
    rotary = [math.cos(d * math.pi / 2) for d in (-1, -2, -4)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(rotary, (0, -1, 1)))
    return {'exhaustive_binary_sequences': sequences, 'row_hop_checks': hop_checks,
            'example_pointers': p, 'example_hop_tokens': ['b', 'c', 'd', 'sentinel'],
            'rotary_inner_products': rotary, 'status': 'passed'}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
