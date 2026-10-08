#!/usr/bin/env python3
"""Reproduce Chapter 3's calculations with Python's standard library.

The forward implementation is a finite, zero-bias instance of the chapter's
model: all LayerNorm gains are one and shifts are zero. It retains multi-head
attention, output projections, both residuals, row-wise MLPs, and final norm.
The deliberately leaking mode is a negative control, not a model to train.
Run from any directory; no files, data downloads, or training are required.
"""
import itertools
import json
import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[sum(x * b[k][j] for k, x in enumerate(row))
             for j in range(len(b[0]))] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def softmax(scores):
    maximum = max(scores)
    weights = [math.exp(s - maximum) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def layer_norm(rows, epsilon=0.2):
    result = []
    for row in rows:
        mean = sum(row) / len(row)
        variance = sum((x - mean) ** 2 for x in row) / len(row)
        result.append([(x - mean) / math.sqrt(variance + epsilon) for x in row])
    return result


def across_token_norm(rows):
    """Intentionally noncausal normalization used for the negative control."""
    n, d = len(rows), len(rows[0])
    means = [sum(row[j] for row in rows) / n for j in range(d)]
    variances = [sum((row[j] - means[j]) ** 2 for row in rows) / n
                 for j in range(d)]
    return [[(x - means[j]) / math.sqrt(variances[j] + 1)
             for j, x in enumerate(row)] for row in rows]


def attention(rows, params, causal=True, temperature=1.0):
    if temperature <= 0:
        raise ValueError('attention temperature must be positive')
    queries = matmul(rows, params['q'])
    keys = matmul(rows, params['k'])
    values = matmul(rows, params['v'])
    key_width = len(queries[0])
    weights = []
    for i, query in enumerate(queries):
        available = keys[:i + 1] if causal else keys
        scores = [dot(query, key) / (math.sqrt(key_width) * temperature)
                  for key in available]
        weight = softmax(scores)
        weights.append(weight + [0.0] * (len(rows) - len(weight)))
    return {'weights': weights, 'output': matmul(weights, values)}


def block(rows, params, causal=True, leaking_norm=False, epsilon=0.2):
    normalized = across_token_norm(rows) if leaking_norm else layer_norm(rows, epsilon)
    heads = [attention(normalized, head, causal) for head in params['heads']]
    joined = [[x for head in heads for x in head['output'][i]]
              for i in range(len(rows))]
    residual = add(rows, matmul(joined, params['o']))
    mlp_input = layer_norm(residual, epsilon)
    middle = [[max(x, 0.0) for x in row] for row in matmul(mlp_input, params['w1'])]
    output = add(residual, matmul(middle, params['w2']))
    return {'hidden': output, 'normalized': normalized, 'residual': residual,
            'mlp_input': mlp_input, 'heads': heads}


def forward(rows, layers, vocabulary_projection, causal=True,
            leaking_norm=False, epsilon=0.2):
    traces = []
    for index, params in enumerate(layers):
        trace = block(rows, params, causal, leaking_norm and index == 0, epsilon)
        traces.append(trace)
        rows = trace['hidden']
    normalized = layer_norm(rows, epsilon)
    logits = matmul(normalized, vocabulary_projection)
    return {'hidden': rows, 'normalized': normalized, 'logits': logits,
            'probabilities': [softmax(row) for row in logits], 'traces': traces}


def close(a, b, tolerance=1e-12):
    if abs(a - b) > tolerance:
        raise AssertionError(f'{a} differs from {b}')


def max_difference(a, b):
    return max(abs(x - y) for ar, br in zip(a, b) for x, y in zip(ar, br))


def entropy(a):
    return -sum(p * math.log(p) for p in a if p > 0)


def kl(a, b):
    return sum(p * math.log(p / q) for p, q in zip(a, b) if p > 0)


def variance(values):
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def run_checks():
    scores = [0, math.log(3)]
    weights = softmax(scores)
    close(weights[0], 1 / 4)
    close(weights[1], 3 / 4)
    half_temp = softmax([2 * s for s in scores])
    two_temp = softmax([s / 2 for s in scores])
    close(half_temp[0], 1 / 10)
    close(two_temp[0], 1 / (1 + math.sqrt(3)))

    objective = lambda a: dot(a, scores) + entropy(a)
    gaps = []
    for candidate in ([0.4, 0.6], [1, 0], [0, 1]):
        gap = objective(weights) - objective(candidate)
        close(gap, kl(candidate, weights))
        gaps.append(gap)
    reference, temperature = [0.7, 0.3], 0.5
    tilted = softmax([math.log(r) + s / temperature
                      for r, s in zip(reference, scores)])
    weighted_objective = lambda a: dot(a, scores) - temperature * kl(a, reference)
    candidate = [0.2, 0.8]
    close(weighted_objective(tilted) - weighted_objective(candidate),
          temperature * kl(candidate, tilted))

    signs = list(itertools.product((-1, 1), repeat=4))
    independent = [dot(q, k) for q in signs for k in signs]
    close(variance(independent), 4)
    close(variance([x / 2 for x in independent]), 1)
    correlated = [4 * x * y for x in (-1, 1) for y in (-1, 1)]
    close(variance(correlated), 16)
    close(variance([x / 2 for x in correlated]), 4)
    close(variance([dot(q, q) for q in signs]), 0)

    identity = [[1, 0], [0, 1]]
    zeros = [[0], [0]]
    toy_layer = {'heads': [{'q': zeros, 'k': zeros, 'v': identity}],
                 'o': identity, 'w1': identity, 'w2': identity}
    hand = forward([[1, -1], [0, 0]], [toy_layer], [[1, -1], [0, 0]], epsilon=1)
    a0 = 1 / (2 * math.sqrt(2))
    b0 = a0 + 1 / 6
    c0 = b0 / math.sqrt(1 + b0 * b0)
    close(hand['traces'][0]['mlp_input'][1][0], 1 / 3)
    close(hand['traces'][0]['mlp_input'][1][1], -1 / 3)
    close(hand['hidden'][1][0], a0 + 1 / 3)
    close(hand['hidden'][1][1], -a0)
    close(hand['probabilities'][1][0], 1 / (1 + math.exp(-2 * c0)))

    # One frozen model across all comparisons. Parameter construction does not
    # depend on the input length, its contents, or the chosen intervention.
    counter = itertools.count(1)
    def matrix(rows, columns):
        return [[((next(counter) * 37) % 101 - 50) / 73
                 for _ in range(columns)] for _ in range(rows)]
    layers = [{'heads': [{'q': matrix(4, 2), 'k': matrix(4, 2), 'v': matrix(4, 2)}
                         for _ in range(2)],
               'o': matrix(4, 4), 'w1': matrix(4, 5), 'w2': matrix(5, 4)}
              for _ in range(2)]
    vocabulary = matrix(4, 3)
    inputs = [[1, -1, 0.5, 2], [0, 2, -0.5, 1],
              [-1, 0.2, 1.1, -0.7], [0.3, 0.9, -2, 1]]
    altered = inputs[:2] + [[2, 1, -3, 0.2], [-2, 4, 0.2, -1]]
    good = forward(inputs, layers, vocabulary)
    good_changed = forward(altered, layers, vocabulary)
    suffix_difference = max_difference(good['logits'][:2], good_changed['logits'][:2])
    close(suffix_difference, 0)
    prefix_difference = max_difference(good['logits'][:2],
        forward(inputs[:2], layers, vocabulary)['logits'])
    close(prefix_difference, 0)
    unmasked = forward(inputs, layers, vocabulary, causal=False)
    unmasked_changed = forward(altered, layers, vocabulary, causal=False)
    unmasked_difference = max_difference(unmasked['logits'][:2], unmasked_changed['logits'][:2])
    assert unmasked_difference > 1e-6
    leaking = forward(inputs, layers, vocabulary, leaking_norm=True)
    leaking_changed = forward(altered, layers, vocabulary, leaking_norm=True)
    leaking_difference = max_difference(leaking['logits'][:2], leaking_changed['logits'][:2])
    assert leaking_difference > 1e-6
    order = [2, 0, 3, 1]
    permuted_inputs = [inputs[i] for i in order]
    equivariance_difference = max_difference([unmasked['logits'][i] for i in order],
        forward(permuted_inputs, layers, vocabulary, causal=False)['logits'])
    close(equivariance_difference, 0)
    causal_permutation_difference = max_difference([good['logits'][i] for i in order],
        forward(permuted_inputs, layers, vocabulary)['logits'])
    assert causal_permutation_difference > 1e-6

    leak_layer = dict(toy_layer, w2=[[0, 0], [0, 0]])
    leak_a = forward([[0, 0], [0, 0]], [leak_layer], identity, leaking_norm=True, epsilon=1)
    leak_b = forward([[0, 0], [2, -2]], [leak_layer], identity, leaking_norm=True, epsilon=1)
    close(leak_a['probabilities'][0][0], 0.5)
    close(leak_b['probabilities'][0][0], 1 / (1 + math.exp(2 / math.sqrt(3))))

    duplicated = softmax([0, math.log(3), math.log(3)])
    close(sum(duplicated[1:]), 6 / 7)
    preserved = softmax([math.log(0.5), math.log(0.25) + math.log(3),
                         math.log(0.25) + math.log(3)])
    close(sum(preserved[1:]), 3 / 4)
    prior, gap, temp = [0.2, 0.3, 0.5], 0.7, 0.4
    mass = softmax([math.log(r) + s / temp
                    for r, s in zip(prior, [2, 2 - gap, 2 - gap])])[0]
    close(mass, prior[0] / (prior[0] + (1 - prior[0]) * math.exp(-gap / temp)))
    ambiguous_a, ambiguous_b, values = [0.25, 0.5, 0.25], [0.4, 0.2, 0.4], [0, 1, 2]
    close(dot(ambiguous_a, values), 1)
    close(dot(ambiguous_b, values), 1)
    for candidate in (ambiguous_a, ambiguous_b):
        for actual, expected in zip(softmax([math.log(x) for x in candidate]), candidate):
            close(actual, expected)
    exposed_target_loss = math.log1p(math.exp(-math.sqrt(2) * 5))
    assert exposed_target_loss < 0.001

    return {
        'status': 'all targeted checks passed',
        'simple_attention': {'weights': weights, 'output': [weights[0], 2 * weights[1]],
                             'half_temperature': half_temp, 'two_temperature': two_temp},
        'entropy_objective_gaps': gaps, 'tilted_weights': tilted,
        'covariance': {'independent_coordinate_variance': 4, 'scaled': 1,
                       'correlated_coordinate_variance': 16, 'correlated_scaled': 4,
                       'dependent_vectors_variance': 0},
        'full_block_row2': {'a0': a0, 'b0': b0, 'c0': c0,
                            'hidden': hand['hidden'][1], 'probabilities': hand['probabilities'][1]},
        'two_layer_two_head_controls': {
            'causal_suffix_difference': suffix_difference,
            'prefix_vs_full_difference': prefix_difference,
            'unmasked_suffix_difference': unmasked_difference,
            'global_normalization_suffix_difference': leaking_difference,
            'unmasked_permutation_difference': equivariance_difference,
            'causal_permutation_difference': causal_permutation_difference},
        'explicit_normalization_leak_probabilities': {
            'first': leak_a['probabilities'][0], 'changed_future': leak_b['probabilities'][0]},
        'duplicate_aggregate_mass': sum(duplicated[1:]),
        'preserved_aggregate_mass': sum(preserved[1:]),
        'sharp_gap_mass': mass,
        'ambiguous_weights_outputs': [dot(ambiguous_a, values), dot(ambiguous_b, values)],
        'exposed_target_log_loss_at_lambda5': exposed_target_loss,
        'past_only_optimum': math.log(2),
    }


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2))
