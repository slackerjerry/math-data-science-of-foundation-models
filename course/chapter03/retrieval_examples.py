"""Chapter 3: sharp attention bounds and controlled retrieval constructions.

These finite calculations accompany, rather than replace, the proofs.
Only Python's standard library is required.
"""
import json
import math


def attention(scores, temperature=1.0, reference=None):
    assert temperature > 0 and len(scores) > 0
    reference = reference or [1/len(scores)]*len(scores)
    assert len(scores) == len(reference) and all(r > 0 for r in reference)
    logits = [s/temperature + math.log(r) for s, r in zip(scores, reference)]
    maximum = max(logits)
    weights = [math.exp(z-maximum) for z in logits]
    return [w/sum(weights) for w in weights]


def close(a, b):
    assert abs(a-b) < 1e-12, (a, b)


def run():
    mass_table = []
    for n in (2, 128, 4096):
        gap = math.log((n-1)*99)
        weights = attention([gap]+[0]*(n-1))
        close(weights[0], 0.99)
        # Values 0 at the desired key and 1 elsewhere attain the error bound.
        close(sum(weights[1:]), 0.01)
        mass_table.append({"positions": n, "margin_for_mass_0.99": gap})
    assert attention([7.0]) == [1.0]
    close(sum(attention([0, math.log(3), math.log(3)])[1:]), 6/7)
    close(sum(attention([0, math.log(3), math.log(3)],
                        reference=[1/2, 1/4, 1/4])[1:]), 3/4)

    n, k, eta, diameter, error = 256, 4, 0.02, 1.0, 0.1
    gap = math.log((n-k)/k*(diameter-error)/(error-eta))
    weights = attention([gap]*k+[0]*(n-k))
    output = eta*sum(weights[:k])+diameter*sum(weights[k:])
    close(output, error)
    # The simultaneous adverse perturbations consume twice their size.
    perturbation = 0.1
    intended_gap = gap+2*perturbation
    perturbed = attention([intended_gap-perturbation]*k+[perturbation]*(n-k))
    close(eta*sum(perturbed[:k])+sum(perturbed[k:]), error)
    failed = attention([gap-0.01]*k+[0]*(n-k))
    assert eta*sum(failed[:k])+sum(failed[k:]) > error

    # The single-head construction uses actual query/key dot products.
    width, target, scale = 5, 2, 7.0
    query = [scale*math.sqrt(width) if j == target else 0 for j in range(width)]
    keys = [[float(i == j) for j in range(width)] for i in range(width)]
    scores = [sum(q*k for q, k in zip(query, key))/math.sqrt(width) for key in keys]
    output_weights = attention(scores)
    close(output_weights[target], 1/(1+(width-1)*math.exp(-scale)))
    return {
        "margin_table": mass_table,
        "useful_set_example": {"implemented_margin": gap,
                              "intended_margin_with_error_0.1": intended_gap,
                              "attained_output_error": output},
        "lookup_target_mass": output_weights[target]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
