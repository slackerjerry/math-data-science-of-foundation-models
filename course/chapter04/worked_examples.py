#!/usr/bin/env python3
"""Finite calculations for Chapter 4. No model training or external data."""
import argparse
from fractions import Fraction as F
from itertools import product
import json
import math

CHECKS = 0

def close(actual, expected, tol=1e-10):
    global CHECKS
    CHECKS += 1
    assert abs(float(actual) - float(expected)) <= tol, (actual, expected)

def kl(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q) if x)

def hinge(score, label):
    return max(0, 1 - score * label)

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def run():
    report = {}
    signs = list(product((-1, 1), repeat=2))
    for b, w1, w2 in [(0, 0, 0), (0.3, 0.7, -1.2)]:
        risk = sum((x*y-b-w1*x-w2*y)**2 for x, y in signs)/4
        close(risk, 1+b*b+w1*w1+w2*w2)

    p, q, a = (0.2, 0.8), (0.35, 0.65), (-2, 2)
    excess = kl(p, q)
    raw_loss = -sum(x*math.log(y) for x,y in zip(p,q))
    entropy = -sum(x*math.log(x) for x in p)
    close(raw_loss-entropy, excess)
    close(hinge(dot(a,p),1), 0)
    close(hinge(dot(a,q),1), 0.4)
    bound = 2*math.sqrt(2*excess)
    assert 0.4 <= bound < 1
    report["probability_features"] = dict(excess_log_loss=excess,
        task_risk=0.4, bound=bound, constant_score_risk=1)
    restricted = kl((0.1,0.9), (0.4,0.6))
    report["restricted_family"] = dict(optimization_gap=0, bayes_excess=restricted)
    report["large_readouts"] = []
    for r in (0.1, 0.03, 0.01, 0.003):
        pp = (0.5-r,0.5+r)
        aa = (-1/(2*r), 1/(2*r))
        ee = kl(pp, (0.5,0.5))
        close(dot(aa,pp),1)
        close(dot(aa,(0.5,0.5)),0)
        report["large_readouts"].append(dict(r=r, excess=ee,
            bound=math.sqrt(2*ee)/(2*r), task_risk=1))
    for alpha in (0.5, 0.1, 0.001):
        pp, qq = (0.1,0.9), (0.9,0.1)
        ee = alpha*kl(pp,qq)
        close(hinge(dot(a,qq),1),2.6)
        close(ee/alpha,0.8*math.log(9))
        assert 2.6 < 2*math.sqrt(2*ee/alpha)
    report["rare_context"] = dict(task_risk=2.6,
        bound=2*math.sqrt(1.6*math.log(9)))
    for score in (-4, -1, 0, 0.8, 3):
        assert (hinge(score,1)+hinge(score,-1))/2 >= 1

    # Enumerate the joint distribution independently of its conditional-mean formula.
    joint = []
    for latent, x1, x2 in product((-1,1), repeat=3):
        prob = F(1,2)
        prob *= F(3,4) if x1 == latent else F(1,4)
        prob *= F(3,4) if x2 == latent else F(1,4)
        joint.append((prob, latent, x1, x2))
    close(sum(t[0] for t in joint),1)
    for x1, x2 in signs:
        mass = sum(p for p,s,u,v in joint if (u,v)==(x1,x2))
        mean = sum(p*s for p,s,u,v in joint if (u,v)==(x1,x2))/mass
        close(mean,F(2,5)*(x1+x2))
    beta = F(1,10)
    raw_by_noise = []
    for amplitude in (F(0), F(2)):
        task, pretext, optimum = F(0), F(0), F(0)
        for prob, s, x1, x2 in joint:
            r = F(2,5)*(x1+x2)
            prediction = r + beta*(x1-x2)
            task += prob*(s-prediction)**2
            for nsign in (-1,1):
                target = s + amplitude*nsign
                pretext += prob*F(1,2)*(target-prediction)**2
                optimum += prob*F(1,2)*(target-r)**2
        close(task,F(123,200))
        close(pretext-optimum,F(3,200))
        raw_by_noise.append(dict(noise_amplitude=float(amplitude),
            raw_view_loss=float(pretext), optimal_view_loss=float(optimum),
            view_excess=float(pretext-optimum), task_risk=float(task)))
    report["noisy_latent_view"] = raw_by_noise
    report["weak_view"] = []
    for gamma in (1, 0.1, 0.01, 0):
        view_excess = sum((gamma*s)**2 for s in (-1,1))/2
        risk = sum(s*s for s in (-1,1))/2
        close(risk,1)
        if gamma:
            close(view_excess/gamma**2,1)
        report["weak_view"].append(dict(gamma=gamma,
            view_excess=view_excess, best_constant_task_risk=risk))

    # P2: two different posterior vectors yield the same mean.
    g = (-1,0,1)
    eta0, eta1 = (0,1,0), (0.5,0,0.5)
    close(dot(g,eta0),dot(g,eta1))
    close((0.5-1)**2/2+(0.5-0)**2/2,0.25)
    xi, eps = 0.3, 0.04
    head_risk = sum((1-(1+(xi+math.sqrt(eps))*x))**2
                    for x in (-1,1))/2
    close(head_risk,(xi+math.sqrt(eps))**2)
    assert head_risk > xi*xi+eps
    report["approximate_view"] = dict(constructed_head_excess=head_risk,
                                    sum_without_cross_term=xi*xi+eps)

    # Check softmax covariance by differentiating expected embeddings numerically.
    phi = [(1.0,0.0),(-0.5,1.0),(0.2,-0.7)]
    biases, z = [0.1,-0.2,0.4], [0.3,-0.6]
    def expected_embedding(zz):
        logits = [dot(v,zz)+b for v,b in zip(phi,biases)]
        weights = [math.exp(t-max(logits)) for t in logits]
        probs = [t/sum(weights) for t in weights]
        return probs, [sum(p*v[j] for p,v in zip(probs,phi)) for j in (0,1)]
    probs, mean = expected_embedding(z)
    covariance = [[sum(p*(v[i]-mean[i])*(v[j]-mean[j])
        for p,v in zip(probs,phi)) for j in (0,1)] for i in (0,1)]
    step = 1e-5
    for j in (0,1):
        zp, zm = z[:],z[:]
        zp[j] += step
        zm[j] -= step
        mp, mm = expected_embedding(zp)[1], expected_embedding(zm)[1]
        for i in (0,1):
            close((mp[i]-mm[i])/(2*step),covariance[i][j],1e-9)
    report["softmax_covariance"] = covariance

    # Bayes calculation from the complete sampling experiment, including repeats.
    for positive in ((0.8,0.2),(1.0,0.0)):
        for negative in ((0.5,0.5),(0.95,0.05)):
            total_tuple_mass = 0
            for tup in product(range(2), repeat=3):
                masses = [positive[tup[j]] *
                    math.prod(negative[tup[i]] for i in range(3) if i != j)/3
                    for j in range(3)]
                mass = sum(masses)
                total_tuple_mass += mass
                if not mass:
                    continue
                ratio = [positive[v]/negative[v] for v in tup]
                for j in range(3):
                    close(masses[j]/mass,ratio[j]/sum(ratio))
            close(total_tuple_mass,1)
    rho2_weights = [0.8/0.95,0.2/0.05]
    posterior = [x/sum(rho2_weights) for x in rho2_weights]
    close(posterior[0],4/23)
    close(posterior[1],19/23)
    max_temp = 2/math.log(19/4)
    for temp in (max_temp/2,max_temp):
        gap = temp*math.log(19/4)
        va = (-gap/2,math.sqrt(max(0,1-gap*gap/4)))
        vb = (gap/2,math.sqrt(max(0,1-gap*gap/4)))
        close(dot(va,va),1)
        close(dot(vb,vb),1)
        recovered = 1/(1+math.exp((vb[0]-va[0])/temp))
        close(recovered,4/23)
    # K=2 and M=2 identifier experiment: duplicate with probability 1/2.
    report["matching"] = dict(rho2_posterior=posterior,
        maximum_unit_score_temperature=max_temp,
        identifier_matching_bayes_loss=0.5*math.log(2),
        identifier_task_error=0.5)

    report["parity_readouts"] = []
    for lam in (0,0.25,0.5,1,2):
        features = [(x,y,lam*x*y) for x,y in signs]
        accuracy = 0
        for i,query in enumerate(features):
            candidates = [(sum((u-v)**2 for u,v in zip(query,features[j])),j)
                          for j in range(4) if i != j]
            distance = min(d for d,j in candidates)
            nearest = [j for d,j in candidates if abs(d-distance)<1e-12]
            correct = sum(signs[j][0]*signs[j][1] == signs[i][0]*signs[i][1]
                          for j in nearest)
            accuracy += correct/len(nearest)/4
        expected = 0 if lam < 1 else (1/3 if lam == 1 else 1)
        close(accuracy,expected)
        risks = {}
        for radius in (0,0.5,1,3):
            coeff = min(radius,1/lam) if lam else 0
            mse = sum((x*y-coeff*feat[2])**2
                      for (x,y),feat in zip(signs,features))/4
            close(mse,max(0,1-lam*radius)**2)
            risks[str(radius)] = mse
        report["parity_readouts"].append(dict(scale=lam,
            retrieval_accuracy=accuracy, norm_constrained_risks=risks))

    seen, unseen = 0,0
    assignments = list(product((-1,1), repeat=4))
    for labels in assignments:
        lookup = dict(enumerate(labels))
        seen += sum(lookup[i]==labels[i] for i in range(4))/4/len(assignments)
        # Fit identifiers 0,1. Predict the first fitting label on unseen 2,3.
        unseen += sum(labels[0]==labels[i] for i in (2,3))/2/len(assignments)
    close(seen,1)
    close(unseen,0.5)
    report["group_label_control"] = dict(seen_identifier_accuracy=seen,
                                        unseen_identifier_expected_accuracy=unseen)
    # Positive finite binary distributions and arbitrary bounded action costs.
    for pt, qt in product((0.1,0.3,0.7,0.9),repeat=2):
        pp,qq = (1-pt,pt),(1-qt,qt)
        tv = abs(pt-qt)
        assert kl(pp,qq)+1e-12 >= 2*tv*tv
        costs = [(0,1),(1,0),(0.2,0.4)]
        true = [dot(pp,c) for c in costs]
        model = [dot(qq,c) for c in costs]
        chosen = min(range(3),key=model.__getitem__)
        regret = true[chosen]-min(true)
        assert -1e-12 <= regret <= min(1,2*tv)+1e-12
    report["numerical_equality_checks"] = CHECKS
    report["scope"] = "Finite enumerations and numerical checks for the probability-feature and noisy-view models; analytic proofs are in the notes and solutions."
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.json:
        print(json.dumps(result,indent=2))
    else:
        print("Chapter 4 finite calculations passed.")
        print("Probability-feature excess:", result["probability_features"]["excess_log_loss"])
        print("Probability-feature bound:", result["probability_features"]["bound"])
        print("Noisy-view task risk:", result["noisy_latent_view"][0]["task_risk"])
        print("Numerical equality checks:", CHECKS)
