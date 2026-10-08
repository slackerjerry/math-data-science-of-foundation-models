"""Finite, independently enumerated checks for Chapter 1. From the source root: python3 verify_examples.py.
Standard library only. Exact rational arithmetic; no research experiment replication.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
report = {}

def require(name, condition, values):
    if not condition:
        raise AssertionError((name, values))
    report[name] = {k: str(v) for k, v in values.items()}

def joint(source_a_weight, pa=F(1, 2), pb=F(1, 2),
          ma=(F(3, 4), F(1, 4)), mb=(F(1, 4), F(3, 4))):
    """Generate raw (source, context, outcome) probabilities."""
    out = {}
    for source, sw, px, means in [(0, source_a_weight, pa, ma),
                                  (1, 1-source_a_weight, pb, mb)]:
        for x in (0, 1):
            mass = sw * (px if x == 0 else 1-px)
            for y in (0, 1):
                out[source, x, y] = mass * (means[x] if y else 1-means[x])
    assert sum(out.values()) == 1
    return out

def conditional_means(data, marked=False):
    keys = {(s,x) if marked else x for s,x,y in data}
    means = {}
    for key in keys:
        rows = [(y,p) for (s,x,y),p in data.items()
                if ((s,x) if marked else x) == key]
        mass = sum(p for y,p in rows)
        if mass:
            means[key] = sum(y*p for y,p in rows) / mass
    return means

def risk(data, means, classification=False, marked=False):
    value = F(0)
    for (s,x,y), p in data.items():
        q = means[(s,x) if marked else x]
        if classification:
            value += p * (int(q > F(1,2)) != y)
        else:
            value += p * (q-y)**2
    return value

target = joint(F(7,10))
for lam, expected_train, expected_target in [
    (F(7,10), F(6,25), F(6,25)),
    (F(1), F(3,16), F(21,80)),
    (F(1,4), F(15,64), F(93,320)),
]:
    training = joint(lam)
    means = conditional_means(training)
    rt = risk(training, means)
    rd = risk(target, means)
    require("mixture_"+str(lam), (rt,rd)==(expected_train,expected_target),
            {"q_a":means[0], "q_b":means[1], "train":rt, "target":rd})

marked = conditional_means(joint(F(1,2)), marked=True)
require("marked_target", risk(target, marked, marked=True)==F(3,16)
        and risk(target, marked, True, True)==F(1,4),
        {"squared":risk(target,marked,marked=True),
         "classification":risk(target,marked,True,True)})

means_bad = conditional_means(joint(F(1,4)))
means_good = conditional_means(joint(F(7,10)))
require("classification_target",
        risk(target,means_bad,True)==F(3,5)
        and risk(target,means_good,True)==F(2,5),
        {"original":risk(target,means_bad,True),
         "corrected":risk(target,means_good,True)})

texts = ["aaaa", "aaab"]
counts = Counter(pair for word in texts for pair in zip(word,word[1:]))
def encode(word):
    out, i = [], 0
    while i < len(word):
        if word[i:i+2] == "aa":
            out.append("AA"); i += 2
        else:
            out.append(word[i]); i += 1
    return out
tokens = [encode(word) for word in texts]
require("token_merge", counts["a","a"]==5 and list(map(len,tokens))==[2,3]
        and ["".join(t.replace("AA","aa") for t in ts) for ts in tokens]==texts,
        {"pair_counts":dict(counts),"tokens":tokens})
require("token_ranking",
        (2*F(3,10)+3*F(9,10))/5 > F(65,100)
        and (F(3,10)+F(9,10))/2 < F(65,100),
        {"token":F(66,100),"document":F(6,10)})

source_a = {key:p for key,p in joint(F(1)).items() if key[0]==0}
retained = {key:p*(1 if key[2] else F(1,2)) for key,p in source_a.items()}
retention = sum(retained.values())
retained = {key:p/retention for key,p in retained.items()}
filtered_means = conditional_means(retained)
pxa = sum(p for (s,x,y),p in retained.items() if x==0)
require("filter", (retention,pxa,filtered_means[0],filtered_means[1])
        ==(F(3,4),F(7,12),F(6,7),F(2,5)),
        {"retention":retention,"p_a":pxa,"m_a":filtered_means[0],
         "m_b":filtered_means[1]})

shift_risks = []
for lam in [F(0),F(1,2),F(1)]:
    data = joint(lam, F(9,10),F(1,10),(F(1),F(1)),(F(0),F(0)))
    means = conditional_means(data)
    shift_risks.append(means[0]**2)
require("nonconvex_shift", shift_risks==[0,F(81,100),1]
        and shift_risks[1]>(shift_risks[0]+shift_risks[2])/2,
        {"risks":shift_risks})

# Exhaust all possible independent binary outcomes rather than using a variance formula.
p=F(3,10)
copies=(1,1,1,9)
raw_moments=[F(0),F(0)]
group_moments=[F(0),F(0)]
for ys in product([0,1],repeat=4):
    probability=F(1)
    for y in ys:
        probability *= p if y else 1-p
    raw=F(sum(r*y for r,y in zip(copies,ys)),sum(copies))
    group=F(sum(ys),4)
    for moments, value in [(raw_moments,raw),(group_moments,group)]:
        moments[0] += probability*value
        moments[1] += probability*value*value
rv=raw_moments[1]-raw_moments[0]**2
gv=group_moments[1]-group_moments[0]**2
require("unequal_copies", rv==F(7,12)*p*(1-p) and gv==p*(1-p)/4,
        {"row_variance":rv,"group_variance":gv,"effective_size":F(12,7)})

# Two independent groups; W_ij = U_i + E_ij, all U and E independent signs.
# Every W has variance 2 and within-group correlation 1/2.
means=[]
for signs in product([-1,1],repeat=8):
    us=signs[:2]; es=signs[2:]
    measurements=[us[i]+es[3*i+j] for i in range(2) for j in range(3)]
    means.append(F(sum(measurements),6))
average=sum(means)/len(means)
variance=sum((m-average)**2 for m in means)/len(means)
require("correlated_groups", variance==F(2,3),
        {"variance":variance,"n":2,"r":3,"sigma_squared":2,"rho":F(1,2)})

ra=F(7,8)
require("collection_to_pairs", 2*ra/(2*ra+6*(1-ra))==F(7,10),
        {"record_fraction":ra,"pair_fraction":F(7,10)})

# Check a finite family of approximate source functions against all 21 grid mixtures.
epsilon=F(1,10)
m1=(F(3,4),F(1,4)); m2=(F(1,4),F(3,4))
proxies=[(tuple(max(F(0),min(F(1),v+d1)) for v in m1),
          tuple(max(F(0),min(F(1),v+d2)) for v in m2))
         for d1,d2 in product([-epsilon,F(0),epsilon],repeat=2)]
largest=F(0)
for a,b in proxies:
    for lam in [F(i,20) for i in range(21)]:
        proxy={x:lam*a[x]+(1-lam)*b[x] for x in [0,1]}
        exact=conditional_means(joint(lam))
        difference=abs(risk(target,proxy)-risk(target,exact))
        largest=max(largest,difference)
        assert difference<=2*epsilon
require("proxy_bound_grid", largest<=2*epsilon,
        {"largest_objective_difference":largest,"bound":2*epsilon,
         "scope":"finite check, not proof of the uniform theorem"})


# Sequence calculations are enumerated from joint distributions, independently of the written derivation.
seq_a=dict(zip(['00','01','10','11'],[F(72,100),F(18,100),F(8,100),F(2,100)]))
seq_b=dict(zip(seq_a,[F(2,100),F(8,100),F(18,100),F(72,100)]))
assert sum(seq_a.values())==sum(seq_b.values())==1
sequence_risks=[]
for weight in [F(0),F(1,2),F(1)]:
    sequence_distribution={w:weight*seq_a[w]+(1-weight)*seq_b[w] for w in seq_a}
    prefix={h:sum(p for w,p in sequence_distribution.items() if w[0]==h) for h in ['0','1']}
    conditional={h:sequence_distribution[h+'1']/prefix[h] for h in prefix}
    sequence_risks.append((sequence_distribution['01'],conditional['0']))
    if weight==F(1,2):
        assert conditional=={'0':F(13,50),'1':F(37,50)}
        assert sequence_distribution['01']==prefix['0']*conditional['0']
        require('sequence_mixture',True,{'joint':sequence_distribution,'conditional_next_one':conditional})
assert sequence_risks==[(F(2,25),F(4,5)),(F(13,100),F(13,50)),(F(9,50),F(1,5))]
# -log is decreasing, so these exact likelihood comparisons also check the loss optima.
assert sequence_risks[2][0]>sequence_risks[1][0]>sequence_risks[0][0]
assert sequence_risks[0][1]>sequence_risks[1][1]>sequence_risks[2][1]
assert sequence_risks[1][1]**2 < sequence_risks[0][1]*sequence_risks[2][1]
require('joint_vs_conditional',True,{'likelihood_pairs':sequence_risks,'conditional_midpoint_nonconvex':True})
for weight in [F(1,2),F(7,10)]:
    pred=conditional_means(joint(weight))
    risks=[risk(joint(F(1)),pred),risk(joint(F(0)),pred)]
    expected=[F(3,16)+(1-weight)**2/4,F(3,16)+weight**2/4]
    assert risks==expected
    require('objective_choice_'+str(weight),True,{'source_risks':risks,'worst':max(risks)})
require('mismatch_excess',risk(target,means_bad)-risk(target,means_good)==F(81,1600),{'excess':F(81,1600)})
report['scope']='Exact finite enumeration and arithmetic for the source-mixture, weighting, and independent-observation examples. General proofs are in the solutions.'
(ROOT/'check_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
