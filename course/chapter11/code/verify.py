"""Exact finite Bayesian stopping and seeded policy evaluation. Standard library only."""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import json, math, random

ROOT = Path(__file__).resolve().parent
D, C = F(1, 5), F(3, 100)

def branches(p):
    m = p * (1 - D) + (1 - p) * D
    return [(m, p * (1 - D) / m), (1 - m, p * D / (1 - m))]

@lru_cache(None)
def solve(p, h):
    risk = min(p, 1 - p)
    stop = (risk, F(0), 'stop')
    if h == 0:
        return stop
    sub = [(prob, solve(post, h - 1)) for prob, post in branches(p)]
    new_risk = sum(prob * out[0] for prob, out in sub)
    calls = 1 + sum(prob * out[1] for prob, out in sub)
    return (new_risk, calls, 'call') if new_risk + C * calls < risk else stop

def simulate(p, h, seed, n=100000, cached=False):
    rng = random.Random(seed)
    errors = calls_total = 0
    costs = []
    for _ in range(n):
        z = int(rng.random() < float(p))
        belief, remain, calls, first = p, h, 0, None
        while solve(belief, remain)[2] == 'call':
            if first is None or not cached:
                o = z if rng.random() >= float(D) else 1-z
                if first is None:
                    first = o
            else:
                o = first
            belief = branches(belief)[0 if o else 1][1]
            remain -= 1
            calls += 1
        err = int(int(belief >= F(1,2)) != z)
        errors += err
        calls_total += calls
        costs.append(err + float(C) * calls)
    mean = sum(costs) / n
    se = math.sqrt(sum((v-mean)**2 for v in costs)/(n-1)/n)
    return dict(n=n, seed=seed, cached=cached, error=errors/n,
                calls=calls_total/n, total=mean, total_se=se)

rows = []
for p in [F(1,2), F(4,5), F(19,20)]:
    for h in [0, 1, 2, 3, 5]:
        error, calls, action = solve(p,h)
        rows.append(dict(prior=str(p), horizon=h, error=float(error),
                         calls=float(calls), total=float(error+C*calls),
                         exact=[str(error),str(calls),str(error+C*calls)], action=action))
simulations = [simulate(F(1,2),5,11300,cached=x) for x in [False,True]]
verifier=[]
p,a,b=F(1,5),F(9,10),F(1,10)
s=p*a+(1-p)*b
for k in [0,1,4,20]:
    accepted=1-(1-s)**k
    cr,wr,ab,attempts=p*a/s*accepted,(1-p)*b/s*accepted,(1-s)**k,accepted/s
    verifier.append(dict(k=k, correct=float(cr),wrong=float(wr),abstain=float(ab),
      attempts=float(attempts),cost=float(wr+F(2,5)*ab+C*attempts),
      max_score_correct=float(F(9,10)**k-F(2,5)**k)))
result=dict(stopping=rows,simulations=simulations,verifier=verifier,
  coverage_heterogeneous=float(1-((1-F(1,10))**2+(1-F(9,10))**2)/2),
  pass_estimator=float(1-F(math.comb(3,2),math.comb(5,2))))
(ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
tex=[r'\begin{center}',r'\begin{tabular}{rrrrl}',r'\toprule',
 r'Prior $p$ & Horizon & Final error & Expected calls & Total cost\\',r'\midrule']
for row in rows:
    if row['prior']=='1/2' or row['prior']=='4/5' and row['horizon'] in [0,1,2]:
        tex.append(f"{float(F(row['prior'])):.2f} & {row['horizon']} & {row['error']:.5f} & {row['calls']:.4f} & {row['total']:.5f}\\\\")
tex += [r'\bottomrule',r'\end{tabular}',r'\end{center}',
 'The channel error is $0.2$ and each call costs $0.03$. The horizon is a maximum, not a requirement to use every call. The prior-$0.8$, horizon-two row is the policy calculated above. Longer horizons use the same posterior-dependent choice at each branch.',
 'A seeded simulation of $100{,}000$ independent-channel episodes at prior $0.5$ and horizon five gives total cost '
 +f"${simulations[0]['total']:.5f}$ (Monte Carlo standard error ${simulations[0]['total_se']:.5f}$). "
 +'This checks the enumerated model and policy. It is not a measured language-model tool result.']
(ROOT.parent/'experiment_results.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps(result,indent=2))
