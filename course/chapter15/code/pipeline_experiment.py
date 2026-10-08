"""Exact finite-state course synthesis. Python standard library only."""
from pathlib import Path
from itertools import product
from fractions import Fraction as F
import json
r=Path(__file__).resolve().parents[1]
d=F(1,10);ep=F(1,20);cost=F(2,5)
task_error=F(0);lookup_error=F(0);adaptive=F(0);callprob=F(0)
for obs in product((-1,1),repeat=3):
 weights=[]
 for theta in (-1,1):
  p=F(1,2)
  for s in obs:p*=1-d if s==theta else d
  weights.append(p)
 mass=sum(weights);e=min(weights)/mass
 rr=ep+(1-2*ep)*e
 task_error+=mass*e;lookup_error+=mass*rr
 call=F(1,2)-rr>cost
 adaptive+=mass*(rr+cost if call else F(1,2))
 callprob+=mass*call
assert task_error==F(7,250)
assert lookup_error==F(47,625)
assert adaptive==F(1161,2500)
# Price each complete procedure from its enumerated prediction risk.
setup=F(3,10);context_per_demo=F(1,50);adapt_per_query=F(1,200)
lookup_price=F(3,100);n_demo=3;queries=20
context_per_query=n_demo*context_per_demo
context_objective=lookup_error+context_per_query+lookup_price
adaptation_objective=lookup_error+setup/queries+adapt_per_query+lookup_price
drift_slope=(1-2*ep)*(1-2*task_error)
drift_threshold=(context_per_query-adapt_per_query-setup/queries)/drift_slope
es=F(1,100)
p=1/(1+es-es*es);q=es/(1+es-es**3)
result={'enumeration':'all eight three-demonstration sequences','task_error':float(task_error),'lookup_error':float(lookup_error),'lookup_only_or_task_only_error':.5,'adaptive_loss_plus_cost':float(adaptive),'always_lookup_loss_plus_cost':float(lookup_error+cost),'adaptive_lookup_probability':float(callprob),'context_objective':float(context_objective),'adaptation_objective_Q20':float(adaptation_objective),'drift_break_even_Q20':float(drift_threshold),'rare_true_posterior':float(p),'rare_working_posterior':float(q),'rare_conditional_regret':float(2*p-1),'rare_observation_probability':float(es+(1-es)*es*es)}
(r/'pipeline_results.json').write_text(json.dumps(result,indent=2)+'\n')
tex=['\\begin{center}\\begin{tabular}{lr}','\\toprule','Exact quantity & Value\\\\\\midrule']
for title,key in [('Task error, three demonstrations','task_error'),('Task plus fresh lookup error','lookup_error'),('Evidence-dependent lookup, loss plus cost','adaptive_loss_plus_cost'),('Always lookup, loss plus cost','always_lookup_loss_plus_cost'),('Context objective at 20 queries','context_objective'),('Adaptation objective at 20 queries','adaptation_objective_Q20'),('Rare-observation conditional regret','rare_conditional_regret')]:tex.append(f'{title} & {result[key]:.6f}\\\\')
tex+=['\\bottomrule\\end{tabular}\\end{center}','These are exact finite-model calculations, rounded for display. They have no Monte Carlo sampling error.']
(r/'experiment_results.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps(result,indent=2))
