"""Synthetic evaluation checks. Run with Python 3, NumPy and SciPy installed."""
from pathlib import Path
import json
import numpy as np
r=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(14001)
R=50000;n=100;L=10
q=rng.choice([.1,.9],size=(R,n))
x=rng.binomial(L,q).mean(axis=1)/L
cluster={'replications':R,'n':n,'L':L,'observed_variance':float(x.var(ddof=1)),
         'exact_variance':(.16+.09/L)/n,'iid_answer_variance':.25/(n*L)}
v=rng.binomial(100,.5,size=(R,20))/100
winner=v.argmax(axis=1)
fresh=rng.binomial(100,.5,size=R)/100
selection={'menu':20,'validation_mean':float(v[np.arange(R),winner].mean()),'fresh_mean':float(fresh.mean()),'fresh_se':float(fresh.std(ddof=1)/np.sqrt(R))}
rows=[]
for rho in [0.,.1,.5]:
 f=rng.normal(size=(R,20));y=rho*f+np.sqrt(1-rho*rho)*rng.normal(size=(R,20))
 fb=f.mean(axis=1);yb=y.mean(axis=1)
 cov=((f-fb[:,None])*(y-yb[:,None])).sum(axis=1)/19
 t=yb-cov*fb
 ratio=t*t/.05
 rows.append({'rho':rho,'mse_ratio':float(ratio.mean()),'mc_se':float(ratio.std(ddof=1)/np.sqrt(R)),'exact_ratio':1+1/19-18*rho*rho/19})
result={'seed':14001,'cluster':cluster,'selection':selection,'gaussian_tuning':rows}
(r/'evaluation_results.json').write_text(json.dumps(result,indent=2)+'\n')
tex=['\\begin{center}\\begin{tabular}{lrr}','\\toprule','Quantity & Simulation & Analytic target\\\\\\midrule',f'Prompt mean variance & {cluster["observed_variance"]:.6f} & {cluster["exact_variance"]:.6f}\\\\',f'Selected validation mean & {selection["validation_mean"]:.4f} & ---\\\\',f'Fresh selected-system mean & {selection["fresh_mean"]:.4f} & 0.5000\\\\']
for a in rows:tex.append(f'Tuning MSE ratio, $\\rho={a["rho"]:.1f}$ & {a["mse_ratio"]:.4f} & {a["exact_ratio"]:.4f}\\\\')
tex+=['\\bottomrule\\end{tabular}\\end{center}',f'There are {R:,} independent replications with seed 14001. The Monte Carlo standard errors of the three MSE ratios are '+', '.join(f'{a["mc_se"]:.4f}' for a in rows)+'.']
(r/'experiment_results.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps(result,indent=2))
