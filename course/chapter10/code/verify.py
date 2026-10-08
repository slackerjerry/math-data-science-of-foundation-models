"""Exact finite-policy and group-estimator checks; no language-model training."""
from pathlib import Path
import json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
p0=np.array([.89,.10,.01]);u=np.array([1.,.7,0]);r=np.array([1.,.7,3.])
def softmax(z):p=np.exp(z-z.max());return p/p.sum()
rows=[]
for beta in [2.,1.,.5,.2]:
 opt=softmax(np.log(p0)+r/beta);z=np.log(p0).copy()
 for t in range(60000):
  p=softmax(z);v=r-beta*np.log(p/p0);g=p*(v-p@v);z+=.15*g
  if np.linalg.norm(g)<1e-10:break
 p=softmax(z);gap=beta*np.sum(p*np.log(p/opt))
 rows.append({'beta':beta,'opt':opt.tolist(),'proxy':float(opt@r),'utility':float(opt@u),'KL':float(np.sum(opt*np.log(opt/p0))),'steps':t+1,'gap':float(gap),'maxprob_error':float(abs(p-opt).max())})
groups=[]
for p in [.2,.5,.8]:
 for c in [1.,2.]:
  totals=np.zeros(4)
  for ys in itertools.product([0.,1.],repeat=2):
   y=np.array(ys);pr=float(np.prod(np.where(y==1,p,1-p)));rew=c*y;s=y-p
   adv=rew-rew.mean();sd=rew.std();norm=adv/sd if sd else np.zeros(2)
   totals+=pr*np.array([(rew*s).mean(),(adv*s).mean(),(2*adv*s).mean(),(norm*s).mean()])
  groups.append({'p':p,'c':c,'expected_raw_mean_loo_std':totals.tolist()})
report={'policy':rows,'group':groups,'gradient_exact':float(.16*np.log(16)),'integrand_gradient':.6}
(ROOT/'results.json').write_text(json.dumps(report,indent=2))
tex=[r'\begin{center}',r'\begin{tabular}{@{}rrrrr@{}}',r'\toprule',r'$\beta$ & Third-response mass & Proxy reward & True utility & Reference KL\\',r'\midrule']
for a in rows:tex.append(f"{a['beta']:.1f} & {a['opt'][2]:.4f} & {a['proxy']:.4f} & {a['utility']:.4f} & {a['KL']:.4f}"+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{center}',r'These are exact finite-model evaluations, not measured language-model performance. The reference has true utility $0.96$. The script also optimizes response logits by exact expected gradients and records the gap to the analytic KL-regularized optimum. It enumerates group estimators separately, so sampling bias is not confused with optimization error.']
(ROOT.parent/'experiment_results.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps(report,indent=2))
