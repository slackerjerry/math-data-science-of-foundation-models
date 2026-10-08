"""Two-cell Gaussian inference. Run from the source root. NumPy and SciPy only."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr
r=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(13001);n=100000
z=rng.normal(size=(n,2))*np.array([3.,1.]);eps=rng.normal(size=(n,2));rows=[]
for j,name in enumerate(['Common','Contrast']):
    y=z[:,j]+eps[:,j];C=np.diag([9.,1.]);a=np.eye(2)[j];K=C@a/(1+a@C@a);Cp=C-np.outer(C@a,C@a)/(1+a@C@a)
    pred=y*K[1];err=z[:,1]-pred;v=Cp[1,1];loss=err**2
    rows.append(dict(sensor=name,analytic_mse=float(v),mse=float(loss.mean()),se=float(loss.std(ddof=1)/np.sqrt(n)),coverage=float((abs(err)<=1.959963984540054*np.sqrt(v)).mean())))
true=2+rng.normal(size=n);noise=.1*rng.normal(size=n);y=true+noise;mis=[]
for name,m,v in [('Wrong prior',0.,.01),('Matched prior',2.,1.)]:
    gain=v/(v+.01);pred=m+gain*(y-m);postv=v*.01/(v+.01);err=true-pred
    expected=(1-gain)**2*((2-m)**2+1)+gain**2*.01
    mis.append(dict(prior=name,mse=float(np.mean(err**2)),analytic_mse=float(expected),reported_var=postv,coverage=float(np.mean(abs(err)<=1.959963984540054*np.sqrt(postv)))))
# Exact Gaussian likelihood integration versus the denoised plug-in.
a=b=tau=h=1.;sig2=.1;k=a*tau*tau/(a*a*tau*tau+b*b);v=tau*tau*b*b/(a*a*tau*tau+b*b)
q=dict(kappa=k,conditional_variance=v,gradient_ratio=(sig2+h*h*v)/sig2)
# Pooled and joint posteriors for two alternative outcomes.
pooled=dict(mean=0.,variance=.5,joint_variance=1/3)
result=dict(protocol=dict(seed=13001,specimens=n,interval_z=1.959963984540054),target_design=rows,prior_shift=mis,likelihood=q,pooled=pooled)
(r/'measurement_results.json').write_text(json.dumps(result,indent=2))
lines=[r'\begin{center}',r'\begin{tabular}{lrrr}',r'\toprule',r'Sensor & Exact target MSE & Observed target MSE & 95\% coverage\\',r'\midrule']
for x in rows:lines.append(f"{x['sensor']} & {x['analytic_mse']:.3f} & {x['mse']:.5f} & {x['coverage']:.4f}\\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}',r'\begin{center}',r'\begin{tabular}{lrrr}',r'\toprule',r'Inference prior & Observed target MSE & Reported variance & 95\% coverage\\',r'\midrule']
for x in mis:lines.append(f"{x['prior']} & {x['mse']:.5f} & {x['reported_var']:.5f} & {x['coverage']:.4f}\\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}']
(r/'experiment_results.tex').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
