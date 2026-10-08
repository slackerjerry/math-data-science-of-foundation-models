"""Population-stream flow matching for a one-dimensional two-mode distribution.
Run from source root: python course/chapter12/code/flow_experiment.py
Requires numpy, scipy, torch, matplotlib. No pretrained weights or network access.
"""
from pathlib import Path
import json, math, platform, argparse
import numpy as np
from scipy.special import ndtr
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
SIGMA=.35
STEPS=4000
BATCH=256
SEEDS=[12,23,34]
torch.set_num_threads(2)
class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x,t):
        if not torch.is_tensor(t): t=torch.full_like(x,t)
        return self.net(torch.cat((x,t),dim=1))
def exact_field(x,t):
    q=(1-t)**2+(t*SIGMA)**2
    mu=2*np.tanh(2*t*x/q)
    return mu+(t*SIGMA**2-(1-t))/q*(x-t*mu)
def cdf(x): return .5*ndtr((x+2)/SIGMA)+.5*ndtr((x-2)/SIGMA)
def transport(z):
    u=ndtr(z);lo=np.full_like(z,-12.);hi=np.full_like(z,12.)
    for _ in range(65):
        mid=(lo+hi)/2;left=cdf(mid)<u
        lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
    return (lo+hi)/2
@torch.no_grad()
def rollout(model,z,n):
    x=torch.tensor(z[:,None],dtype=torch.float32)
    for k in range(n):x=x+model(x,k/n)/n
    return x[:,0].numpy().astype(float)
def metrics(x,ref):
    return dict(paired_mse=float(np.mean((x-ref)**2)),mean=float(x.mean()),variance=float(x.var()),positive_fraction=float(np.mean(x>0)),valley_fraction=float(np.mean(abs(x)<1)),mean_abs_quantile_error=float(np.mean(abs(np.sort(x)-np.sort(ref)))))
def main(evaluate_only=False):
    rng=np.random.default_rng(12001);z=rng.normal(size=12000);ref=transport(z)
    out={'protocol':{'seeds':SEEDS,'steps':STEPS,'batch':BATCH,'architecture':'2-64-64-1 SiLU','optimizer':'Adam lr=0.001','training':'fresh independent endpoint pairs, uniform t at every update','evaluation_seed':12001,'evaluation_samples':len(z),'reference':'mixture quantile transport of the same Gaussian samples','torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()},'target':{'mean':0.,'variance':4+SIGMA**2,'positive_fraction':.5,'valley_fraction':float(cdf(1)-cdf(-1))},'finite_reference':metrics(ref,ref),'exact':{},'learned':{}}
    for n in [8,32,128]:
        x=z.copy()
        for k in range(n):x+=exact_field(x,k/n)/n
        out['exact'][str(n)]=metrics(x,ref)
    hold=np.random.default_rng(12002);tt=hold.random(10000);zz=hold.normal(size=10000);xx=2*hold.choice([-1,1],size=10000)+SIGMA*hold.normal(size=10000);xt=(1-tt)*zz+tt*xx;v=exact_field(xt,tt)
    plot_data={'reference':ref}
    for seed in SEEDS:
        torch.manual_seed(seed);m=Field();opt=torch.optim.Adam(m.parameters(),lr=.001)
        trace=[]
        if evaluate_only:
            checkpoint=torch.load(ROOT/f'field_seed{seed}.pt',map_location='cpu',weights_only=True)
            m.load_state_dict(checkpoint['state_dict'])
            saved=json.loads((ROOT/'results.json').read_text())
            trace=saved['learned'][str(seed)]['training_loss_snapshots']
        else:
            for k in range(STEPS):
                z0=torch.randn(BATCH,1);x1=2*(2*torch.randint(0,2,(BATCH,1))-1)+SIGMA*torch.randn(BATCH,1);t=torch.rand(BATCH,1)
                loss=((m((1-t)*z0+t*x1,t)-(x1-z0))**2).mean()
                opt.zero_grad();loss.backward();opt.step()
                if (k+1)%1000==0:trace.append([k+1,float(loss.detach())])
            torch.save({'state_dict':m.state_dict(),'seed':seed,'steps':STEPS},ROOT/f'field_seed{seed}.pt')
        m.eval()
        with torch.no_grad():vh=m(torch.tensor(xt[:,None],dtype=torch.float32),torch.tensor(tt[:,None],dtype=torch.float32)).numpy()[:,0]
        item={'training_loss_snapshots':trace,'heldout_marginal_field_mse':float(np.mean((vh-v)**2)),'rollouts':{}}
        for n in [8,32,128]:
            y=rollout(m,z,n);item['rollouts'][str(n)]=metrics(y,ref)
            if seed==SEEDS[0] and n==128:plot_data['learned']=y
        out['learned'][str(seed)]=item
        print(seed,item,flush=True)
    out['conditioning_control']={}
    for component in [-2.,2.]:
        reference=component+SIGMA*z
        x=z.copy();n=128
        for k in range(n):
            t=k/n;q=(1-t)**2+(t*SIGMA)**2
            x+=(component+(t*SIGMA**2-(1-t))/q*(x-t*component))/n
        out['conditioning_control'][str(component)]=metrics(x,reference)
    (ROOT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    table=[r'\begin{center}',r'\begin{tabular}{lrrr}',r'Field & Steps & Paired endpoint MSE & Valley fraction\\\hline']
    for n in [8,32,128]:
        a=out['exact'][str(n)];table.append(f'Exact & {n} & {a["paired_mse"]:.6f} & {a["valley_fraction"]:.4f}'+r'\\')
    for seed in SEEDS:
        a=out['learned'][str(seed)]['rollouts']['128'];table.append(f'Learned, seed {seed} & 128 & {a["paired_mse"]:.6f} & {a["valley_fraction"]:.4f}'+r'\\')
    table.extend([r'\end{tabular}',r'\end{center}'])
    (ROOT.parent/'experiment_results.tex').write_text('\n'.join(table)+'\n')
    fig,ax=plt.subplots(figsize=(6.2,2.5));grid=np.linspace(-4,4,600)
    density=(np.exp(-.5*((grid+2)/SIGMA)**2)+np.exp(-.5*((grid-2)/SIGMA)**2))/(2*SIGMA*np.sqrt(2*np.pi))
    ax.plot(grid,density,label='Target density',color='black',lw=1.8)
    ax.hist(plot_data['learned'],bins=80,range=(-4,4),density=True,alpha=.5,label='Learned field, seed 12, 128 steps',color='#437f9f')
    ax.set(xlabel='Generated value',ylabel='Density');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT.parent/'flow_distribution.pdf');plt.close(fig)
    print(json.dumps({'exact':out['exact'],'target':out['target']},indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate-only',action='store_true')
    main(parser.parse_args().evaluate_only)
