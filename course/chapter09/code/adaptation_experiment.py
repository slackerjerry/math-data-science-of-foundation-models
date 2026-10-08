"""Information-matched adaptation on the saved Chapter 8 seed-22 checkpoint.
No hyperparameter selection using test labels. All task/retention sets saved by seed.
"""
from pathlib import Path
import sys,copy,math,json,time,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'chapter08/code'))
from icl_experiment import Model,D,N,SIGMA
STEPS=120
TASKS=12

def hidden(m,x):
 t=torch.zeros(len(x),1,D+2);t[:,0,:D]=x
 z=m.embed(t)+m.pos(torch.zeros(1,dtype=torch.long))
 return m.net(z,mask=torch.zeros(1,1,dtype=torch.bool))[:,0]

def single(m,x): return m.out(hidden(m,x)).squeeze(-1)
class LowRank(nn.Module):
 def __init__(self,shape,r):
  super().__init__();o,i=shape;self.A=nn.Parameter(torch.randn(r,i)/math.sqrt(i));self.B=nn.Parameter(torch.zeros(o,r));self.scale=1/math.sqrt(r)
 def forward(self,W):return W+self.scale*self.B@self.A

def lora(m,r):
 for p in m.parameters():p.requires_grad_(False)
 for layer in m.net.layers:
  for mod,name in [(layer.self_attn,'in_proj_weight'),(layer.self_attn.out_proj,'weight'),(layer.linear1,'weight'),(layer.linear2,'weight')]:
   parametrize.register_parametrization(mod,name,LowRank(getattr(mod,name).shape,r))
 for p in m.out.parameters():p.requires_grad_(True)
 return m

@torch.no_grad()
def risk(m,x,t):return float(((single(m,x)-t)**2).mean())

@torch.no_grad()
def retention(m,x,y,t):return float(((m(x,y)[:,-1]-t)**2).mean())

def main():
 torch.set_num_threads(2)
 cp=ROOT.parents[1]/'chapter08/code/checkpoint_22.pt';saved=torch.load(cp,weights_only=True);base=Model();base.load_state_dict(saved['state_dict']);base.eval()
 torch.manual_seed(901)
 wr=torch.randn(256,D)/math.sqrt(D);xr=torch.randn(256,N+1,D);tr=(xr*wr[:,None]).sum(-1)
 yr=tr[:,:N]+SIGMA*torch.randn(256,N);oldrisk=retention(base,xr,yr,tr[:,-1]);records=[]
 for task in range(TASKS):
  torch.manual_seed(9100+task);w=torch.randn(D)/math.sqrt(D);xt=torch.randn(N,D);yt=xt@w+SIGMA*torch.randn(N);xq=torch.randn(512,D);tq=xq@w
  Xctx=torch.cat([xt[None].expand(512,-1,-1),xq[:,None]],1);Yctx=yt[None].expand(512,-1)
  with torch.no_grad():
   bayes=torch.linalg.solve(xt.T@xt+SIGMA**2*D*torch.eye(D),xt.T@yt)
   row={'task':task,'w':w.tolist(),'raw_bayes_risk':float(((xq@bayes-tq)**2).mean()),'base_zero_context':risk(base,xq,tq),'base_icl':float(((base(Xctx,Yctx)[:,-1]-tq)**2).mean()),'base_retention':oldrisk,'methods':{}}
  for name in ['head','lora1','lora4','full']:
   torch.manual_seed(10000+task);m=copy.deepcopy(base);start=time.time()
   if name=='head':
    with torch.no_grad():
     H=hidden(m,xt);Ha=torch.cat([H,torch.ones(N,1)],1);res=yt-m.out(H).squeeze(-1);delta=Ha.T@torch.linalg.solve(Ha@Ha.T+.1*N*torch.eye(N),res)
     m.out.weight.add_(delta[:-1][None]);m.out.bias.add_(delta[-1]);count=65
   else:
    if name.startswith('lora'):m=lora(m,int(name[-1]))
    count=sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=0.);m.train()
    for step in range(STEPS):
     pred=single(m,xt);loss=((pred-yt)**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
   m.eval()
   row['methods'][name]={'parameters':count,'steps':0 if name=='head' else STEPS,'seconds':time.time()-start,'train_mse':risk(m,xt,yt),'test_risk':risk(m,xq,tq),'retention_risk':retention(m,xr,yr,tr[:,-1])}
  records.append(row);(ROOT/'results.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'task_seed_base':9100,'retention_seed':901,'records':records},indent=2));print(task,{k:round(v['test_risk'],4) for k,v in row['methods'].items()},flush=True)
 print('complete',flush=True)
if __name__=='__main__':main()
