"""A declared small causal-Transformer ICL experiment, with no text pretraining.
Run from any directory. Requires PyTorch and NumPy. CPU, float32, 3 fixed seeds.
The script writes local checkpoints, raw metrics, and a LaTeX table beside itself.
"""
from pathlib import Path
import json,math,time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
D,N,WIDTH=4,8,64
SIGMA=.1
STEPS=1200
BATCH=128

torch.set_num_threads(2)
class Model(nn.Module):
 def __init__(self):
  super().__init__(); self.embed=nn.Linear(D+2,WIDTH);self.pos=nn.Embedding(2*N+1,WIDTH)
  layer=nn.TransformerEncoderLayer(WIDTH,4,128,dropout=0.,batch_first=True,norm_first=True)
  self.net=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.out=nn.Linear(WIDTH,1)
 def forward(self,x,y):
  b,k,d=x.shape;t=torch.zeros(b,2*k-1,D+2)
  t[:,::2,:D]=x;t[:,1::2,D]=y;t[:,1::2,D+1]=1
  z=self.embed(t)+self.pos(torch.arange(t.shape[1]))
  mask=torch.triu(torch.ones(t.shape[1],t.shape[1],dtype=torch.bool),1)
  return self.out(self.net(z,mask=mask)).squeeze(-1)[:,::2]

def draw(b,n):
 w=torch.randn(b,D)/math.sqrt(D);x=torch.randn(b,n+1,D);truth=(x*w[:,None]).sum(-1)
 return x,truth[:,:n]+SIGMA*torch.randn(b,n),truth,w

def stats(a):
 a=np.asarray(a,dtype=float);return {'mean':float(a.mean()),'se':float(a.std(ddof=1)/np.sqrt(len(a))),'n':len(a)}

@torch.no_grad()
def evaluate(model):
 torch.manual_seed(8001);out={};raw={}
 for n in (0,2,4,8):
  names=['base','order','sign','no_signal','distractor'] if n else ['base']
  vals={k:[] for k in names};methods={'bayes':[],'one_step':[],'nearest':[],'zero':[]};sens={k:[] for k in names if k!='base'}
  for batch in range(16):
   x,y,truth,w=draw(128,n);q=x[:,-1];target=truth[:,-1];base=model(x,y)[:,-1]
   vals['base'].extend(((base-target)**2).tolist());methods['zero'].extend((target**2).tolist())
   if n:
    xd=x[:,:n];eye=torch.eye(D).expand(128,D,D)
    hat=torch.linalg.solve(xd.transpose(1,2)@xd+SIGMA**2*D*eye,xd.transpose(1,2)@y[...,None]).squeeze(-1)
    bayes=(q*hat).sum(-1);eta=1/(1+(D+1)/n+D*SIGMA**2/n)
    gd=eta*(q*(xd*y[...,None]).mean(1)).sum(-1)
    ii=((xd-q[:,None])**2).sum(-1).argmin(-1);nnv=y[torch.arange(128),ii]
    methods['bayes'].extend(((bayes-target)**2).tolist());methods['one_step'].extend(((gd-target)**2).tolist());methods['nearest'].extend(((nnv-target)**2).tolist())
    perm=torch.randperm(n);xp=torch.cat([xd[:,perm],q[:,None]],1);yp=y[:,perm]
    pred=model(xp,yp)[:,-1];vals['order'].extend(((pred-target)**2).tolist());sens['order'].extend(((pred-base)**2).tolist())
    pred=model(x,-y)[:,-1];vals['sign'].extend(((pred+target)**2).tolist());sens['sign'].extend(((pred+base)**2).tolist())
    w2=torch.randn(128,D)/math.sqrt(D);yr=(xd*w2[:,None]).sum(-1)+SIGMA*torch.randn(128,n)
    pred=model(x,yr)[:,-1];vals['no_signal'].extend(((pred-target)**2).tolist());sens['no_signal'].extend(((pred-base)**2).tolist())
    ymix=y.clone();ymix[:,::2]=yr[:,::2];pred=model(x,ymix)[:,-1]
    vals['distractor'].extend(((pred-target)**2).tolist());sens['distractor'].extend(((pred-base)**2).tolist())
   else:
    methods['bayes'].extend((target**2).tolist());methods['one_step'].extend((target**2).tolist());methods['nearest'].extend((target**2).tolist())
  out[str(n)]={'transformer':{k:stats(v) for k,v in vals.items()},'baselines':{k:stats(v) for k,v in methods.items()},'paired_output_change':{k:stats(v) for k,v in sens.items()}}
  raw[str(n)]=vals
 return out,raw

def main():
 records=[]
 for seed in (11,22,33):
  torch.manual_seed(seed);model=Model();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
  trace=[];start=time.time()
  for step in range(STEPS):
   x,y,target,w=draw(BATCH,N);pred=model(x,y);loss=((pred-target)**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   if step%200==0 or step==STEPS-1:trace.append([step+1,float(loss.detach())])
  model.eval();metrics,raw=evaluate(model)
  torch.save({'state_dict':model.state_dict(),'seed':seed,'steps':STEPS,'d':D,'n':N,'width':WIDTH,'sigma':SIGMA},ROOT/f'checkpoint_{seed}.pt')
  rec={'seed':seed,'parameter_count':sum(p.numel() for p in model.parameters()),'steps':STEPS,'batch':BATCH,'trace':trace,'seconds':time.time()-start,'metrics':metrics}
  records.append(rec);(ROOT/'results.json').write_text(json.dumps(records,indent=2));(ROOT/f'raw_errors_{seed}.json').write_text(json.dumps(raw))
  print(seed,rec['seconds'],metrics['8']['transformer']['base'],flush=True)
 print('saved',ROOT/'results.json',flush=True)
if __name__=='__main__':main()
