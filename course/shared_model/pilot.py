"""Small causal Transformer on permutation episodes. No downloaded model or data."""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, os, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

K, M, D, HEADS, LAYERS = 8, 3, 48, 4, 2
SPLIT_SEED = 20260929
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3*D)
        self.out = nn.Linear(D, D)
        self.ff1, self.ff2 = nn.Linear(D, 2*D), nn.Linear(2*D, D)
    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(self.n1(x)).chunk(3,dim=-1)
        q,k,v = [a.reshape(b,t,HEADS,d//HEADS).transpose(1,2) for a in (q,k,v)]
        s = q @ k.transpose(-1,-2) / (d//HEADS)**0.5
        mask = torch.ones(t,t,dtype=torch.bool,device=x.device).triu(1)
        p = s.masked_fill(mask, -torch.inf).softmax(-1)
        x = x + self.out((p@v).transpose(1,2).reshape(b,t,d))
        return x + self.ff2(F.gelu(self.ff1(self.n2(x))))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(2*K+1,D)
        self.pos = nn.Parameter(torch.randn(2*M+3,D)*0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm, self.head = nn.LayerNorm(D), nn.Linear(D,K)
    def forward(self, x):
        h = self.emb(x) + self.pos[:x.shape[1]]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h), h

def splits():
    p = np.array(list(itertools.permutations(range(K))),dtype=np.int64)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(p))
    return p, dict(train=order[:20160], validation=order[20160:30240], test=order[30240:])

def batch(rng, perms, ids, n, seen_probability=.75, task_ids=None, seen=None):
    task = rng.choice(ids,n) if task_ids is None else np.asarray(task_ids)
    keys = np.argsort(rng.random((n,K)),axis=1)[:,:M]
    vals = perms[task[:,None],keys]
    if seen is None:
        seen = rng.random(n)<seen_probability
    query = np.empty(n,dtype=np.int64)
    for i in range(n):
        query[i] = rng.choice(keys[i] if seen[i] else np.setdiff1d(np.arange(K),keys[i]))
    target = perms[task,query]
    tok = np.empty((n,2*M+2),dtype=np.int64)
    tok[:,:2*M:2],tok[:,1:2*M:2] = keys, vals+K
    tok[:,-2],tok[:,-1] = 2*K,query
    return dict(x=tok, y=target, nuisance=vals[:,-1], seen=np.asarray(seen),
                task=task, keys=keys, vals=vals, query=query)

def fixed_set(perms, ids, seed, tasks, per_task=4, all_seen=False):
    rng=np.random.default_rng(seed)
    task=np.repeat(rng.choice(ids,tasks,replace=False),per_task)
    seen=np.ones(len(task),dtype=bool) if all_seen else np.tile([True,True,False,False],tasks)
    return batch(rng,perms,ids,len(task),task_ids=task,seen=seen)

def tensor(a):
    return torch.from_numpy(a)

@torch.no_grad()
def extract(model, data):
    zs,hs=[],[]
    model.eval()
    for i in range(0,len(data['x']),256):
        z,h=model(tensor(data['x'][i:i+256]))
        zs.append(z[:,-1]);hs.append(h[:,-1])
    z,h=torch.cat(zs),torch.cat(hs)
    q=z.softmax(-1)
    return dict(logits=z.numpy(),h=h.numpy(),q=q.numpy(),
                output_mean=(q@model.head.weight).numpy())

def metrics(z,y,mask=None):
    if mask is not None: z,y=z[mask],y[mask]
    logq=z-z.max(1,keepdims=True)
    logq-=np.log(np.exp(logq).sum(1,keepdims=True))
    return dict(n=len(y),nll=float(-logq[np.arange(len(y)),y].mean()),
                accuracy=float((z.argmax(1)==y).mean()))

def oracle(perms, ids, data):
    pool=perms[ids]
    ans=np.zeros((len(data['x']),K),dtype=np.float64)
    counts=[]
    for i,(keys,vals,q) in enumerate(zip(data['keys'],data['vals'],data['query'])):
        compatible=(pool[:,keys]==vals).all(1)
        v=pool[compatible,q]
        ans[i]=np.bincount(v,minlength=K)/len(v)
        counts.append(len(v))
    return ans,np.array(counts)

def verification(model, data):
    model=copy.deepcopy(model).double()
    x=tensor(data['x'][:4]); y=tensor(data['y'][:4])
    loss=F.cross_entropy(model(x)[0][:,-1],y)
    loss.backward()
    rng=torch.Generator().manual_seed(310)
    params=list(model.parameters())
    directions=[torch.randn(p.shape,dtype=p.dtype,generator=rng) for p in params]
    norm=torch.sqrt(sum((d*d).sum() for d in directions))
    directions=[d/norm for d in directions]
    analytic=sum((p.grad*d).sum().item() for p,d in zip(params,directions))
    eps=1e-5
    originals=[p.detach().clone() for p in params]
    values=[]
    with torch.no_grad():
        for sign in (1,-1):
            for p,p0,d in zip(params,originals,directions): p.copy_(p0+sign*eps*d)
            values.append(F.cross_entropy(model(x)[0][:,-1],y).item())
        for p,p0 in zip(params,originals): p.copy_(p0)
        z=model(x)[0]
        suffix=torch.cat([x, tensor(data['y'][:4,None])+K],1)
        zp=model(suffix)[0][:,:x.shape[1]]
        causality=(z-zp).abs().max().item()
    numeric=(values[0]-values[1])/(2*eps)
    error=abs(analytic-numeric)/max(1e-10,abs(analytic),abs(numeric))
    assert error<1e-5,(analytic,numeric,error)
    assert causality<1e-12,causality
    group_grads={name:float(p.grad.norm()) for name,p in model.named_parameters()}
    assert all(np.isfinite(list(group_grads.values())))
    return dict(directional_autograd=analytic,directional_finite_difference=numeric,
                relative_error=error,causal_prefix_max_error=causality,gradient_norms=group_grads)

def train_one(seed, arm, steps, out, perms, split):
    torch.manual_seed(seed)
    model=Model()
    initial=copy.deepcopy(model.state_dict())
    val=fixed_set(perms,split['validation'],881,128)
    checks=verification(model,val)
    ref0=extract(model,val)['h']
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(.9,.999),eps=1e-8)
    rng=np.random.default_rng(5000+seed)
    history=[]
    checkpoints=sorted(set([0,100,300,600,1000,steps]))
    checkpoints=[a for a in checkpoints if a<=steps]
    start=time.perf_counter()
    train_seconds=0.
    for step in range(steps+1):
        if step in checkpoints:
            ex=extract(model,val)
            target=val['y'] if arm=='lookup' else val['nuisance']
            row=dict(step=step,elapsed_seconds=time.perf_counter()-start,
                objective=metrics(ex['logits'],target),
                seen_lookup=metrics(ex['logits'],val['y'],val['seen']),
                unseen_lookup=metrics(ex['logits'],val['y'],~val['seen']),
                relative_feature_change=float(np.linalg.norm(ex['h']-ref0)/np.linalg.norm(ref0)))
            history.append(row)
            print(json.dumps(dict(seed=seed,arm=arm,**row)),flush=True)
        if step==steps: break
        t0=time.perf_counter()
        model.train()
        data=batch(rng,perms,split['train'],128)
        target=data['y'] if arm=='lookup' else data['nuisance']
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(tensor(data['x']))[0][:,-1],tensor(target))
        loss.backward()
        optimizer.step()
        train_seconds+=time.perf_counter()-t0
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=dict(state=model.state_dict(),initial=initial,seed=seed,arm=arm,steps=steps,
                    architecture=dict(K=K,M=M,D=D,heads=HEADS,layers=LAYERS))
    torch.save(checkpoint,out/f'{arm}_{seed}.pt')
    record=dict(seed=seed,arm=arm,steps=steps,batch_size=128,learning_rate=.001,
                train_seconds=train_seconds,total_seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.parameters()),history=history,verification=checks,
                changes={k:float((v-initial[k]).norm()) for k,v in model.state_dict().items()})
    (out/f'{arm}_{seed}.json').write_text(json.dumps(record,indent=2))
    return record

def ridge(train_x,train_y,val_x,val_y,test_x):
    mean=train_x.mean(0)
    scale=train_x.std(0);scale=np.maximum(scale,1e-6)
    x=(train_x-mean)/scale
    xv=(val_x-mean)/scale
    xt=(test_x-mean)/scale
    ybar=train_y.mean(0)
    h=x.T@x/len(x); beta=x.T@(train_y-ybar)/len(x)
    options=[]
    for rho in [1e-4,1e-3,.01,.1,1.,10.]:
        coef=np.linalg.solve(h+rho*np.eye(h.shape[0]),beta)
        pred=xv@coef+ybar
        mse=((pred-val_y)**2).mean(0)
        options.append((rho,coef,mse))
    ix=np.argmin(np.stack([o[2] for o in options]),axis=0)
    w=np.stack([options[j][1][:,i] for i,j in enumerate(ix)],axis=1)
    return xt@w+ybar,[options[j][0] for j in ix],x@w+ybar

def nonlinear(train_x,train_y,val_x,val_y,test_x,seed):
    mean=train_x.mean(0);scale=np.maximum(train_x.std(0),1e-6)
    x,xv,xt=[torch.tensor((a-mean)/scale,dtype=torch.float32) for a in (train_x,val_x,test_x)]
    y,yv=[torch.tensor(a,dtype=torch.float32) for a in (train_y,val_y)]
    torch.manual_seed(8000+seed)
    net=nn.Sequential(nn.Linear(x.shape[1],32),nn.Tanh(),nn.Linear(32,y.shape[1]))
    optim=torch.optim.Adam(net.parameters(),lr=.005,weight_decay=.001)
    best=float('inf'); best_state=None; best_epoch=0
    for epoch in range(1,301):
        optim.zero_grad(set_to_none=True)
        F.mse_loss(net(x),y).backward();optim.step()
        if epoch%10==0:
            with torch.no_grad(): err=F.mse_loss(net(xv),yv).item()
            if err<best: best=err;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch
    net.load_state_dict(best_state)
    with torch.no_grad(): return net(xt).numpy(),best_epoch

def label_bits(values):
    return 2*((values[:,None]>>np.arange(3))&1)-1

def evaluate(out,seeds):
    perms,split=splits()
    fit=fixed_set(perms,split['train'],901,128,all_seen=True)
    val=fixed_set(perms,split['validation'],902,128,all_seen=True)
    test=fixed_set(perms,split['test'],903,512)
    test_seen={k:v[test['seen']] for k,v in test.items()}
    o,counts=oracle(perms,split['test'],test)
    assert np.allclose(o.sum(1),1)
    assert np.all(o[np.arange(len(o)),test['y']]>0)
    structural=np.zeros_like(o)
    for i in range(len(o)):
        if test['seen'][i]: structural[i,test['y'][i]]=1
        else: structural[i,np.setdiff1d(np.arange(K),test['vals'][i])]=1/(K-M)
    rng=np.random.default_rng(919)
    random_label=rng.choice([-1,1],len(perms))
    labels=[np.column_stack((label_bits(d['y']),random_label[d['task']])) for d in (fit,val,test_seen)]
    result=[]
    raw=dict(task_ids=test['task'],seen=test['seen'],y=test['y'],oracle=o,structural_oracle=structural,
             compatible_tasks=counts,probe_task_ids=test_seen['task'],probe_labels=labels[2])
    shuffled={k:v.copy() for k,v in test.items()}
    rng=np.random.default_rng(1010)
    for row in shuffled['x']:
        row[1:2*M:2]=rng.permutation(row[1:2*M:2])
    for seed in seeds:
        for arm in ['random','lookup','nuisance']:
            source='lookup' if arm=='random' else arm
            ckpt=torch.load(out/f'{source}_{seed}.pt',map_location='cpu',weights_only=True)
            model=Model();model.load_state_dict(ckpt['initial'] if arm=='random' else ckpt['state'])
            fitx,valx,tx=[extract(model,d) for d in (fit,val,test)]
            objective_y=test['nuisance'] if arm=='nuisance' else test['y']
            row=dict(seed=seed,arm=arm,objective=metrics(tx['logits'],objective_y),
                seen_lookup=metrics(tx['logits'],test['y'],test['seen']),
                unseen_lookup=metrics(tx['logits'],test['y'],~test['seen']),
                shuffled_seen_lookup=metrics(extract(model,shuffled)['logits'],test['y'],test['seen']),
                probes={})
            raw[f'{arm}_{seed}_logits']=tx['logits']
            for feature in ['h','q','output_mean']:
                pred,rhos,trainpred=ridge(fitx[feature],labels[0],valx[feature],labels[1],tx[feature][test['seen']])
                err=(pred-labels[2])**2
                correct=(pred>=0)==(labels[2]>0)
                row['probes'][feature]=dict(mse=err.mean(0).tolist(),accuracy=correct.mean(0).tolist(),
                    rho=rhos,train_mse=((trainpred-labels[0])**2).mean(0).tolist())
                raw[f'{arm}_{seed}_{feature}_pred']=pred
            pred,epoch=nonlinear(fitx['h'],labels[0][:,:3],valx['h'],labels[1][:,:3],
                                 tx['h'][test['seen']],seed)
            row['nonlinear_h']=dict(mse=((pred-labels[2][:,:3])**2).mean(0).tolist(),
                            accuracy=((pred>=0)==(labels[2][:,:3]>0)).mean(0).tolist(),epoch=epoch)
            raw[f'{arm}_{seed}_nonlinear_pred']=pred
            result.append(row)
            print(json.dumps(row),flush=True)
    def entropy(p):
        a=np.where(p>0,p,1.)
        return -(p*np.log(a)).sum(1)
    summary=dict(results=result,oracle=dict(
        seen_nll=float(entropy(o[test['seen']]).mean()),
        unseen_expected_nll=float(entropy(o[~test['seen']]).mean()),
        unseen_realized_nll=float(-np.log(o[np.arange(len(o)),test['y']])[~test['seen']].mean()),
        unseen_expected_accuracy=float(o[~test['seen']].max(1).mean()),
        structural_unseen_nll=float(np.log(K-M)),
        compatible_tasks_min=int(counts.min()),compatible_tasks_max=int(counts.max())),
        protocol=dict(train_tasks=len(split['train']),validation_tasks=len(split['validation']),test_tasks=len(split['test']),
            probe_fit_episodes=len(fit['x']),probe_fit_tasks=128,probe_validation_episodes=len(val['x']),
            evaluation_episodes=len(test['x']),evaluation_tasks=512,probe_seen_episodes=int(test['seen'].sum()),
            evaluation_seed=903,split_seed=SPLIT_SEED,final_seeds=seeds))
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'evaluation_arrays.npz',**raw)
    np.savez_compressed(out/'task_splits.npz',permutations=perms,**split)
    for name,data in [('probe_fit',fit),('probe_validation',val),('test',test)]:
        np.savez_compressed(out/f'{name}_episodes.npz',**data)
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('shared_model/results'))
    p.add_argument('--seeds',type=int,nargs='+',default=[21,22,23])
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--mode',choices=['train','evaluate','all','check'],default='all')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    perms,split=splits()
    assert len(set(split['train'])&set(split['validation']))==0
    assert len(set(split['train'])&set(split['test']))==0
    assert len(set(split['validation'])&set(split['test']))==0
    if a.mode=='check':
        torch.manual_seed(0)
        print(json.dumps(verification(Model(),fixed_set(perms,split['validation'],881,4)),indent=2))
        return
    (a.out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
        threads=torch.get_num_threads(),device='cpu'),indent=2))
    if a.mode in ('train','all'):
        for seed in a.seeds:
            for arm in ['lookup','nuisance']:
                train_one(seed,arm,a.steps,a.out,perms,split)
    if a.mode in ('evaluate','all'): evaluate(a.out,a.seeds)

if __name__=='__main__': main()

