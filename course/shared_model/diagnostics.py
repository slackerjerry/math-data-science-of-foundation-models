"""Additional diagnostic: exact training-prefix overlap and conditional oracle risks."""
import json
from pathlib import Path
import numpy as np
from pilot import splits,batch

def encode(x):
    return x.astype(np.int64) @ (17**np.arange(x.shape[1],dtype=np.int64))

def audit(out):
    out=Path(out)
    p,s=splits()
    ev=json.loads((out/'evaluation.json').read_text())
    data=dict(np.load(out/'test_episodes.npz'))
    arr=dict(np.load(out/'evaluation_arrays.npz'))
    target_codes=encode(data['x'])
    rows=[]
    for seed in ev['protocol']['final_seeds']:
        steps=json.loads((out/f'lookup_{seed}.json').read_text())['steps']
        rng=np.random.default_rng(5000+seed)
        codes=[]
        for step in range(steps):
            codes.append(encode(batch(rng,p,s['train'],128)['x']))
        unique=np.unique(np.concatenate(codes))
        novel=~np.isin(target_codes,unique)
        stats=dict(seed=seed,unique_train_prefixes=len(unique),
            seen_prefix_overlap=int((~novel & data['seen']).sum()),
            novel_seen_count=int((novel & data['seen']).sum()),
            unseen_prefix_overlap=int((~novel & ~data['seen']).sum()),models={})
        for arm in ['random','lookup','nuisance']:
            z=arr[f'{arm}_{seed}_logits'].astype(np.float64)
            logq=z-z.max(1,keepdims=True)
            logq-=np.log(np.exp(logq).sum(1,keepdims=True))
            o=arr['oracle']
            structural=arr['structural_oracle']
            mask=~data['seen']
            logp=np.log(np.where(o>0,o,1))
            logps=np.log(np.where(structural>0,structural,1))
            correct=z.argmax(1)==data['y']
            a=dict(novel_seen_accuracy=float(correct[novel & data['seen']].mean()),
                unseen_oracle_kl=float((o*(logp-logq)).sum(1)[mask].mean()),
                unseen_structural_kl=float((structural*(logps-logq)).sum(1)[mask].mean()),
                unseen_forbidden_mass=float((np.exp(logq)*(structural==0)).sum(1)[mask].mean()))
            stats['models'][arm]=a
        rows.append(stats)
    # Paired task bootstrap, averaged over the three fixed training seeds and three fixed binary tasks.
    labels=arr['probe_labels'][:,:3]
    byarm={}
    for arm in ['random','lookup','nuisance']:
        pred=np.stack([arr[f'{arm}_{seed}_h_pred'][:,:3] for seed in ev['protocol']['final_seeds']])
        errors=((pred-labels[None])**2).mean((0,2)).reshape(512,2).mean(1)
        byarm[arm]=errors
    rng=np.random.default_rng(447)
    ix=rng.integers(0,512,(4000,512))
    intervals={}
    for comparison in ['random','nuisance']:
        diff=byarm[comparison]-byarm['lookup']
        intervals[comparison]=dict(mse_reduction=float(diff.mean()),
            task_bootstrap_95_percent_interval=np.quantile(diff[ix].mean(1),[.025,.975]).tolist(),
            note="Paired resampling of 512 task clusters. Fixed split, seeds, binary tasks and trained models.")
    report=dict(prefix_checks=rows,paired_mse_reduction=intervals)
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='shared_model/results')
    audit(parser.parse_args().out)

