"""Whole-pair reordering control for Problem P8; no refitting."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from pilot import Model,extract,metrics,M

def run(out):
    out=Path(out)
    data=dict(np.load(out/'test_episodes.npz'))
    shuffled={k:v.copy() for k,v in data.items()}
    rng=np.random.default_rng(2020)
    for row in shuffled['x']:
        pairs=row[:2*M].reshape(M,2).copy()
        row[:2*M]=pairs[rng.permutation(M)].reshape(-1)
    result=[]
    for seed in [21,22,23]:
        for arm in ['lookup','nuisance']:
            m=Model()
            c=torch.load(out/f'{arm}_{seed}.pt',weights_only=True,map_location='cpu')
            m.load_state_dict(c['state'])
            stats=metrics(extract(m,shuffled)['logits'],data['y'],data['seen'])
            result.append(dict(seed=seed,arm=arm,whole_pair_reordering=stats))
    (out/'interventions.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='shared_model/results')
    run(p.parse_args().out)

