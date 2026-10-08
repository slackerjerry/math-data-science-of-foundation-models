"""Evaluate a saved chapter checkpoint without retraining."""
import argparse,json
from pathlib import Path
import torch
from icl_experiment import Model,evaluate,ROOT
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(11,22,33),default=11);a=p.parse_args()
ck=torch.load(ROOT/f'checkpoint_{a.seed}.pt',map_location='cpu',weights_only=True)
m=Model();m.load_state_dict(ck['state_dict']);m.eval();metrics,raw=evaluate(m)
out=ROOT/f'reevaluation_{a.seed}.json'
out.write_text(json.dumps({'seed':a.seed,'steps':ck['steps'],'torch':torch.__version__,'metrics':metrics},indent=2))
print(out)
