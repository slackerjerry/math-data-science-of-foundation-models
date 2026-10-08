"""Regenerate the lecture figure from saved numerical results."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate(src,dest):
    src,dest=Path(src),Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    ev=json.loads((src/'evaluation.json').read_text())
    seeds=ev['protocol']['final_seeds']
    histories=[json.loads((src/f'lookup_{s}.json').read_text())['history'] for s in seeds]
    steps=np.array([v['step'] for v in histories[0]])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.75),gridspec_kw={'width_ratios':[1.4,1]})
    for key,label,color,style in [('seen_lookup','Key shown','#235d87','-'),
                                 ('unseen_lookup','Key unshown','#bb643c','--')]:
        y=np.array([[r[key]['nll'] for r in h] for h in histories])
        ax[0].plot(steps,y.mean(0),style,color=color,label=label,linewidth=1.8,marker='o',ms=3)
        ax[0].fill_between(steps,y.min(0),y.max(0),color=color,alpha=.12,linewidth=0)
    ax[0].axhline(np.log(5),color='0.5',linestyle=':',linewidth=1,label='Full-prior oracle (unshown)')
    ax[0].set(xlabel='Parameter updates',ylabel='Validation log loss (nats)',ylim=(0,3.2),
              title='Query-prediction training')
    ax[0].legend(fontsize=7.5,frameon=False,loc='upper right')
    xs=np.arange(3)
    for i,(arm,label,col) in enumerate([('random','Untrained','#808b94'),('nuisance','Last-value\nprediction','#bb643c'),
                                      ('lookup','Query\nprediction','#235d87')]):
        values=np.array([np.mean(r['probes']['h']['mse'][:3]) for r in ev['results'] if r['arm']==arm])
        ax[1].bar(i,values.mean(),color=col,width=.6)
        ax[1].scatter(np.full(3,i),values,s=14,facecolors='white',edgecolors='black',linewidths=.6,zorder=3)
    ax[1].set(xticks=xs,xticklabels=['Untrained','Last-value\nprediction','Query\nprediction'],
              ylabel='Test mean squared error',ylim=(0,1.02),title='Frozen hidden-state readouts')
    fig.tight_layout(w_pad=2)
    fig.savefig(dest/'training_pilot.pdf',bbox_inches='tight')
    fig.savefig(dest/'training_pilot.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    with (dest/'summary.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['checkpoint','seen_query_accuracy','hidden_binary_accuracy','hidden_binary_mse','own_objective_nll'])
        for arm in ['random','nuisance','lookup']:
            rows=[r for r in ev['results'] if r['arm']==arm]
            w.writerow([arm,np.mean([r['seen_lookup']['accuracy'] for r in rows]),
                np.mean([r['probes']['h']['accuracy'][:3] for r in rows]),
                np.mean([r['probes']['h']['mse'][:3] for r in rows]),
                np.mean([r['objective']['nll'] for r in rows])])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='shared_model/results')
    p.add_argument('--out',default='shared_model/figures')
    a=p.parse_args();generate(a.results,a.out)

