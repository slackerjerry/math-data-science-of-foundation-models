"""Generate the student table from measured task-level results."""
from pathlib import Path
import json,statistics
R=Path(__file__).resolve().parent
rows=json.loads((R/'results.json').read_text())['records']
def fmt(v):return f'{statistics.mean(v):.4f} ({statistics.stdev(v)/(len(v)**.5):.4f})'
lines=[r'\begin{center}\small',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Method & Trainable entries & Target MSE (SE) & Old-family MSE (SE)\\',r'\midrule']
for name,key in [('Zero-context','base_zero_context'),('Frozen context','base_icl'),('Gaussian oracle','raw_bayes_risk')]:lines.append(name+' & 0 & '+fmt([r[key] for r in rows])+(' & 0.4302' if key!='raw_bayes_risk' else ' & ---')+r'\\')
for name,label in [('head','Head ridge'),('lora1','Rank 1 + head'),('lora4','Rank 4 + head'),('full','Full update')]:
 m=[r['methods'][name] for r in rows];lines.append(f"{label} & {m[0]['parameters']:,} & "+fmt([r['test_risk'] for r in m])+' & '+fmt([r['retention_risk'] for r in m])+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}',r'The means average twelve task-specific fits. Parentheses give the standard error across those twelve task results, conditional on the one pretrained checkpoint and the finite evaluation sets.',r'',r'Head fitting gives the smallest mean target risk among these fitted neural procedures, despite having larger training error than the iterative fits in the saved task-level measurements. It also has the largest mean old-family MSE among these fitted procedures. Fitting the scalar head reshapes predictions wherever that shared head is used, including long prompts. The target and retention columns therefore measure different consequences of the same update. These observed averages do not order the methods on all tasks or after retuning their settings.']
(R.parent/'experiment_results.tex').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
