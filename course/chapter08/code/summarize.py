"""Generate the student experiment table from the three saved measured runs."""
from pathlib import Path
import json,math
r=Path(__file__).resolve().parent
z=json.loads((r/'results.json').read_text())
lines=[r'\begin{center}',r'\begin{tabular}{@{}rrrrrr@{}}',r'\toprule',r'Examples & Bayes & One step & Nearest & Zero & Transformer range\\',r'\midrule']
for n in ('0','2','4','8'):
 b=z[0]['metrics'][n]['baselines'];t=[v['metrics'][n]['transformer']['base']['mean'] for v in z]
 near=f"{b['nearest']['mean']:.3f}" if n!='0' else r'---'
 lines.append(f"{n} & {b['bayes']['mean']:.3f} & {b['one_step']['mean']:.3f} & {near} & {b['zero']['mean']:.3f} & {min(t):.3f}-{max(t):.3f}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}', 'Entries are measured mean squared errors on the saved test prompts. The range spans three training seeds, not a confidence interval. At eight examples, the three Transformer errors are '+', '.join(f"{v['metrics']['8']['transformer']['base']['mean']:.3f}" for v in z)+'. Their separate test-prompt standard errors are '+', '.join(f"{v['metrics']['8']['transformer']['base']['se']:.3f}" for v in z)+'.', '', r'\begin{center}',r'\begin{tabular}{@{}lrr@{}}',r'\toprule',r'Length-eight intervention & MSE range & Paired discrepancy range\\',r'\midrule']
for name,label in [('order','Pair permutation'),('sign','Sign relabeling'),('no_signal','No task signal'),('distractor','Mixed tasks')]:
 a=[v['metrics']['8']['transformer'][name]['mean'] for v in z];b=[v['metrics']['8']['paired_output_change'][name]['mean'] for v in z]
 lines.append(f'{label} & {min(a):.3f}-{max(a):.3f} & {min(b):.3f}-{max(b):.3f}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}',r'The paired discrepancy is the mean squared change in predictions, except under sign relabeling, where it is the mean squared sum of old and new predictions. Both should be zero for the corresponding exact symmetry. The two altered-task interventions instead change the evidence model.', '', 'At eight examples, the trained models improve on predicting zero and on the nearest-neighbor rule, but remain well above the known-prior Bayes risk. Their errors also exceed the best scalar one-step baseline in these runs. Pair order and sign transformations expose nonzero departures from symmetries respected by both analytic rules. Thus training has acquired useful task-dependent behavior without reproducing either rule exactly. The no-signal intervention removes the shared task: responding to those unrelated labels can raise risk above the zero predictor. This connects a measured failure to the information assumed by the posterior calculation.']
(r.parent/'experiment_results.tex').write_text('\n'.join(lines)+'\n')
h=lambda p:-p*math.log(p)-(1-p)*math.log(1-p)
checks={'entropy_before':h(.9),'entropy_after':.82*h(81/82)+.18*h(.5),'gap':h(.9)-(.82*h(81/82)+.18*h(.5)),'length_excess_2_to_8':450/637,'length_excess_8_to_2':1800/1183}
print(json.dumps(checks,indent=2))
print('torch',__import__('torch').__version__)
