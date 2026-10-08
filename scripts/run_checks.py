#!/usr/bin/env python3
"""Run selected course checks in an isolated copy, preserving supplied records."""
from pathlib import Path
import argparse, ast, importlib.metadata, json, os, shutil, subprocess, sys, tempfile, time
ROOT=Path(__file__).resolve().parents[1]
CORE=[
 ['verify_examples.py'],
 ['chapter02/worked_examples.py'],['chapter02/generation_examples.py'],
 ['chapter03/worked_examples.py'],['chapter03/retrieval_examples.py'],
 ['chapter03/composition_examples.py'],['chapter03/token_routing_examples.py'],
 ['chapter04/worked_examples.py'],['chapter04/research_examples.py'],
 ['chapter04/mean_distribution_example.py'],['chapter04/finite_candidates_examples.py'],
 ['chapter05/worked_examples.py'],['chapter05/dynamics_examples.py'],['chapter05/feature_examples.py'],
 ['chapter11/code/verify.py'],['chapter15/code/pipeline_experiment.py'],
]
NUMERICAL=[
 ['chapter05/geometry_examples.py'],['chapter05/geometry_figures.py'],
 ['chapter06/worked_examples.py'],['chapter06/interpolation_examples.py'],
 ['chapter07/code/verify.py'],['chapter10/code/verify.py'],
 ['chapter13/code/measurement_experiment.py'],['chapter14/code/evaluation_experiment.py'],
]
TORCH=[
 ['shared_model/pilot.py','--mode','check','--out','check_outputs'],
 ['chapter08/code/evaluate_checkpoint.py','--seed','11'],
 ['chapter12/code/flow_experiment.py','--evaluate-only'],
]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['core','numerical','torch'],default='core')
    parser.add_argument('--output',type=Path,default=ROOT/'reports')
    args=parser.parse_args()
    output=args.output.resolve()/args.suite
    output.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'course').rglob('*.py'))+[Path(__file__)]
    for p in files: ast.parse(p.read_text(),filename=str(p))
    cases=CORE if args.suite=='core' else CORE+NUMERICAL if args.suite=='numerical' else TORCH
    versions={}
    for name in ['numpy','scipy','matplotlib','torch']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    report={'suite':args.suite,'python':sys.version,'packages':versions,
            'syntax_checked_files':len(files),'retraining':False,'cases':[]}
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='math_ds_checks_') as temp:
        work=Path(temp)/'course';shutil.copytree(ROOT/'course',work)
        for i,case in enumerate(cases,1):
            start=time.monotonic();log=output/f'{i:02d}_{Path(case[0]).stem}.log'
            command=[sys.executable,*case]
            with log.open('w') as stream:
                try:
                    result=subprocess.run(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
                    code=result.returncode
                except subprocess.TimeoutExpired:
                    code=124;stream.write('\nTimed out after 300 seconds.\n')
            row={'command':['python',*case],'exit_code':code,'seconds':round(time.monotonic()-start,3),'log':log.name}
            report['cases'].append(row)
            (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'[{i}/{len(cases)}] {case[0]}: exit {code}',flush=True)
    failures=[row for row in report['cases'] if row['exit_code']]
    print(f'{len(cases)-len(failures)}/{len(cases)} completed successfully. Logs: {output}')
    return 1 if failures else 0

if __name__=='__main__':raise SystemExit(main())
