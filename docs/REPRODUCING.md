# Reproducing the computational activities

## Three kinds of execution

1. **Exact and numerical checks:** `python scripts/run_checks.py --suite numerical` runs 24 scripts, including 16 standard-library checks. It also parses every included Python file. The Chapter 6 simulation keeps its original default repetitions.
2. **Saved-model evaluation:** `python scripts/run_checks.py --suite torch` runs three commands with the included checkpoints. It does not retrain. Chapter 8 seed 11 and all three Chapter 12 fields are evaluated. Other saved seeds are available through their chapter evaluator.
3. **New training or adaptation:** use the explicit commands below. These are new executions of the stated protocols. They are not part of the quick checks, and runtimes depend on the machine.

All suites execute a disposable copy of the course tree. Use `--output /path/to/reports` to choose where logs and summaries are kept. The course scripts themselves are unchanged. Some scripts intentionally regenerate JSON, TeX tables or figure files next to themselves. The eight generated `experiment_results.tex` tables are omitted from the initial repository snapshot; the programs regenerate them as outputs and do not require them as inputs. If running them directly, use a separate copy or Git branch and compare changes before retaining them.

## Environment

`requirements.txt` pins the numerical packages used to check this release. `requirements-torch.txt` adds the CPU PyTorch version used for packaging validation:

```bash
python -m pip install -r requirements-torch.txt
```

The saved Chapter 5 runs record Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5 and Matplotlib 3.10.8. The later chapter experiments record PyTorch 2.14.1+cpu. Their original environment records and the Chapter 5 requirements are retained. Installing the packaging-validation environment is not a claim that every historical training run will be byte-identical. No GPU is required by these small examples.

## Chapter 5

From this repository's root, the short activity trains two models at one seed:

```bash
python course/shared_model/pilot.py --mode train --steps 300 --seeds 31 --out runs/ch5_short
```

The recorded full pilot uses six runs, two targets at each of three seeds:

```bash
python course/shared_model/pilot.py --mode all --steps 1600 --seeds 21 22 23 --out runs/ch5_full
python course/shared_model/diagnostics.py --out runs/ch5_full
python course/shared_model/interventions.py --out runs/ch5_full
python course/shared_model/plot_results.py --results runs/ch5_full --out runs/ch5_figures
```

To evaluate existing checkpoints, first copy `course/shared_model/results/` to a new directory and pass that path with `--mode evaluate`. This refits the prescribed probes and writes evaluation records. `diagnostics.py` computes the task bootstrap from the saved paired arrays and checks prefix novelty. The supplied values must not be replaced by a reduced training run under the same name.

The exact geometry figures need no training:

```bash
python course/chapter05/geometry_figures.py --out runs/ch5_geometry
```

A separate Gaussian reduced-rank experiment supports Chapters 4-6:

```bash
python course/shared_model/linear_pretext.py --output-dir runs/linear_pretext
```

Its default is 600 independent corpora per sample size. It is not a Transformer experiment. See the [shared pilot README](../course/shared_model/README.md) for targets, episode layout, task splits, labels, probe selection, pairing, bootstrap and the full numerical settings.

## Chapters 8, 9 and 12

These scripts use paths relative to their source files. To preserve the supplied checkpoints and saved results, copy the whole `course/` directory to `runs/course/` before running them directly.

```bash
mkdir -p runs
cp -R course runs/course
python runs/course/chapter08/code/evaluate_checkpoint.py --seed 22
python runs/course/chapter09/code/adaptation_experiment.py
python runs/course/chapter12/code/flow_experiment.py --evaluate-only
```

- Chapter 8's `icl_experiment.py` without an evaluation wrapper trains three 1,200-step models and replaces their saved outputs.
- Chapter 9 loads Chapter 8 seed 22 and runs twelve-task adaptation. This is parameter fitting, not base-model retraining.
- Chapter 12 without `--evaluate-only` trains three 4,000-step fields. Evaluation compares 8, 32 and 128 integration steps using fixed starting samples. It is a numerical-resolution comparison, not a proof of convergence.

See the respective [Ch8](../course/chapter08/README.md), [Ch9](../course/chapter09/README.md) and [Ch12](../course/chapter12/README.md) READMEs for the preserved protocols.

## Results and provenance

The `course/` tree includes the original saved arrays, seeds, task splits, training histories, model weights and figures used by the notes. `docs/UPSTREAM_FILES.json` records their hashes against source v24. New execution logs are separate in `reports/` or the selected output directory.

The release validation record states exactly which checks were run while packaging. Reusing a saved result is not a new training run, and a passing smoke check is not a mathematical audit or a replication of every research claim.
