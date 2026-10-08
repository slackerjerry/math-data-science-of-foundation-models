# Shared causal-Transformer pilot

This is a synthetic training experiment for Chapter 5. It trains a complete, small causal Transformer. The data and weights are generated locally.

## Files and dependencies

The source package contains the programs, figure inputs, all six final checkpoints, initial states, exact test/probe inputs, task lists, raw predictions, and the reduced-run results. The commands below run from the `course/` directory inside the lecture-note source package. No additional checkpoint archive is needed.

Tested with Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5, and Matplotlib 3.10.8 on Linux CPU. The code sets two CPU threads. No GPU or external dataset is required.

First change into `course/`. In a Python environment of your choice:

    python3 -m pip install -r shared_model/requirements.txt

The requirements select the CPU PyTorch wheel. Matplotlib is needed only for plotting. Compilation of the supplied PDFs does not require the Python training dependencies.

## Execute

A reduced end-to-end run trains both objectives in roughly 3 seconds each on the recorded machine:

    python3 shared_model/pilot.py --mode train --steps 300 --seeds 31 --out shared_model/my_small_run

This writes new checkpoints and validation results in the chosen directory. The supplied reduced_run/ retains the executed seed-31 results. The model's shown-key validation accuracy reaches 100% in that run. The shorter run has worse unshown-key loss than the full budget.

Reproduce the six full runs and evaluate them in a fresh directory:

    python3 shared_model/pilot.py --mode all --steps 1600 --seeds 21 22 23 --out shared_model/my_results

Evaluate the supplied final checkpoints without pretraining again:

    python3 shared_model/pilot.py --mode evaluate --seeds 21 22 23 --out shared_model/results

Evaluation refits the prescribed probes and rewrites the numerical evaluation files. To preserve the delivered records, copy results/ to another directory and use that path.

Additional controls and the figure:

    python3 shared_model/diagnostics.py --out shared_model/results
    python3 shared_model/interventions.py --out shared_model/results
    python3 shared_model/plot_results.py --results shared_model/results --out shared_model/figures

The diagnostics reconstruct the exact training input stream from its seeds and count prefix overlap. They also compute conditional oracle KL and paired task-bootstrap intervals. Interventions reorder complete pairs while keeping labels fixed. Value-only shuffling is included in pilot.py evaluation.

A derivative and masking check without training:

    python3 shared_model/pilot.py --mode check --out shared_model/check

## Derivative and prefix checks

Before training, the program runs these checks on the initial model in double precision:

- **Directional derivative.** The check loss is mean query-target log loss on the first four validation episodes. A Gaussian direction over all parameters is drawn with seed 310 and normalized to total Euclidean norm one. The centered finite difference uses delta=1e-5. Its relative discrepancy from the backpropagated directional derivative divides their absolute difference by the largest of 1e-10 and the two absolute derivative values. The required discrepancy is below 1e-5. Parameters are then restored. The result field is `verification.relative_error`.
- **Prefix invariance.** Append the answer after each original eight-token context. Compare all logits at the eight original positions before and after appending. Their maximum absolute discrepancy must be below 1e-12. The result field is `verification.causal_prefix_max_error`.

These are numerical implementation tolerances, not statistical accuracy guarantees. Chapter 5 explains the different questions the two checks test. P8(b) asks for both recorded discrepancies and the validation query-target metrics in `seen_lookup`. Use the 300-step command above for that exercise, not the six full runs. No external pretrained model or dataset download is required.

## Model and data

- Two pre-norm blocks, width 48, four heads of width 12, GELU MLP width 96, final LayerNorm. Trainable token and absolute position embeddings, no dropout, 39,656 parameters.
- Input tokens 0-7 are keys, 8-15 are values, and 16 is QUERY. An input is three ordered, distinct key-value pairs, QUERY, and one key. Only the final input position is scored. Its 8-output head predicts values 0-7.
- A task is a complete permutation on 8 symbols. Split seed 20260929 shuffles all 40,320 permutations into 20,160 training, 10,080 validation, and 10,080 test tasks. Episodes are then generated.
- Each training episode selects a shown query with probability 0.75, otherwise an unshown query. Its three distinct keys are sampled in a random order. The query objective predicts the permutation's query value. The control predicts the last demonstrated value.
- Within each seed, both objectives share their initialization and sampled inputs. Final seeds are 21, 22, 23. Seed 11 was the separate development run.
- Adam: step size 0.001, beta1=0.9, beta2=0.999, epsilon=1e-8. No pretraining weight decay, schedule, clipping, or dropout. 1,600 steps × 128 episodes. Reported training time includes sampling and updates, and excludes evaluation/probe fitting.

The six recorded full runs took 15.6-18.4 seconds each on the two-thread CPU environment above. This is an execution record, not a runtime guarantee for another machine or software version.

## Evaluation

The held-out set has 512 distinct test tasks × 4 episodes: two shown and two unshown queries for each task. Never count those four episodes as four independent task draws. The same set is used across all seeds and models.

The full-prior oracle is uniform on unused values for an unshown key. The exact split-conditioned oracle enumerates only compatible permutations in the generating task list. It has additional knowledge of that list. Both are saved, and their conditional entropies must not be confused with a single-label realized test loss.

Three binary downstream questions label the three bits of the query value by -1/+1. Only shown-query episodes enter these probes. They test accessible reuse of the pretext value, not new semantics beyond that value.

Readout fitting and selection use 512 fitting episodes from 128 training tasks and 512 validation episodes from 128 validation tasks. All four generated episodes per fitting/validation task are shown-key episodes. Test probes use 1,024 episodes from 512 test tasks.

- h is the final normalized query hidden row.
- q is the softmax probability row.
- output_mean is q U^T, using the trained head projection and excluding its bias.
- Affine probes subtract fitting-coordinate means and divide by fitting population standard deviations (normalization 1/n), floored at 1e-6. The same transformation is used on validation and test data. Responses are centered on fitting means to leave an unpenalized intercept. Each response separately selects rho from 1e-4, 1e-3, 0.01, 0.1, 1, 10 by validation MSE, with the first/smallest rho breaking exact ties. The selected fitting-data head is not refitted on validation data. Test scores are classified at zero.
- The nonlinear h probe is a 32-unit tanh MLP with three outputs. It uses Adam 0.005, weight decay 0.001, 300 full-batch epochs, and selects the epoch by validation MSE every 10 epochs.
- A fourth affine-probe label is one independently sampled sign per complete task, kept constant across its episodes and across model comparisons.

The script saves all seed/task predictions. Its labels are fresh simulated responses, and all conclusions concern this finite family and protocol. Failure of either finite probe class does not establish absence of information.

## Paired task bootstrap

`diagnostics.py` reads the saved affine-probe predictions in `results/evaluation_arrays.npz`. It does not retrain a model or refit a head for this calculation.

1. For each of the 512 test tasks, average squared error over its two shown-query episodes, the three bit responses and the three fixed training seeds.
2. Subtract the query-trained hidden-row error from the random-feature hidden-row error for that same task.
3. Draw 512 task indices with replacement for each of 4,000 replicates and average the selected reductions. Use the same indices for both representations. The supplied implementation uses NumPy's generator with seed 447.
4. Take the 0.025 and 0.975 quantiles of those means. The saved result is approximately [0.810, 0.850].

This calculation conditions on the supplied task split, bit questions and three trained models. It resamples tasks, not individual episodes or retraining seeds. The chapter explains why these are different sources of variation.

## Reuse

Each .pt file contains the architecture dimensions, final state, initial state, seed, objective, and step count. Load with torch.load(..., weights_only=True, map_location="cpu") and Model.load_state_dict. These are inference checkpoints. Optimizer state is not included.

task_splits.npz identifies every complete task. The episode NPZ files preserve tokens, query, demonstrations, target, task ID, and shown-key status. Future context/adaptation experiments should reuse these boundaries and distinguish new examples from repeated task information.

summary.csv and training_pilot.pdf are generated from saved results. The shaded training ranges are minima and maxima across three seeds, not confidence intervals. Bootstrap intervals condition on the fixed split, binary questions, and trained models.

Training a model again with seed 21 reproduced every final parameter exactly on the recorded CPU environment. Other software versions or hardware may differ numerically.


## Linear pretext learner for Chapters 4-6

This separate experiment fits the exact empirical reduced-rank least-squares estimator, extracts its row space, freezes it, and fits OLS heads using fresh independent task labels. It is not an evaluation of the Transformer checkpoints.

    OPENBLAS_NUM_THREADS=1 python3 shared_model/linear_pretext.py

The default run uses seed 930202611 and 600 independent corpora at each of 16, 64, 256, 1024, and 4096 pretext pairs. It compares label budgets 10 and 100 and learned, random, oracle, and raw representations. `linear_results/results.json` records the model, configuration, numerical algebra checks, conditional-risk averages, actual fitted-head risks, and Monte Carlo standard errors. `risk_comparison.csv` and `linear_transfer.pdf` supply the table and figure. The plotted curves average exact conditional head risks. The separate fitted-head measurements are reported in the table.

NumPy and Matplotlib suffice. No downstream labels enter the pretext objective. No finite test set is required for the reported risk: in the specified isotropic Gaussian model, the squared error in fitted raw-coordinate coefficients equals exact fresh-input risk. All claims concern that model and learner.
