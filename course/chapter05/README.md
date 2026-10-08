# Chapter 5 calculations

Run these commands from the source package's `course/` directory.

- `python3 chapter05/worked_examples.py` checks coordinate changes, ridge fits, spectral updates, stochastic moments, and log-loss derivatives.
- `python3 chapter05/dynamics_examples.py` checks factorized flows and residual dynamics.
- `python3 chapter05/feature_examples.py` checks feature rotation and the pretraining-dependent update in P7.
- `python3 chapter05/geometry_examples.py` checks weighted rank selection, record reweighting, and frozen-Jacobian trajectories.

The first three use only Python's standard library. The fourth also uses NumPy. Some calculations reproduce exercise results. Derive and predict the results before consulting those outputs.

For the causal-Transformer experiment and P8, follow `shared_model/README.md`. It separates the supplied six 1,600-step runs from the reduced 300-step exercise, and gives the derivative/prefix checks and output fields. The linear pretext learner used in P9 is also described there.

## Reading and using the experiment

Section 5.10 first compares paired query-prediction and last-value-prediction runs, their use of displayed bindings, and frozen hidden-row readouts for the same three bit labels. Novel-prefix and value-shuffling checks accompany that interpretation. Feature displacement, alternative readouts and random task labels then test what distinguishes the learned representations. The paired task bootstrap changes the weights of complete test tasks while keeping the trained models fixed.

The remaining units treat an unshown query's conditional distribution and derivative/prefix checks. P8 uses these units without changing the course's optional problem-pool policy. The shared-model README provides the commands and exact settings. The prose and formulas for the experiment are maintained in `pilot_experiment.tex`, included by `chapter.tex` in both standalone and combined builds.

## Geometry figures

Run `python3 chapter05/geometry_figures.py` from `course/` to regenerate the two PDF figures and their PNG previews in `chapter05/figures/`. The script requires NumPy and Matplotlib (checked with 2.3.5 and 3.10.8). Use `--out /path/to/empty-directory` to generate a comparison copy without replacing the supplied assets. The main PDF build uses the supplied vector PDFs, so figure regeneration is not required to compile the notes.

- `scale_tilt.pdf` uses the two maps already calculated in Section 5.7: `(2,0)` and `(1,1)`, compared with `(1,0)`. It draws their retained spans and the omitted component of the task. Both view excesses are one, while refitting the optimal task head gives risks zero and one half.
- `feature_rotation.pdf` uses the exact balanced population trajectory in Section 5.8, initialized at `a=1`, `b=(1,0)`. The direction panel displays the unit vector `u=b/||b||`, not `b`. The risk panel minimizes over a new scalar head at each time.

The script writes `geometry_checks.json` and `rotation_values.csv`. It checks the projection risks, scale invariance, the sum and monotonicity of the two rotation risks, and agreement of the closed form with an independent RK4 integration of the original parameter ODEs. These checks supplement the chapter's derivation. They use no saved Transformer data, training runs or exercise answers.
