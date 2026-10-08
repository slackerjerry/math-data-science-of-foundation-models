# Mathematical Data Science of Foundation Models

Computational companion to Jerry Yao-Chieh Hu's fifteen-chapter course. The course follows data, training objectives, model computation, learning, adaptation, inference, generation, and evaluation. These programs make selected finite models, worked calculations and controlled experiments executable.

**Release:** 2026.10.08, paired with lecture notes v50 (305 pages). This repository contains the corresponding code, small saved checkpoints, synthetic data, numerical records and figures. The lecture notes and syllabus are available through the [course website](https://jerryhu.page/courses/mathematical-data-science/), which requires access. The separate solution manual and editable LaTeX manuscript sources are not included.

The computational companion intentionally includes worked examples and executable answers to selected exercises. The eight generated `experiment_results.tex` tables are omitted from the initial repository snapshot; their programs can regenerate them locally from the saved results or stated experiments.

## Start here

Use Python 3.12. The exact packaging environment is recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_checks.py --suite numerical
```

On Windows, activate with `.venv\Scripts\activate`. Run commands from the repository root unless a chapter README says otherwise.

For the smallest start, no third-party packages are needed:

```bash
python scripts/run_checks.py --suite core
```

The runner uses a temporary copy of `course/`, so scripts that generate tables or figures cannot overwrite the supplied records. Logs and the execution summary go to `reports/`. Successful execution checks the selected implementations and their built-in assertions, not every theorem in the notes. Some checks contain worked exercise results. Attempt the corresponding problem before reading those outputs if you want independent practice.

## Neural-network activities

The numerical requirements do not install PyTorch. For the validated Linux CPU environment, install `requirements-torch.txt`. Use a PyTorch build suitable for your machine on other platforms. The existing Chapter 5 environment file is preserved at `course/shared_model/requirements.txt`, including its recorded CPU build. Historical versions are documented in [docs/REPRODUCING.md](docs/REPRODUCING.md).

With PyTorch installed, run the saved-checkpoint and derivative/masking checks without retraining:

```bash
python scripts/run_checks.py --suite torch
```

The `torch` suite does not repeat the `numerical` suite. It checks the Chapter 5 initial-model derivatives and causal mask, evaluates one saved Chapter 8 model, and evaluates the three saved Chapter 12 fields. Evaluation may refit a readout or regenerate measurements. It is distinct from pretraining a network. The full training and adaptation commands are documented separately.

## Find the code for a chapter

[docs/CHAPTER_MAP.md](docs/CHAPTER_MAP.md) maps every chapter to its programs and retained data. [docs/REPRODUCING.md](docs/REPRODUCING.md) explains saved-result use, fresh runs, output locations and the distinction between short activities and the full recorded experiments.

The relative `course/` layout is intentional. Chapter 9 loads the Chapter 8 model and checkpoint. The shared pilot supports Chapters 3-6 and Chapter 5 P8/P9. Moving a single script out of this tree can break these dependencies.

## Data, checkpoints and scope

- The course experiments use synthetic finite or continuous models. The supplied code does not require an external dataset or a large pretrained model.
- Saved checkpoints and small numerical data files are included. No Git LFS setup is required for this snapshot.
- Seeds, inputs, task splits and comparison protocols are retained. Hardware and software changes can alter numerical results. A matching seed alone is not a guarantee of identical training.
- The retained six 1,600-step Chapter 5 runs are not the same experiment as the 300-step exercise. Chapter 8, Chapter 9 and Chapter 12 use their own protocols.
- Exact identities, Monte Carlo checks and trained-model observations serve different purposes. The notes explain which conclusions each supports.

## Course materials and citation

The [course website](https://jerryhu.page/courses/mathematical-data-science/) provides the lecture notes and syllabus to authorized readers. `CITATION.cff` provides citation metadata for this computational companion.

Repository: [math-data-science-of-foundation-models](https://github.com/slackerjerry/math-data-science-of-foundation-models).

No public reuse license has been selected. See [LICENSE_STATUS.md](LICENSE_STATUS.md) for the current licensing status. [THIRD_PARTY.md](THIRD_PARTY.md) identifies the external software dependencies. Contribution guidance is in [CONTRIBUTING.md](CONTRIBUTING.md).

The course's ungraded problem-pool and oral-examination policies are defined by the syllabus. This repository adds no submission requirements.
