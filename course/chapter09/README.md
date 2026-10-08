# Chapter 9 adaptation experiment

The chapter compares context and parameter changes using the same eight labeled examples per task. It states the update classes, target/old-task evaluations and reference models.

From the source-package root, with Python 3, NumPy and PyTorch installed, run:

```sh
python3 course/chapter09/code/adaptation_experiment.py
```

This loads `course/chapter08/code/checkpoint_22.pt` and the model implementation beside it, then performs the twelve-task comparison. It runs adaptation rather than retraining the Chapter 8 model. Task-level measurements are stored in `course/chapter09/code/results.json`.

Low-rank and full updates use 120 full-batch AdamW steps with learning rate 1e-3, zero weight decay and gradient clipping at one. These settings and the stopping point were fixed before query evaluation. The head-only row instead uses the stated exact ridge fit with lambda=0.1. Low-rank scale and factor initialization are specified in the chapter because they determine the induced matrix update.

Each target has 512 independent fresh queries. The old-family comparison uses the same 256 fresh tasks with eight demonstrations for every updated model. The unchanged network is the common retention baseline. Preserve these matched controls when reproducing the tables. Changing optimizer settings or selecting a penalty after inspecting test targets defines a new procedure and requires independent evaluation.

`code/summarize.py` writes the displayed table and interpretation from the saved results. If results change, regenerate `experiment_results.tex` through that program.
