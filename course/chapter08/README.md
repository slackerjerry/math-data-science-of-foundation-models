# Chapter 8 regression experiment

This experiment trains causal Transformers on real-valued linear-regression tasks. It is separate from the Chapter 5 language-token pilot. The chapter states the task distribution, supervision, matched baselines and interventions.

## Evaluate saved models

Requirements: Python 3, NumPy and PyTorch. From the source-package root:

```sh
python3 course/chapter08/code/evaluate_checkpoint.py --seed 11
```

Seeds 11, 22 and 33 select the three included `checkpoint_SEED.pt` files in `course/chapter08/code/`. Evaluation writes a new JSON report beside the checkpoint. It loads saved weights and does not retrain. No external model or dataset download is required.

`course/chapter08/code/results.json` contains each seed's `metrics` by prompt length. The files `raw_errors_11.json`, `raw_errors_22.json` and `raw_errors_33.json` preserve per-prompt squared losses by length and intervention. Use paired measurements when comparing transformed and original predictions.

## Training settings

`icl_experiment.py` runs the three 1,200-step training runs. Each uses batch size 128, AdamW with learning rate 3e-4, zero weight decay and gradient clipping at one. The model has two blocks, width 64, four heads, feed-forward width 128, learned positions and no dropout. The source fixes random seeds and writes checkpoints and measurements. Running this program is retraining, unlike the evaluator above.

The scientific comparison uses new task vectors at evaluation, fixed prior and noise, matched input/label evidence, and all prefix lengths from zero through eight supervised during training. Changing the training or intervention protocol defines a new experiment. `summarize.py` generates the displayed tables from saved results. It should remain synchronized with `experiment_results.tex` if numerical results change. The verification code and summaries contain reference results, so solve the analytic exercises before consulting them.
