# Chapter 12 experiments

From the source-package root, evaluate the supplied checkpoints with:

```bash
python course/chapter12/code/flow_experiment.py --evaluate-only
```

The program requires NumPy, SciPy, PyTorch and Matplotlib. It uses the three local `field_seed12.pt`, `field_seed23.pt` and `field_seed34.pt` files in `course/chapter12/code`. No download is required. Omitting `--evaluate-only` trains the three models again and replaces the saved checkpoints and results. Preserve copies before starting a new experiment.

The saved training protocol uses fresh independent Gaussian-base and two-mode-target pairs, independent uniform times, a 2-64-64-1 SiLU network, batch size 256, and 4,000 Adam updates at learning rate 0.001. Training seeds are 12, 23 and 34. The program fixes two PyTorch CPU threads. The saved `results.json` records software versions, training traces, field errors, endpoint comparisons and the precise protocol.

Evaluation reuses 12,000 Gaussian initial values from seed 12001 across exact and learned fields. The reference endpoint is the target quantile of each base value. This pairing is what makes the reported endpoint MSE a comparison of paths from the same initial sample. The program regenerates `experiment_results.tex` and `flow_distribution.pdf` from its results. Preserve the same initial values and distinguish a finer numerical integration of a fixed checkpoint from training a different velocity field when carrying out P7.
