# Reproducing Chapter 3

Run these standard-library Python programs from the `course/` directory. No model training is required.

| Program | Calculation or comparison |
|---|---|
| `python3 chapter03/worked_examples.py` | Complete-block arithmetic and a fixed two-layer, two-head model under suffix changes, separate prefix passes, and leaking controls. |
| `python3 chapter03/retrieval_examples.py` | Sharp attention bounds, repeated keys, and score perturbations. |
| `python3 chapter03/composition_examples.py` | Finite-score register composition and cleanup under bounded errors. |
| `python3 chapter03/token_routing_examples.py` | Discover pointers from tokens and compare composed answers with direct prefix searches. |

When reproducing the complete forward pass:

1. Compare scalar attention with the matrix calculation, including allowed row sums and the two-key example.
2. Hold parameters fixed and change only future input rows. Earlier logits must stay unchanged in the specified causal model.
3. Introduce a known leakage path and check that the comparison detects it.
4. Compare a full-sequence forward pass with separate prefix passes using identical retained position indices and parameters.
5. Check permutation equivariance in the unmasked, position-free setting. A fixed causal mask changes this comparison.
6. Follow a prediction through attention, both residual additions, both normalizations, the MLP, final normalization, and vocabulary softmax.

Hold any random choices fixed across comparisons. Disable dropout in a model that uses it. The prefix theorem establishes equalities for every input in its stated model. These numerical comparisons reveal implementation errors on the tested inputs.

The normalized complete block and the unnormalized attention-ReLU routing construction have different conventions. Use the parameters and state fields specified for the calculation being reproduced. The chapter's resource counts include the one-hot fields and finite score scales of its explicit construction.
