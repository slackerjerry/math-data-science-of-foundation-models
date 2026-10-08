# Chapter-to-code map

Paths below are relative to `course/`, except where that prefix is explicitly shown. Every listed program is included. The course documents supply the mathematical setup and interpretation.

| Chapter | Topic | Programs | Use |
|---|---|---|---|
| 1 | Data in the Foundation-Model Pipeline | `course/verify_examples.py` | Finite source mixtures and weighting. `check_results.json` is retained. |
| 2 | What Conditional Training Identifies | `chapter02/worked_examples.py`, `generation_examples.py` | Conditional objectives and generation comparisons. |
| 3 | Transformer Computation | `chapter03/worked_examples.py`, `retrieval_examples.py`, `composition_examples.py`, `token_routing_examples.py` | Attention retrieval, composition and routing. See the chapter README. |
| 4 | Self-Supervised Representation Learning and Transfer | `chapter04/worked_examples.py`, `research_examples.py`, `mean_distribution_example.py`, `finite_candidates_examples.py` | Representation and finite-candidate calculations. |
| 5 | Geometry and Learning Dynamics | `chapter05/worked_examples.py`, `dynamics_examples.py`, `feature_examples.py`, `geometry_examples.py`, `geometry_figures.py`; `shared_model/` | Exact fits/updates, geometry figures, and the causal-Transformer pilot. Checkpoints, task splits and raw predictions are included. |
| 6 | Generalization, Data Reuse, and Memorization | `chapter06/worked_examples.py`, `interpolation_examples.py`; `shared_model/linear_pretext.py` | Finite-data risk, reuse, verification and reduced-rank learning. |
| 7 | Compute, Scaling, and Inference Budgets | `chapter07/code/verify.py` | Declared cost and attention calculations, not measured large-model scaling. |
| 8 | In-Context Learning as Task-Conditional Inference | `chapter08/code/evaluate_checkpoint.py`, `icl_experiment.py`, `summarize.py` | Saved seeds 11/22/33, raw prompt errors and interventions. Evaluation and training are separate. |
| 9 | Parameter Adaptation | `chapter09/code/adaptation_experiment.py`, `summarize.py` | Loads Chapter 8 seed 22. Runs adaptation on twelve tasks, not base-model pretraining. |
| 10 | Preferences, Rewards, and Policy Learning | `chapter10/code/verify.py` | Finite-policy expected gradients and group estimators. Writes JSON and a TeX table. |
| 11 | Retrieval, Verification, and Tool Decisions | `chapter11/code/verify.py` | Exact Bayesian stopping plus seeded simulation and verification calculations. |
| 12 | Continuous Generation: Denoising, Diffusion, and Flow | `chapter12/code/flow_experiment.py` | Saved fields 12/23/34. Use `--evaluate-only` to avoid retraining. |
| 13 | Scientific Inference and Experimental Design | `chapter13/code/measurement_experiment.py` | Gaussian measurements, target risk and interval coverage. |
| 14 | Evaluation and Model Selection | `chapter14/code/evaluation_experiment.py` | Clustered observations, selection and Gaussian tuning. |
| 15 | Synthesis and Research Problems | `chapter15/code/pipeline_experiment.py` | Exact rational enumeration, adaptive lookup and separate lifetime-cost comparisons. |

Chapters without a dedicated README are covered here and by the program docstrings. The eight generated `experiment_results.tex` tables are omitted from the initial repository snapshot. They are numerical outputs, not manuscript sources or the solution manual; the listed programs regenerate them locally.
