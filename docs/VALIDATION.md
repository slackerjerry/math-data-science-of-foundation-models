# Packaging validation — 2026-10-08

The programs and retained experimental assets in `course/` were copied unchanged from complete source v24, paired with lecture notes v50. `UPSTREAM_FILES.json` lists their hashes. This release adds repository documentation, environment files and an isolated check runner.

## Publication snapshot

The initial code-only repository retains every Python program, runner, dependency file, saved result, model and figure from the validated computational companion. Eight generated `experiment_results.tex` output tables are omitted; their unchanged programs regenerate them locally. `UPSTREAM_FILES.json` separates retained-file hashes from omitted generated-output hashes. All retained computational files were checked against the validated package and all 36 Python files parsed during staging.

The execution results below are from the original packaging run. The 24 numerical and three PyTorch checks were not rerun while staging this publication snapshot.

## Previously executed packaging checks

- All 36 included Python files (35 course scripts and the new runner) parsed successfully.
- The numerical suite executed 24 scripts successfully, covering exact examples, numerical checks, simulations and the two Chapter 5 geometry figures.
- The PyTorch suite executed three checks successfully: Chapter 5 derivative and causal-prefix checks, Chapter 8 saved seed-11 evaluation, and Chapter 12 saved-field evaluation.
- The Chapter 5 derivative relative discrepancy was about 1.68e-9. Its causal-prefix maximum discrepancy was about 4.44e-16.

Environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and PyTorch 2.14.1+cpu, Linux CPU. The same suites are rerun from the extracted delivery ZIP. Machine-readable results accompany the full course release under `release_checks/`.

## Limits

No network was pretrained during packaging. The historical six-run Chapter 5 pilot, Chapter 8 training, Chapter 9 adaptation and Chapter 12 training were not rerun. The full 600-corpus linear-pretext simulation was not rerun. Saved records are retained, not presented as newly reproduced training. Successful execution is not a new mathematical audit or student-learning validation. GPU, Windows and macOS execution were not tested.
