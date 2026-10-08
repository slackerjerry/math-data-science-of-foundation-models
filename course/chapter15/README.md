# Chapter 15 exact calculation

Run from the source-package root:

```bash
python course/chapter15/code/pipeline_experiment.py
```

This uses only the Python standard library. It enumerates the binary observations using exact rational arithmetic, checks posterior error and adaptive lookup, and computes the two separately specified lifetime-cost comparisons. It writes `course/chapter15/pipeline_results.json` and regenerates `course/chapter15/experiment_results.tex`.

The optional-lookup comparison prices a call at 0.4. The context/adaptation lifetime comparison forces lookup at cost 0.03 and includes the specified processing costs. They are different decisions and should not be combined into one ranking. The program has no trained neural-network checkpoint.
