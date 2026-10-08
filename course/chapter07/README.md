# Chapter 7 computational checks

From the source-package root, run:

```sh
python3 course/chapter07/code/verify.py
```

Requirements: Python 3 and NumPy. Equivalently, from `course/`, run `python3 chapter07/code/verify.py`. The program prints JSON and does not train a model or fit empirical scaling data.

It checks streaming attention against direct attention and central finite differences, including a large common score offset. It also evaluates the declared FLOP/cache, lifetime-cost, candidate-utility and two-response-function examples. For the response-function comparison, the functions agree at the stated calibration point and separate at proposed additional configurations. The question is which run distinguishes these hypotheses, not whether invented data establish a real scaling law.

For Problem P2, complete the derivation and independent implementation before consulting this verification program, as requested in the problem. The program contains reference calculations.
