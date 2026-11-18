# D-ripALM

MATLAB implementation of D-ripALM for consensus convex optimization over multi-agent networks. The method uses a double-loop framework with a relative error criterion to control inner accuracy through a single tolerance parameter.

## Benchmarks

The experiments use adapted code from [NIDS](https://github.com/mingyan08/NIDS) (including its PG-EXTRA implementation) and [IDEAL](https://github.com/hongzhoulin89/IDEAL-Inexact-DEcentralized-Accelerated-Augmented-Lagrangian-Method). We thank the authors for making their code available.

## Usage

In MATLAB, set the current folder to this repository. Run a `RunAll_*` script to execute the experiments and save results under `results/`, then run the matching `print_*` script to output LaTeX tables from those results. Experiment settings are defined near the top of each script; keep the settings consistent between each pair.

Table numbers refer to the revised manuscript.

| Experiment | Table(s) | Run experiments | Print tables |
| --- | --- | --- | --- |
| Logistic regression, synthetic data | 1 | [RunAll_vs_IDEAL_LR_random.m](RunAll_vs_IDEAL_LR_random.m) | [print_vs_IDEAL_LR_random.m](print_vs_IDEAL_LR_random.m) |
| Logistic regression, MNIST (0 vs. 8) | 2 | [RunAll_vs_IDEAL_LR_MNIST.m](RunAll_vs_IDEAL_LR_MNIST.m) | [print_vs_IDEAL_LR_MNIST.m](print_vs_IDEAL_LR_MNIST.m) |
| LASSO, synthetic data | 4 | [RunAll_vs_NIDS_LASSO_random.m](RunAll_vs_NIDS_LASSO_random.m) | [print_vs_NIDS_LASSO_random.m](print_vs_NIDS_LASSO_random.m) |
| LASSO, LIBSVM: abalone_scale, bodyfat_scale, space_ga_scale | 5, 6, 7, respectively | [RunAll_vs_NIDS_LASSO_LIBSVM.m](RunAll_vs_NIDS_LASSO_LIBSVM.m) | [print_vs_NIDS_LASSO_LIBSVM.m](print_vs_NIDS_LASSO_LIBSVM.m) |

## Citation

**Plain text**

> Jiayi Zhu, Hong Wang, Ling Liang, and Lei Yang (2026). *D-ripALM: A Tuning-friendly Decentralized Relative-Type Inexact Proximal Augmented Lagrangian Method*. [arXiv:2602.06398](https://arxiv.org/abs/2602.06398).

**BibTeX**

```bibtex
@misc{zhu2026dripalm,
  title         = {{D-ripALM}: A Tuning-friendly Decentralized Relative-Type Inexact Proximal Augmented Lagrangian Method},
  author        = {Jiayi Zhu and Hong Wang and Ling Liang and Lei Yang},
  year          = {2026},
  eprint        = {2602.06398},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  url           = {https://arxiv.org/abs/2602.06398}
}
```
