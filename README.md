# D-ripALM

MATLAB implementation of D-ripALM for consensus convex optimization over multi-agent networks. The method uses a double-loop framework with a relative error criterion to control inner accuracy through a single tolerance parameter.

## Benchmarks

The experiments use code from [NIDS](https://github.com/mingyan08/NIDS), including its PG-EXTRA implementation, and [IDEAL](https://github.com/hongzhoulin89/IDEAL-Inexact-DEcentralized-Accelerated-Augmented-Lagrangian-Method). Before running the comparisons， please:

1. Download the code from the two original repositories. Place the NIDS repository contents under `solver/NIDS-master/`, and the contents of IDEAL's `Neurips_code/` directory directly under `solver/IDEAL-master/`.
2. Replace the following upstream files with the adapted versions supplied in this repository. Preserve our versions when copying the upstream code, and keep the supplied `solver/IDEAL-master/linear_prediction.m` helper in place.

| Upstream file | Adapted version and destination |
| --- | --- |
| NIDS: `fcns/NIDS.m` | [solver/NIDS-master/fcns/NIDS.m](solver/NIDS-master/fcns/NIDS.m) |
| IDEAL: `Neurips_code/AccGDSolver.m` | [solver/IDEAL-master/AccGDSolver.m](solver/IDEAL-master/AccGDSolver.m) |

These adaptations support the test problems, stopping criteria, and result reporting used in our experiments. 

We sincerely thank both author teams for making their source code publicly available. Their implementations made these comparisons possible.

## Usage

In MATLAB, set the current folder to this repository. Experiment settings and algorithm switches are defined near the top of each script.

### Test D-ripALM

Use a `RunMe_*` script to test a single configured case:

| Problem and data | Script |
| --- | --- |
| Logistic regression on synthetic data | [RunMe_vs_IDEAL_LR_random.m](RunMe_vs_IDEAL_LR_random.m) |
| Logistic regression on MNIST | [RunMe_vs_IDEAL_LR_MNIST.m](RunMe_vs_IDEAL_LR_MNIST.m) |
| LASSO on synthetic data | [RunMe_vs_NIDS_LASSO_random.m](RunMe_vs_NIDS_LASSO_random.m) |
| LASSO on LIBSVM data | [RunMe_vs_NIDS_LASSO_LIBSVM.m](RunMe_vs_NIDS_LASSO_LIBSVM.m) |

For example, run the following in the MATLAB Command Window:

```matlab
RunMe_vs_IDEAL_LR_random
```

The scripts enable the heuristic restarted variant, D-ripALM-re, and the corresponding baselines by default. The `RunMe_*` settings are for individual tests; use the batch scripts below for the paper experiments.

### Reproduce the paper experiments

Run a `RunAll_*` script to execute the experiment grid and save results under `results/`, then run the matching `print_*` script to output LaTeX tables. Keep the seeds, datasets, parameters, and result directories consistent between each pair.

| Experiment | Table(s) | Run experiments | Print tables |
| --- | --- | --- | --- |
| Logistic regression on synthetic data | 1 | [RunAll_vs_IDEAL_LR_random.m](RunAll_vs_IDEAL_LR_random.m) | [print_vs_IDEAL_LR_random.m](print_vs_IDEAL_LR_random.m) |
| Logistic regression on MNIST (0 vs. 8) | 2 | [RunAll_vs_IDEAL_LR_MNIST.m](RunAll_vs_IDEAL_LR_MNIST.m) | [print_vs_IDEAL_LR_MNIST.m](print_vs_IDEAL_LR_MNIST.m) |
| LASSO on synthetic data | 3 | [RunAll_vs_NIDS_LASSO_random.m](RunAll_vs_NIDS_LASSO_random.m) | [print_vs_NIDS_LASSO_random.m](print_vs_NIDS_LASSO_random.m) |
| LASSO on abalone_scale and space_ga_scale | 4, 5 | [RunAll_vs_NIDS_LASSO_LIBSVM.m](RunAll_vs_NIDS_LASSO_LIBSVM.m) | [print_vs_NIDS_LASSO_LIBSVM.m](print_vs_NIDS_LASSO_LIBSVM.m) |

## Citation

If you find this code useful in your research, we would appreciate a citation to our paper.

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

## License

The D-ripALM code is distributed under the [MIT License](LICENSE). For the downloaded benchmark code, retain the original authors' notices and follow the terms in their repositories.
