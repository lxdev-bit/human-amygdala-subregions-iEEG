# Analysis Functions and Scripts for iEEG Local activity, Connectivity, Decoding, and Statistical Analysis

This repository contains eight MATLAB functions and R scripts used in the connectivity, decoding, time–frequency, and statistical analyses reported in the paper:

**Dan Cao, Jiachen Cui, Xinyuan Yan, Xinru Zhang, Yuanyuan Dang, Hulin Zhao, Jin Li, Jianning Zhang, and Zhang, Yanyang. "Functional specialization and dynamical interaction in human amygdala subregions support fearful-expression processing." (2026).**

## Result-to-script mapping

| Figure panel(s)       | Analysis script(s)                  | Statistical script(s)          |
| --------------------- | ----------------------------------- | ------------------------------ |
| Fig. 3A               | `cal_zpower.m`                      | —                              |
| Fig. 3C–E             | `subregional_decode_time_windows.m` | `subregional_decode_shuffle.m` |
| Fig. 4A / Fig. 6A     | `tf_stat.m`                         | `permutest.m`                  |
| Fig. 4C / Fig. 6B     | `conditional_decode_time_windows.m` | `conditional_decode_shuffle.m` |
| Fig. 4D–E / Fig. 6C–D | `extra_point.m`                     | `lme_stat.R`                   |
| Fig. 5A               | `cal_dwpli.m`                       | —                              |
| Fig. 5B / Fig. 6E     | `tf_curve_stat.m`                   | `permutest.m`                  |
| Fig. 5C / Fig. 6F     | `extra_point.m`                     | `lme_stat.R`                   |
| Fig. 5D / Fig. 6G     | `cal_npCGC.m`, `extra_point.m`      | `lme_stat.R`                   |
| Fig. S3               | `tf_curve_stat.m`                   | `permutest.m`                  |
| Fig. S8A–C            | `conditional_LOSO_decode.m`         | `lme_stat.R`                   |
| Fig. S8D–F            | `subregional_LOSO_decode.m`                                          |


## Files at a glance

* **cal_dwpli.m** — Computes debiased weighted phase-lag index (dwPLI) connectivity between two channels across time and frequency.

* **cal_npCGC.m** — Computes nonparametric conditional Granger causality in the frequency domain from multichannel electrophysiological data.

* **cal_zpower.m** — Computes single-condition BOSC-based time–frequency power and applies z-score normalization relative to a prestimulus baseline.

* **extra_point.m** — Extract fixed-window values for subsequent linear mixed-effects models.

* **lme_stat.R** — R code for the linear mixed-effects model analyses with Bonferroni-corrected post hoc comparisons. The analytical procedures were identical across analyses, except for the fixed-effect variables of interest. This code for the analysis of z-scored power is provided below as an example.

* **permutest.m** — Implements a cluster-based permutation test for multidimensional electrophysiological data. The function identifies adjacent samples exceeding a cluster-forming threshold, calculates cluster-level statistics, and evaluates the observed clusters against a permutation-derived null distribution while controlling for multiple comparisons. The procedure is based on the cluster-based permutation framework described by Maris and Oostenveld (2007).

* **tf_stat.m** — Runs the time–frequency statistical analyses used in this study by preparing the condition-level data and calling `permutest.m`. The script performs cluster-based permutation comparisons of time–frequency power between conditions and returns the observed statistical maps, cluster assignments, and cluster-level permutation probabilities.

* **tf_curve_stat.m** — Extract frequency-averaged time courses and perform paired cluster-based permutation tests between conditions.

* **conditional_decode_time_windows.m** — Performs time-resolved SVM decoding between two task conditions using repeated 70% training and 30% testing splits at the trial level. PCA is fitted exclusively to the training data within each split and applied to the corresponding test data for dimensionality reduction.

* **conditional_decode_shuffle.m** — Label-shuffling permutation analysis for sliding-window conditional decoding using repeated train/test splits.

* **conditional_LOSO_decode.m** — Subject-level leave-one-subject-out decoding with label-shuffling permutation analysis.

* **subregional_decode_time_windows.m** — Performs time-resolved SVM decoding between lateral and medial amygdala contacts using leave-one-contact-out cross-validation. In each fold, one contact is retained for testing and all remaining contacts are used for training, with PCA fitted exclusively to the training data.

* **subregional_decode_shuffle.m** —  Label-shuffling permutation analysis for time-resolved subregional decoding using leave-one-contact-out cross-validation.

* **subregional_LOSO_decode.m** —  Fixed-window lateral-versus-medial amygdala decoding using leave-one-subject-out cross-validation and within-subject label permutation.
