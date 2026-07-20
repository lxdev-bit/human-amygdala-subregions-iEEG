# Analysis Functions and Scripts for iEEG Local activity, Connectivity, Decoding, and Statistical Analysis

This repository contains seven MATLAB functions and R scripts used in the connectivity, decoding, time–frequency, and statistical analyses reported in the paper:

**“Functional specialization and dynamical interaction in human amygdala subregions support fearful-expression recognition.”**

The analysis functions avoid hard-coded data paths where possible and operate primarily on in-memory arrays. Users should adapt data-loading procedures, variable names, and analysis parameters to their own datasets.

## Files at a glance

* **cal_dwpli.m** — Computes debiased weighted phase-lag index (dwPLI) connectivity between two channels across time and frequency.

* **cal_npCGC.m** — Computes nonparametric conditional Granger causality in the frequency domain from multichannel electrophysiological data.

* **cal_zpower.m** — Computes single-condition BOSC-based time–frequency power and applies z-score normalization relative to a prestimulus baseline.

* **lme_stat.R** — Provides example R code for the linear mixed-effects model analyses used in the study. The script illustrates factor specification, model fitting with `lmer`, Type III tests, estimated marginal means, and interaction contrasts. The same general workflow was applied across outcome indices, with the fixed-effect structure adapted to the statistical model specified for each index in the Methods. All post hoc comparisons following the linear mixed-effects models were corrected for multiple comparisons using the Bonferroni method.

* **permutest.m** — Implements a cluster-based permutation test for multidimensional electrophysiological data. The function identifies adjacent samples exceeding a cluster-forming threshold, calculates cluster-level statistics, and evaluates the observed clusters against a permutation-derived null distribution while controlling for multiple comparisons. The procedure is based on the cluster-based permutation framework described by Maris and Oostenveld (2007).

* **tf_stat.m** — Runs the time–frequency statistical analyses used in this study by preparing the condition-level data and calling `permutest.m`. The script performs cluster-based permutation comparisons of time–frequency power between conditions and returns the observed statistical maps, cluster assignments, and cluster-level permutation probabilities.

* **conditional_decode_time_windows.m** — Performs time-resolved SVM decoding between two task conditions using repeated 70% training and 30% testing splits at the trial level. PCA is fitted exclusively to the training data within each split and applied to the corresponding test data for dimensionality reduction.

* **subregional_decode_time_windows.m** — Performs time-resolved SVM decoding between lateral and medial amygdala contacts using leave-one-contact-out cross-validation. In each fold, one contact is retained for testing and all remaining contacts are used for training, with PCA fitted exclusively to the training data.


## Reference for the cluster-based permutation procedure

Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data. *Journal of Neuroscience Methods, 164*(1), 177–190. https://doi.org/10.1016/j.jneumeth.2007.03.024
