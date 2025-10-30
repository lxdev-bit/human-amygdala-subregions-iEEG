# MATLAB Functions for iEEG Connectivity & Decoding

This repo contains four MATLAB functions for iEEG connectivity & decoding workflows of the paper  
**"Functional specialization and dynamical interaction in human amygdala subregions support fearful-expression recognition"**.  

All functions avoid hard-coded data paths and operate on in-memory arrays.

---

## Files at a glance

- **cal_dwpli.m** — Debiased weighted PLI (dwPLI) between two channels over time–frequency.

- **cal_npCGC.m** — Nonparametric conditional Granger causality (frequency-domain) from multichannel data.

- **cal_zpower.m** — Single-condition BOSC time–frequency power with z-baseline correction.

- **decode_time_windows.m** — Sliding-window SVM decoding (cond1 vs. cond2) with PCA pipeline.
