# Reviewer Hardening Summary

This revision does not change experimental results. It strengthens reporting,
protocol clarity, and claim boundaries in response to reviewer concerns.

## Metric and Matrix Reconciliation

- Main Fig. 2 now shows the exact leave-one-scale-out (LOSO) protocol used for
  the headline scalar metric.
- Supplementary Fig. S3 remains the stricter single-train-scale/single-test-scale
  diagnostic.
- Supplementary Fig. S12 and Table S11 report model-by-held-out-scale LOSO
  accuracies, making the aggregation to FlyVis `A_offdiag = 0.924` auditable.

## Decoder and Information Metrics

- The main Methods now specify the decoder as `StandardScaler` plus
  scikit-learn `LogisticRegression` with L2 penalty, `C=1`,
  `class_weight=balanced`, `lbfgs`, and `max_iter=1000`.
- No temperature scaling, Platt scaling, or post-hoc calibration is used.
- The information quantity is now described as an uncalibrated decoder-based
  proxy, not exact mutual information or a formal calibrated lower bound.

## Baseline Fairness

- Pixel, nuisance, and TemporalResNet feature sets are explicitly defined in
  Supplementary Table S12.
- The main text states that CNN controls rely on a deterministic hex-to-grid
  projection and that future work should include hex-native controls.
- A scale-aware STN-CNN baseline was added under the same LOSO protocol. It
  reaches `A_offdiag = 0.597` with 95% CI `[0.580, 0.616]`, remaining below
  FlyVis by 0.326.
- A hex-native temporal model was added to remove the square-grid projection
  disadvantage. It consumes raw `(T, 721)` sequences with six-neighbor message
  passing and temporal convolutions, and reaches `A_offdiag = 1.000` in a
  one-seed bounded run. This narrows the paper's comparative claim: the raw
  direction task is not FlyVis-unique.
- A pixel spatial-scramble diagnostic was added. Original pixel features reach
  0.604 in the diagnostic rerun, while spatially scrambled features drop to
  0.166, indicating that organized retinal motion structure is necessary.
- A phase-scramble pixel diagnostic was added. It reaches 0.569, so temporal
  phase scrambling alone does not eliminate pixel-summary direction information.
- The limitations explicitly acknowledge missing scale-equivariant CNNs,
  log-polar pipelines, modern video transformers, and state-space models.

## Calibration and Variability

- Expected calibration error and Brier score are now reported for the FlyVis
  linear probe and STN-CNN baseline in Supplementary Table S14.
- The paper explicitly treats bits/activity as a qualitative,
  calibration-sensitive proxy.
- Additional pretrained FlyVis checkpoints were unavailable locally. A small
  response-noise diagnostic was added instead; 1% and 5% feature-std noise
  perturbations retain high LOSO accuracy (0.949 and 0.938 mean accuracy across
  three seeds), so the linear readout is not brittle to small response noise.

## Graph Controls

- Supplementary Table S13 audits the graph/type-rate controls.
- Identical graph-control aggregate scores are framed as insensitivity of a
  simplified type-rate baseline, not as evidence about exact FlyVis connectome
  causality.

## Cell-Group Interpretation

- Main text now separates sufficiency-style restricted readouts from zeroing
  analyses.
- T4-only, T5-only, T4+T5, and non-T4/T5 results are reported.
- Zeroing analyses are described as representational perturbations, not causal
  neural lesions.

## Final Claim Boundary

The paper now supports:

- robust dynamic direction scale-generalization in pretrained FlyVis;
- FlyVis model-specific temporal/cell-type representation relative to projected-grid and simplified controls;
- mid/late and distributed decodability across cell groups.

The paper does not claim:

- physical distance modeling;
- generic object recognition;
- exact connectome necessity;
- superiority over all possible artificial scale-robust models.
- uniqueness of the raw direction LOSO task.
