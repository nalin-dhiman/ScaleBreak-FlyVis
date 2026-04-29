# Submission Checklist

- Figures referenced correctly: yes
- Supplementary referenced: yes
- Claims consistent: yes
- No overclaiming: yes; exact connectome necessity remains not established.
- All data traceable: yes; final tables and figures cite existing output tables.
- No unresolved placeholders left: yes
- Reviewer numerical consistency pass: yes; LOSO scalar metric is distinguished from Fig. S3 pairwise matrices.
- Reviewer LOSO matrix pass: yes; Fig. S12 and Table S11 show model-by-held-out-scale LOSO accuracies.
- Main figure consistency pass: yes; main Fig. 2 now shows the LOSO protocol that produces the 0.924 scalar.
- Reviewer baseline fairness pass: yes; hex-grid projection and CNN validation-vs-heldout behavior are documented.
- Reviewer baseline-definition pass: yes; Table S12 defines pixel/nuisance/TemporalResNet features.
- Reviewer graph-control pass: yes; Table S13 and text explain identical graph-control scores as type-rate-control insensitivity.
- Decoder/calibration pass: yes; logistic-regression settings and uncalibrated information-proxy caveat are documented.
- STN-CNN baseline pass: yes; Table 1 and Fig. 3 include the scale-aware baseline.
- Hex-native baseline pass: yes; Table 1 and Fig. 3 include the raw hex-sequence temporal baseline, which reaches ceiling and narrows the comparative claim.
- Pixel robustness pass: yes; Table S15 reports the spatial-scramble control.
- Phase-scramble pass: yes; Table S20 reports the temporal phase-scramble control.
- ID-vs-OOD pass: yes; Table S18 separates in-scale validation from LOSO held-out-scale accuracy.
- FlyVis variability pass: partial; no additional pretrained checkpoints were available, but Table S16 reports a small response-noise stability diagnostic.

## Warnings
- Serious CNN has leave-one-scale-out heldout accuracies, not a full single-train-scale/test-scale matrix.
- The temporal 3D CNN result is a lightweight single-seed control, not a manuscript-grade multi-seed sweep.
- The STN-CNN is a bounded reviewer-facing baseline, not an exhaustive scale-aware architecture search.
- The hex-native temporal model solves the raw direction task, so the manuscript must not claim FlyVis uniqueness or superiority over all artificial baselines.
- Modern full scale-equivariant, log-polar, transformer, and state-space baselines remain future work.
