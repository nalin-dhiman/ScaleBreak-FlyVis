# ScaleBreak-FlyVis Final Submission Package

## Included Files
- `manuscript/`: manuscript skeleton, final abstract, methods draft, limitations, and figure plan.
- `figures/`: main publication figures as PDF/SVG plus PNG previews.
- `tables/`: final CSV and Markdown tables with rounded values and CI strings.
- `supplementary/`: supplementary material TeX/Markdown, figures, tables, and manifest.
- `main_reviewer_revised.tex` / `main_reviewer_revised.pdf`: compiled reviewer-revision main manuscript.
- `supplementary_reviewer_revised.tex` / `supplementary_reviewer_revised.pdf`: compiled reviewer-revision supplement.

## Figure Mapping
- Fig. 1: concept and stimulus examples.
- Fig. 2: LOSO held-out-scale direction decoding, matching the headline scalar metric.
- Fig. 3: FlyVis versus controls.
- Fig. 4: feature-family breakdown.
- Fig. 5: temporal lesion.
- Fig. 6: cell-group ablation.
- Fig. 7: efficiency / retained bits per activity proxy.

## Supplementary Mapping
- Fig. S1-S12 and Table S1-S20 are under `supplementary/`.
- Fig. S3 now uses available scale matrices and serious CNN LOSO split summaries; no fabricated matrix is included.
- `scale_matrix_summary.csv` explicitly distinguishes the paper's LOSO scalar metric from single-scale-pair matrix cells.
- Dense technical diagnostics, including full RSA/CKA grids, remain in the supplementary material.
- Fig. S12 and Table S11 provide the LOSO model-by-held-out-scale diagnostic requested to reconcile the 0.924 scalar with pairwise matrices.
- Table S12 defines pixel, nuisance, and trained CNN feature sets.
- Table S13 audits the graph controls and explains why identical aggregate scores are treated as a limitation, not causal evidence.
- Table S14 reports calibration reliability diagnostics for FlyVis and STN-CNN.
- Table S15 reports the pixel spatial-scramble robustness control.
- Table S16 reports the small FlyVis response-noise variability diagnostic.
- Table S17 reports the hex-native temporal baseline.
- Table S18 reports ID versus LOSO/OOD accuracy.
- Table S19 reports temperature-scaling calibration diagnostics.
- Table S20 reports the temporal phase-scramble pixel control.

## Claim Boundary
Primary claim: A pretrained connectome-constrained fly visual model preserves dynamic direction information across held-out retinal apparent scales.

Comparative claim: FlyVis encodes direction in a scale-stable subspace that transfers across retinal apparent scales and exceeds pixel, projected-grid trained CNN, STN-CNN, local RNN, graph, and destructive controls under identical protocols, but a lightweight hex-native temporal model reaches ceiling accuracy. The raw direction task is therefore not FlyVis-unique.

Mechanistic claim: The representation emerges in mid/late activity and is distributed across cell types.

Limitations: This work does not model physical distance, does not establish generic object recognition, and does not prove exact connectome necessity.

## Reviewer-Hardening Notes
- Main Fig. 2 now shows the LOSO protocol directly; pairwise single-scale transfer is kept in Supplementary Fig. S3.
- Decoder details, calibration caveats, pixel/nuisance features, graph-control limitations, and cell-group sufficiency/zeroing distinctions are documented in the main text and supplement.
- Missing stronger baselines are acknowledged explicitly as future work: full scale-equivariant models, log-polar pipelines, video transformers, and state-space models.
