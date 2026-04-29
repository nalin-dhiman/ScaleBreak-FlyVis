# ScaleBreak-FlyVis

Clean public repository for the ScaleBreak-FlyVis analysis pipeline.

ScaleBreak-FlyVis tests whether dynamic visual variables remain decodable across
changes in **retinal apparent scale** in a pretrained connectome-constrained fly
visual model. The project deliberately does **not** claim physical-distance
perception, generic object recognition, or exact connectome causality.

## What Is Included

- Reproducible pipeline code:
  - `scalebreak_flyvis/scripts/`
  - `scalebreak_flyvis/src/`
  - `scalebreak_flyvis/configs/`
  - `scalebreak_flyvis/tests/`
- Lightweight final results:
  - publication figures
  - supplementary figures
  - result tables
  - reviewer-hardening diagnostics
- Baseline summaries:
  - TemporalResNet18Small
  - STN-CNN
  - hex-native temporal model
  - calibration and pixel robustness checks

## What Is Not Included

The repository intentionally excludes heavy/local artifacts:

- paper draft folder and compiled manuscript package
- `.venv_flyvis/`
- `flyvis_data/`
- raw FlyVis stimuli and response arrays
- activation/feature `.npy` arrays
- model checkpoints
- local neuPrint export tables

This keeps the repository small and reviewable. Scripts retain the expected
paths and can regenerate heavy outputs when the local FlyVis and neuPrint data
are available.

## Repository Layout

```text
scalebreak_flyvis/
  configs/                 # pipeline configs
  scripts/                 # audit, stimulus, model, analysis, plotting scripts
  src/scalebreak/          # reusable pipeline modules
  tests/                   # smoke tests
  outputs/
    final_submission/      # final figures/tables only, not manuscript folder
    final_hardening/       # final tables and publication figure exports
    serious_cnn_baseline/  # summary tables/training curves only
    stn_cnn_baseline/      # summary tables
    hex_native_temporal_baseline/
    calibration_reliability/
    pixel_robustness_control/
    flyvis_variability_control/
    final_reviewer_metrics/
```

## Main Result Snapshot

The final reviewer-hardened comparison is intentionally conservative:

| Model | LOSO direction accuracy |
|---|---:|
| Hex-native temporal model | 1.000 |
| FlyVis + response noise (5%) | 0.938 |
| FlyVis | 0.924 |
| TemporalResNet18Small | 0.608 |
| STN-CNN | 0.597 |
| Pixel | 0.597 |
| Local RNN | 0.385 |
| Nuisance | 0.167 |

The hex-native temporal model reaching ceiling is important: the raw direction
LOSO task is not FlyVis-unique. The defensible claim is that FlyVis encodes
dynamic direction in a scale-stable, biologically interpretable temporal and
cell-type representation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scalebreak_flyvis/requirements.txt
```

Optional FlyVis installation is required only for scripts that directly run the
pretrained FlyVis model.

## Smoke Tests

```bash
PYTHONPATH=scalebreak_flyvis/src pytest scalebreak_flyvis/tests -q
```

## Useful Commands

Generate publication figures from included lightweight tables:

```bash
python scalebreak_flyvis/scripts/30_make_publication_figures.py \
  --outputs-dir scalebreak_flyvis/outputs \
  --out-dir scalebreak_flyvis/outputs/final_hardening/figures_pub_clean
```

Run final reviewer metrics from existing outputs:

```bash
python scalebreak_flyvis/scripts/41_final_reviewer_metrics.py \
  --outputs-dir scalebreak_flyvis/outputs
```

Run the hex-native temporal baseline, if raw stimuli are available locally:

```bash
python scalebreak_flyvis/scripts/40_train_hex_native_temporal_baseline.py \
  --seeds 42 \
  --epochs 10 \
  --batch-size 64
```

## Data Requirements For Full Reproduction

Full reproduction requires local data that are not committed:

- FlyVis data/checkpoint cache
- Pilot v2 native stimuli, e.g. `outputs/flyvis_pilot_v2/stimuli/stimuli.npy`
- FlyVis response tensors, e.g. `outputs/flyvis_pilot_v2/responses/flyvis_central_cell_responses.npy`
- neuPrint optic-lobe export tables for graph-audit scripts

The included result tables and figures are enough to inspect the final reported
results without downloading multi-GB arrays.

## Scientific Guardrails

Use these phrases:

- retinal apparent scale
- retinal projection
- dynamic direction representation
- scale-generalization
- activity proxy
- connectome-constrained model

Avoid these claims:

- physical distance perception
- generic object recognition
- exact connectome necessity
- FlyVis uniqueness over all possible artificial models

