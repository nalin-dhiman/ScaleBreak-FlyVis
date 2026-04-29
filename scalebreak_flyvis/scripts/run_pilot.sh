#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/home/ub/codes/My_published_projects/Flow-concentration-and-structural-fragility-in-spatially-embedded-directed-networks/optic-lobe-v1.1-neuprint-tables"
OUT_DIR="scalebreak_flyvis/outputs"

python scalebreak_flyvis/scripts/00_audit_connectome_tables.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/audits"
python scalebreak_flyvis/scripts/01_build_connectome_graph.py --data-dir "$DATA_DIR" --audit-json "$OUT_DIR/audits/schema_report.json" --out-dir "$OUT_DIR/connectome"
python scalebreak_flyvis/scripts/02_generate_stimuli.py --config scalebreak_flyvis/configs/stimulus_grid_pilot.yaml
python scalebreak_flyvis/scripts/03_run_model_responses.py --config scalebreak_flyvis/configs/analysis.yaml --models pixel,local_rnn,optic_lobe_type_rate
python scalebreak_flyvis/scripts/04_extract_features.py --config scalebreak_flyvis/configs/analysis.yaml
python scalebreak_flyvis/scripts/05_train_linear_probes.py --config scalebreak_flyvis/configs/analysis.yaml
python scalebreak_flyvis/scripts/06_compute_rsa_cka.py --config scalebreak_flyvis/configs/analysis.yaml
python scalebreak_flyvis/scripts/07_estimate_breakpoints.py --config scalebreak_flyvis/configs/analysis.yaml
python scalebreak_flyvis/scripts/08_run_controls.py --config scalebreak_flyvis/configs/analysis.yaml
python scalebreak_flyvis/scripts/09_make_figures.py --config scalebreak_flyvis/configs/analysis.yaml
