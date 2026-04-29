#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_flyvis}"
export FLYVIS_ROOT_DIR="${FLYVIS_ROOT_DIR:-/home/ub/Downloads/ScaleBreak-FlyVis/scalebreak_flyvis/flyvis_data}"

PY="scalebreak_flyvis/.venv_flyvis/bin/python"

"$PY" scalebreak_flyvis/scripts/22_flyvis_connectome_causality_audit.py

if [[ "${1:-}" == "--quick" ]]; then
  "$PY" scalebreak_flyvis/scripts/23_run_flyvis_causal_variants.py \
    --variants full,edge_dropout,t4t5_attenuation,non_t4t5_attenuation \
    --dropout-levels 0.10 \
    --attenuation-levels 0.5 \
    --quick
else
  "$PY" scalebreak_flyvis/scripts/23_run_flyvis_causal_variants.py \
    --variants full,weight_shuffle,edge_dropout,t4t5_attenuation,non_t4t5_attenuation \
    --dropout-levels 0.05,0.10,0.20 \
    --attenuation-levels 0.0,0.25,0.5,0.75
fi

"$PY" scalebreak_flyvis/scripts/24_analyze_connectome_causality.py
"$PY" scalebreak_flyvis/scripts/25_update_manuscript_after_hardening.py
