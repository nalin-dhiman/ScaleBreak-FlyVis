#!/usr/bin/env bash
set -euo pipefail

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_flyvis}"

PY="scalebreak_flyvis/.venv_flyvis/bin/python"

if [[ "$QUICK" == "1" ]]; then
  "$PY" scalebreak_flyvis/scripts/17_train_strong_vision_controls.py --seeds 42 --epochs 3 --quick
  "$PY" scalebreak_flyvis/scripts/18_temporal_lesion_analysis.py --quick
else
  "$PY" scalebreak_flyvis/scripts/17_train_strong_vision_controls.py --seeds 42,84,96,123,777 --epochs 25
  "$PY" scalebreak_flyvis/scripts/18_temporal_lesion_analysis.py
fi

"$PY" scalebreak_flyvis/scripts/19_finalize_claims_and_figures.py
"$PY" scalebreak_flyvis/scripts/20_write_manuscript_skeleton.py
