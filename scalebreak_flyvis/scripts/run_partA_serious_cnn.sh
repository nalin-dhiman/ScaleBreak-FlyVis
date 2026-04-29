#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_flyvis}"

PY="scalebreak_flyvis/.venv_flyvis/bin/python"

if [[ "${1:-}" == "--quick" ]]; then
  "$PY" scalebreak_flyvis/scripts/21_train_serious_cnn_baseline.py --seeds 42 --epochs 5 --quick
else
  "$PY" scalebreak_flyvis/scripts/21_train_serious_cnn_baseline.py --seeds 42,84,96,123,777 --epochs 50 --batch-size 64
fi

"$PY" scalebreak_flyvis/scripts/25_update_manuscript_after_hardening.py
