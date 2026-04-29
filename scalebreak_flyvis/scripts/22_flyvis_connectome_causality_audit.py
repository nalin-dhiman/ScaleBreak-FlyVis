#!/usr/bin/env python
"""Audit FlyVis internals for connectome-causality perturbation feasibility."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def module_tree(obj: Any, max_depth: int = 4) -> str:
    lines: list[str] = []

    def rec(x: Any, name: str, depth: int) -> None:
        lines.append("  " * depth + f"{name}: {type(x).__module__}.{type(x).__name__}")
        if depth >= max_depth:
            return
        try:
            children = list(x.named_children())
        except Exception:
            children = []
        for child_name, child in children:
            rec(child, child_name, depth + 1)

    rec(obj, "network", 0)
    return "\n".join(lines)


def summarize_candidate(name: str, value: Any) -> dict[str, Any]:
    out = {"name": name, "type": f"{type(value).__module__}.{type(value).__name__}"}
    for attr in ["shape", "dtype", "device", "requires_grad"]:
        try:
            out[attr] = getattr(value, attr)
        except Exception:
            pass
    try:
        arr = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
        if arr.size and arr.ndim >= 1:
            out.update({"array_shape": arr.shape, "finite": bool(np.isfinite(arr).all()), "nonzero": int(np.count_nonzero(arr))})
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/connectome_causality")
    parser.add_argument("--flyvis-root", default="scalebreak_flyvis/flyvis_data")
    parser.add_argument("--model", default="flow/0000/000")
    args = parser.parse_args()

    root = Path.cwd()
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FLYVIS_ROOT_DIR", str((root / args.flyvis_root).resolve()))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_flyvis")

    feasibility_rows = []
    inventory: dict[str, Any] = {"status": "started", "model": args.model}
    try:
        import torch
        import flyvis
        from flyvis import results_dir
        from flyvis.network import NetworkView

        nv = NetworkView(results_dir / args.model)
        network = nv.init_network()
        (out / "model_tree.txt").write_text(module_tree(network), encoding="utf-8")
        params = []
        for name, p in network.named_parameters():
            arr = p.detach().cpu().numpy()
            params.append(
                {
                    "name": name,
                    "shape": tuple(arr.shape),
                    "dtype": str(arr.dtype),
                    "requires_grad": bool(p.requires_grad),
                    "n_parameters": int(arr.size),
                    "n_nonzero": int(np.count_nonzero(arr)),
                    "mean": float(arr.mean()) if arr.size else np.nan,
                    "std": float(arr.std()) if arr.size else np.nan,
                    "min": float(arr.min()) if arr.size else np.nan,
                    "max": float(arr.max()) if arr.size else np.nan,
                }
            )
        pd.DataFrame(params).to_csv(out / "parameter_inventory.csv", index=False)

        conn = getattr(nv, "connectome", None) or getattr(network, "connectome", None)
        conn_items: dict[str, Any] = {}
        if conn is not None:
            for name in dir(conn):
                if name.startswith("_"):
                    continue
                try:
                    value = getattr(conn, name)
                except Exception:
                    continue
                if any(token in name.lower() for token in ["edge", "weight", "connect", "node", "type", "sign", "index", "mask"]):
                    conn_items[name] = summarize_candidate(name, value)
        write_json(conn_items, out / "connectivity_inventory.json")

        try:
            all_cell_types = nv.connectome.nodes.type[:].astype(str)
            all_u = nv.connectome.nodes.u[:]
            all_v = nv.connectome.nodes.v[:]
            central = nv.connectome.central_cells_index[:]
            cell_df = pd.DataFrame(
                {
                    "node_index": np.arange(len(all_cell_types)),
                    "cell_type": all_cell_types,
                    "u": all_u,
                    "v": all_v,
                    "is_central": np.isin(np.arange(len(all_cell_types)), central),
                }
            )
        except Exception:
            cell_df = pd.DataFrame()
        cell_df.to_csv(out / "cell_type_inventory.csv", index=False)

        names = [p["name"].lower() for p in params]
        likely_weight = [p for p in params if any(k in p["name"].lower() for k in ["weight", "edge", "syn", "connect"])]
        likely_recurrent = [p for p in params if any(k in p["name"].lower() for k in ["rec", "recur", "time"])]
        feasibility_rows.extend(
            [
                {"item": "model_load", "feasible": True, "note": "NetworkView.init_network succeeded."},
                {"item": "parameter_inventory", "feasible": len(params) > 0, "note": f"{len(params)} parameter tensors discovered."},
                {"item": "candidate_weight_tensors", "feasible": len(likely_weight) > 0, "note": f"{len(likely_weight)} weight-like tensors by name."},
                {"item": "candidate_recurrent_tensors", "feasible": len(likely_recurrent) > 0, "note": f"{len(likely_recurrent)} recurrent/time-like tensors by name."},
                {"item": "safe_in_place_weight_editing", "feasible": False, "note": "Not marked safe automatically; requires manual validation of FlyVis parameter semantics and constraints."},
            ]
        )
        inventory.update(
            {
                "status": "completed",
                "flyvis_version": getattr(flyvis, "__version__", "unknown"),
                "torch_version": torch.__version__,
                "n_parameters": len(params),
                "n_cell_inventory_rows": len(cell_df),
                "candidate_weight_tensors": [p["name"] for p in likely_weight],
                "candidate_recurrent_tensors": [p["name"] for p in likely_recurrent],
            }
        )
    except Exception as exc:
        inventory.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (out / "model_tree.txt").write_text("FlyVis audit failed before model tree extraction.\n", encoding="utf-8")
        pd.DataFrame().to_csv(out / "parameter_inventory.csv", index=False)
        pd.DataFrame().to_csv(out / "cell_type_inventory.csv", index=False)
        write_json({}, out / "connectivity_inventory.json")
        feasibility_rows.append({"item": "model_load", "feasible": False, "note": repr(exc)})

    feasibility = pd.DataFrame(feasibility_rows)
    feasibility.to_csv(out / "table_causal_feasibility.csv", index=False)
    write_json(inventory, out / "audit_run_info.json")
    verdict = "Direct in-place FlyVis connectivity editing was not certified safe by the automatic audit."
    if len(feasibility) and bool(feasibility.loc[feasibility["item"] == "safe_in_place_weight_editing", "feasible"].any()):
        verdict = "Direct in-place FlyVis connectivity editing appears feasible, but still requires validation."
    md = [
        "# FlyVis Connectome-Causality Feasibility",
        "",
        verdict,
        "",
        "This audit intentionally does not guess FlyVis internals. Parameter and connectivity candidates are saved for inspection.",
        "",
        "Automatic causal-variant scripts should label response-space perturbations as proxy perturbations unless direct weight editing is manually validated.",
    ]
    (out / "flyvis_causality_feasibility.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote FlyVis causality audit to {out}")


if __name__ == "__main__":
    main()
