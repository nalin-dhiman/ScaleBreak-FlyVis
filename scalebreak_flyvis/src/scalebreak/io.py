"""Input/output helpers for neuPrint tables and array stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from .utils import write_json


TABLE_SUFFIXES = {".feather", ".csv", ".json"}


def discover_tables(data_dir: str | Path) -> list[Path]:
    base = Path(data_dir)
    return sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES])


def feather_schema_and_rows(path: str | Path) -> tuple[list[str], dict[str, str], int]:
    p = Path(path)
    with ipc.open_file(p) as f:
        cols = f.schema.names
        dtypes = {field.name: str(field.type) for field in f.schema}
        rows = 0
        for i in range(f.num_record_batches):
            rows += f.get_batch(i).num_rows
    return cols, dtypes, rows


def read_table_sample(path: str | Path, n: int = 5) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, nrows=n)
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj[:n])
        if isinstance(obj, dict):
            items = list(obj.items())[:n]
            return pd.DataFrame({"key": [k for k, _ in items], "value": [v for _, v in items]})
        return pd.DataFrame({"value": [obj]})
    with ipc.open_file(p) as f:
        if f.num_record_batches == 0:
            return pd.DataFrame()
        return f.get_batch(0).slice(0, n).to_pandas()


def table_basic_info(path: str | Path, sample_rows: int = 5) -> dict[str, Any]:
    p = Path(path)
    info: dict[str, Any] = {"name": p.name, "path": str(p), "suffix": p.suffix.lower()}
    if p.suffix.lower() == ".feather":
        cols, dtypes, rows = feather_schema_and_rows(p)
        sample = read_table_sample(p, sample_rows)
        info.update({"n_rows": rows, "n_cols": len(cols), "columns": cols, "dtypes": dtypes})
    elif p.suffix.lower() == ".csv":
        sample = pd.read_csv(p, nrows=sample_rows)
        try:
            rows = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1
        except Exception:
            rows = None
        info.update(
            {
                "n_rows": rows,
                "n_cols": len(sample.columns),
                "columns": list(sample.columns),
                "dtypes": {c: str(t) for c, t in sample.dtypes.items()},
            }
        )
    else:
        sample = read_table_sample(p, sample_rows)
        info.update(
            {
                "n_rows": len(sample),
                "n_cols": len(sample.columns),
                "columns": list(sample.columns),
                "dtypes": {c: str(t) for c, t in sample.dtypes.items()},
            }
        )
    info["missingness_sample"] = sample.isna().mean().to_dict() if not sample.empty else {}
    return info


def infer_column_candidates(columns: list[str]) -> dict[str, list[str]]:
    lower = {c: c.lower() for c in columns}

    def has_any(col: str, parts: tuple[str, ...]) -> bool:
        return any(part in lower[col] for part in parts)

    coordinate_names = []
    for c in columns:
        lc = lower[c]
        if "point" in lc or "coord" in lc or "location" in lc or lc in {"x", "y", "z"} or lc.endswith(":x") or lc.endswith(":y") or lc.endswith(":z"):
            coordinate_names.append(c)
    return {
        "body_id": [c for c in columns if "body" in lower[c] or lower[c] in {"id", ":id"}],
        "pre": [c for c in columns if has_any(c, ("start_id", "pre", "source", "from"))],
        "post": [c for c in columns if has_any(c, ("end_id", "post", "target", "to"))],
        "cell_type": [c for c in columns if "type" in lower[c] and "location" not in lower[c]],
        "roi": [c for c in columns if "roi" in lower[c]],
        "weight": [c for c in columns if has_any(c, ("weight", "syn", "count"))],
        "coordinate": coordinate_names,
    }


def choose_first(columns: list[str], preferred: list[str], candidates: list[str]) -> str | None:
    for pref in preferred:
        if pref in columns:
            return pref
    return candidates[0] if candidates else None


def save_array_store(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> None:
    """Save arrays as zarr when available, otherwise as a NumPy-backed directory."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    try:
        import zarr  # type: ignore

        root = zarr.open_group(str(p), mode="w")
        for key, value in arrays.items():
            root.create_dataset(key, data=value, chunks=True, overwrite=True)
        if metadata:
            root.attrs.update(metadata)
        backend = "zarr"
    except Exception as exc:
        for key, value in arrays.items():
            np.save(p / f"{key}.npy", value)
        backend = f"numpy_directory_fallback:{type(exc).__name__}"
    write_json({"backend": backend, "arrays": {k: list(v.shape) for k, v in arrays.items()}, "metadata": metadata or {}}, p / "manifest.json")


def load_array_store(path: str | Path, key: str) -> np.ndarray:
    p = Path(path)
    npy = p / f"{key}.npy"
    if npy.exists():
        return np.load(npy, allow_pickle=False)
    try:
        import zarr  # type: ignore

        return np.asarray(zarr.open_group(str(p), mode="r")[key])
    except Exception as exc:
        raise FileNotFoundError(f"Cannot load {key} from {p}") from exc


def read_edges_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".feather":
        return feather.read_table(p, columns=columns).to_pandas()
    return pd.read_csv(p, usecols=columns)
