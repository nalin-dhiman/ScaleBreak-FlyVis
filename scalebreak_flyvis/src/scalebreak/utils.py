"""Shared utilities for reproducible ScaleBreak-FlyVis scripts."""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_out_dir(path: str | Path, overwrite: bool = False) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {out}. Pass --overwrite to replace/add outputs intentionally."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_subdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(out_dir: str | Path, name: str, level: str = "INFO") -> logging.Logger:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(Path(out_dir) / f"{name}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def git_commit(repo_root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def package_versions(packages: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "installed")
        except Exception:
            versions[package] = None
    return versions


def run_info(script: str, seed: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "script": script,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git_commit": git_commit("."),
        "seed": seed,
        "package_versions": package_versions(
            ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "networkx", "pyarrow", "torch", "zarr"]
        ),
    }
    if extra:
        info.update(extra)
    return info


def copy_config(config_path: str | Path | None, out_dir: str | Path, config: dict[str, Any] | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copy2(config_path, out / "config_used.yaml")
    elif config is not None:
        write_yaml(config, out / "config_used.yaml")


def parse_list_arg(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]
