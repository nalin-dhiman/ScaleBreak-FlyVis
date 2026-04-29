"""Connectome audit and graph construction utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from .io import choose_first, infer_column_candidates


NEURON_PREFS = {
    "body": ["bodyId:long", ":ID(Body-ID)", "body", "bodyId"],
    "type": ["type:string", "flywireType:string", "type"],
    "roi": ["roiInfo:string", "roi", "roiset_hash"],
    "coord": ["somaLocation:point{srid:9157}", "somaLocation"],
}
EDGE_PREFS = {
    "pre": [":START_ID(Body-ID)", "body_pre", "pre", "source"],
    "post": [":END_ID(Body-ID)", "body_post", "post", "target"],
    "weight": ["weight:int", "weightHR:int", "synweight", "weight"],
    "roi": ["roiInfo:string", "roi"],
}


def recommend_mapping(schema_report: dict[str, Any]) -> dict[str, Any]:
    tables = schema_report.get("tables", {})
    neuron_name = next((n for n in tables if "neurons" in n.lower()), None)
    conn_candidates = [n for n in tables if "connections" in n.lower() and "synapse" not in n.lower()]
    conn_name = next((n for n in conn_candidates if n.endswith(".feather")), None) or (conn_candidates[0] if conn_candidates else None)
    mapping: dict[str, Any] = {"neuron_table": neuron_name, "connection_table": conn_name}
    if neuron_name:
        cols = tables[neuron_name]["columns"]
        cand = infer_column_candidates(cols)
        mapping["neuron_id_column"] = choose_first(cols, NEURON_PREFS["body"], cand["body_id"])
        mapping["cell_type_column"] = choose_first(cols, NEURON_PREFS["type"], cand["cell_type"])
        mapping["neuron_roi_column"] = choose_first(cols, NEURON_PREFS["roi"], cand["roi"])
        mapping["coordinate_column"] = choose_first(cols, NEURON_PREFS["coord"], cand["coordinate"])
    if conn_name:
        cols = tables[conn_name]["columns"]
        cand = infer_column_candidates(cols)
        mapping["pre_column"] = choose_first(cols, EDGE_PREFS["pre"], cand["pre"])
        mapping["post_column"] = choose_first(cols, EDGE_PREFS["post"], cand["post"])
        mapping["weight_column"] = choose_first(cols, EDGE_PREFS["weight"], cand["weight"])
        mapping["edge_roi_column"] = choose_first(cols, EDGE_PREFS["roi"], cand["roi"])
    return mapping


def parse_point(value: Any) -> tuple[float | None, float | None, float | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None, None
    text = str(value)
    nums = []
    for token in text.replace("{", " ").replace("}", " ").replace(",", " ").split():
        try:
            nums.append(float(token))
        except ValueError:
            continue
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return None, None, None


def load_neuron_metadata(data_dir: str | Path, mapping: dict[str, Any], max_nodes: int | None = None) -> pd.DataFrame:
    table = Path(data_dir) / mapping["neuron_table"]
    cols = [c for c in [mapping.get("neuron_id_column"), mapping.get("cell_type_column"), mapping.get("neuron_roi_column"), mapping.get("coordinate_column")] if c]
    df = feather.read_table(table, columns=cols).to_pandas()
    if max_nodes:
        df = df.head(max_nodes)
    rename = {
        mapping.get("neuron_id_column"): "body_id",
        mapping.get("cell_type_column"): "cell_type",
        mapping.get("neuron_roi_column"): "roi_info",
        mapping.get("coordinate_column"): "coordinate",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k})
    if "body_id" not in df:
        raise ValueError("No neuron body ID column was identified.")
    df = df.drop_duplicates("body_id")
    if "cell_type" not in df:
        df["cell_type"] = "unknown"
    df["cell_type"] = df["cell_type"].fillna("unknown").astype(str)
    if "roi_info" not in df:
        df["roi_info"] = ""
    if "coordinate" in df:
        coords = df["coordinate"].map(parse_point)
        df[["x", "y", "z"]] = pd.DataFrame(coords.tolist(), index=df.index)
    return df


def load_connection_edges(data_dir: str | Path, mapping: dict[str, Any], max_edges: int | None = None) -> pd.DataFrame:
    table = Path(data_dir) / mapping["connection_table"]
    cols = [c for c in [mapping.get("pre_column"), mapping.get("post_column"), mapping.get("weight_column"), mapping.get("edge_roi_column")] if c]
    if table.suffix == ".feather":
        if max_edges is None:
            df = feather.read_table(table, columns=cols).to_pandas()
        else:
            batches = []
            total = 0
            with ipc.open_file(table) as f:
                for i in range(f.num_record_batches):
                    b = f.get_batch(i).select([f.schema.names.index(c) for c in cols]).to_pandas()
                    batches.append(b)
                    total += len(b)
                    if total >= max_edges:
                        break
            df = pd.concat(batches, ignore_index=True).head(max_edges)
    else:
        df = pd.read_csv(table, usecols=cols, nrows=max_edges)
    df = df.rename(
        columns={
            mapping.get("pre_column"): "pre_id",
            mapping.get("post_column"): "post_id",
            mapping.get("weight_column"): "weight",
            mapping.get("edge_roi_column"): "roi_info",
        }
    )
    if "weight" not in df:
        df["weight"] = 1.0
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(float)
    return df


def build_graph_tables(
    data_dir: str | Path,
    schema_report: dict[str, Any],
    max_edges: int | None = None,
    max_nodes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    mapping = schema_report.get("recommended_mapping") or recommend_mapping(schema_report)
    nodes = load_neuron_metadata(data_dir, mapping, max_nodes=max_nodes)
    edges = load_connection_edges(data_dir, mapping, max_edges=max_edges)
    type_map = nodes.set_index("body_id")["cell_type"]
    edges["pre_type"] = edges["pre_id"].map(type_map).fillna("unknown")
    edges["post_type"] = edges["post_id"].map(type_map).fillna("unknown")

    grouped = edges.groupby(["pre_type", "post_type"], dropna=False)
    type_edges = grouped.agg(
        total_weight=("weight", "sum"),
        neuron_edge_count=("weight", "size"),
        unique_pre_neurons=("pre_id", "nunique"),
        unique_post_neurons=("post_id", "nunique"),
    ).reset_index()
    denom = type_edges["unique_pre_neurons"].clip(lower=1) * type_edges["unique_post_neurons"].clip(lower=1)
    type_edges["density_proxy"] = type_edges["neuron_edge_count"] / denom

    type_nodes = pd.DataFrame({"cell_type": sorted(set(type_edges["pre_type"]) | set(type_edges["post_type"]))})
    out_w = type_edges.groupby("pre_type")["total_weight"].sum()
    in_w = type_edges.groupby("post_type")["total_weight"].sum()
    type_nodes["out_weight"] = type_nodes["cell_type"].map(out_w).fillna(0.0)
    type_nodes["in_weight"] = type_nodes["cell_type"].map(in_w).fillna(0.0)
    counts = nodes["cell_type"].value_counts()
    type_nodes["neuron_count"] = type_nodes["cell_type"].map(counts).fillna(0).astype(int)

    neuron_summary = graph_summary(edges, nodes, "neuron")
    type_summary = graph_summary(
        type_edges.rename(columns={"pre_type": "pre_id", "post_type": "post_id", "total_weight": "weight"}),
        type_nodes.rename(columns={"cell_type": "body_id"}),
        "type",
    )
    return nodes, edges, type_nodes, type_edges, neuron_summary, type_summary


def graph_summary(edges: pd.DataFrame, nodes: pd.DataFrame, graph_kind: str) -> dict[str, Any]:
    g = nx.DiGraph()
    g.add_nodes_from(nodes.iloc[:, 0].astype(str).tolist())
    g.add_weighted_edges_from(
        [(str(r.pre_id), str(r.post_id), float(r.weight)) for r in edges[["pre_id", "post_id", "weight"]].itertuples(index=False)]
    )
    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    hubs = sorted(
        ((n, in_deg.get(n, 0), out_deg.get(n, 0)) for n in g.nodes), key=lambda x: x[1] + x[2], reverse=True
    )[:25]
    weak = nx.number_weakly_connected_components(g) if g.number_of_nodes() else 0
    strong = nx.number_strongly_connected_components(g) if g.number_of_nodes() else 0
    return {
        "graph_kind": graph_kind,
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "weak_components": int(weak),
        "strong_components": int(strong),
        "top_hubs": [{"node": n, "in_degree": i, "out_degree": o} for n, i, o in hubs],
        "missing_metadata_percent": float(nodes.isna().mean().mean() * 100) if len(nodes) else 0.0,
    }


def plot_graph_diagnostics(nodes: pd.DataFrame, edges: pd.DataFrame, type_edges: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    in_deg = edges.groupby("post_id").size()
    out_deg = edges.groupby("pre_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist([in_deg.values, out_deg.values], bins=50, label=["in", "out"], log=True)
    plt.legend()
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "fig_connectome_degree_distribution.png", dpi=160)
    plt.close()

    top = type_edges.nlargest(60, "total_weight")
    pivot = top.pivot_table(index="pre_type", columns="post_type", values="total_weight", aggfunc="sum", fill_value=0)
    plt.figure(figsize=(8, 7))
    plt.imshow(np.log1p(pivot.values), aspect="auto", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
    plt.colorbar(label="log1p(weight)")
    plt.tight_layout()
    plt.savefig(out / "fig_type_graph_adjacency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    labels = (top["pre_type"] + "->" + top["post_type"]).head(25)
    plt.barh(labels[::-1], top["total_weight"].head(25).iloc[::-1])
    plt.xlabel("Total synapse weight")
    plt.tight_layout()
    plt.savefig(out / "fig_top_type_edges.png", dpi=180)
    plt.close()


def type_adjacency_matrix(type_edges: pd.DataFrame, max_types: int = 128) -> tuple[np.ndarray, list[str]]:
    totals = Counter()
    for row in type_edges.itertuples(index=False):
        totals[row.pre_type] += float(row.total_weight)
        totals[row.post_type] += float(row.total_weight)
    types = [t for t, _ in totals.most_common(max_types)]
    idx = {t: i for i, t in enumerate(types)}
    w = np.zeros((len(types), len(types)), dtype=np.float32)
    for row in type_edges.itertuples(index=False):
        if row.pre_type in idx and row.post_type in idx:
            w[idx[row.post_type], idx[row.pre_type]] += float(row.total_weight)
    col_sum = w.sum(axis=0, keepdims=True)
    w = np.divide(w, col_sum, out=np.zeros_like(w), where=col_sum > 0)
    return w, types
