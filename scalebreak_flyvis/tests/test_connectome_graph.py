from __future__ import annotations

import pandas as pd
import pyarrow.feather as feather

from scalebreak.connectome import build_graph_tables


def test_graph_builder_with_toy_data(tmp_path) -> None:
    neurons = pd.DataFrame({"bodyId:long": [1, 2, 3], "type:string": ["A", "B", "B"], "roiInfo:string": ["{}", "{}", "{}"]})
    edges = pd.DataFrame({":START_ID(Body-ID)": [1, 2, 1], ":END_ID(Body-ID)": [2, 3, 3], "weight:int": [5, 2, 1], "roiInfo:string": ["{}", "{}", "{}"]})
    feather.write_feather(neurons, tmp_path / "Neuprint_Neurons.feather")
    feather.write_feather(edges, tmp_path / "Neuprint_Neuron_Connections.feather")
    schema = {
        "tables": {
            "Neuprint_Neurons.feather": {"columns": list(neurons.columns)},
            "Neuprint_Neuron_Connections.feather": {"columns": list(edges.columns)},
        }
    }
    nodes, edge_df, type_nodes, type_edges, ns, ts = build_graph_tables(tmp_path, schema)
    assert len(nodes) == 3
    assert len(edge_df) == 3
    assert len(type_edges) > 0
    assert ns["n_edges"] == 3
    assert ts["n_nodes"] >= 2
