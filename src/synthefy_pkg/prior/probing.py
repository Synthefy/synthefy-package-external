from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn

from synthefy_pkg.prior.activations import get_activation_name
from synthefy_pkg.prior.mlp_scm import MLPSCM


def construct_mlp_dag(
    dag: MLPSCM,
    indices_X: torch.Tensor,
    indices_y: torch.Tensor,
    weight_threshold: float = 1e-6,
) -> nx.DiGraph:
    """
    Construct a DAG from an MLP that includes all nodes that are part of paths
    between indices_X and indices_y.

    This function creates a directed graph representation of the MLP structure,
    including all intermediate nodes that lie on any path from input/feature nodes
    to feature/output nodes.

    Parameters
    ----------
    dag : MLPSCM
        The MLP-based Structural Causal Model
    indices_X : torch.Tensor
        Indices of feature nodes
    indices_y : torch.Tensor
        Indices of output nodes
    weight_threshold : float, default=1e-6
        Threshold for considering a weight as non-zero (creating an edge)

    Returns
    -------
    nx.DiGraph
        A directed graph representing the DAG structure with the following node attributes:
        - 'label': Human-readable label for the node
        - 'type': Type of node ('input', 'feature', 'output', 'hidden', 'input_feature', 'input_output')
        - 'layer': Layer index the node belongs to
        - 'local_index': Local index within the layer
        - 'global_index': Global node index across all layers
    """
    # Create directed graph
    G = nx.DiGraph()

    # Get number of features from the DAG's actual structure
    # n_features = dag.num_features
    total_nodes = dag.num_causes + sum(
        [
            layer[1].weight.shape[0]
            if isinstance(layer, nn.Sequential)
            else layer.weight.shape[0]
            for layer in dag.layers
        ]
    )
    hidden_size = (
        dag.layers[0][1].weight.shape[0]
        if isinstance(dag.layers[0], nn.Sequential)
        else dag.layers[0].weight.shape[0]
    )

    # Convert indices to lists and handle negative indices for output nodes
    input_nodes = torch.arange(dag.num_causes).tolist()
    feature_nodes = indices_X.tolist()
    # Convert negative indices to positive positions for output nodes
    output_nodes = [n + total_nodes if n < 0 else n for n in indices_y.tolist()]
    all_nodes = input_nodes + feature_nodes + output_nodes

    # Create node index lookup for mapping (layer, local_idx) to global index
    node_index_lookup = {(0, i): i for i in input_nodes}
    counter = 0
    for layer_idx in range(len(dag.layers)):
        for i in range(hidden_size):
            node_index_lookup[(layer_idx + 1, i)] = len(input_nodes) + counter
            counter += 1

    # Create a temporary graph with all nodes and edges
    G_temp = nx.DiGraph()

    # Add input, feature, and output nodes to temporary graph
    for i in range(total_nodes):
        if i in all_nodes:
            if i in input_nodes and i in feature_nodes:
                G_temp.add_node(
                    i,
                    label=f"I{i}/X{i}",
                    type="input_feature",
                    layer=0,
                    local_index=i,
                    global_index=i,
                )
            elif i in input_nodes and i in output_nodes:
                G_temp.add_node(
                    i,
                    label=f"I{i}/Y{i}",
                    type="input_output",
                    layer=0,
                    local_index=i,
                    global_index=i,
                )
            elif i in input_nodes:
                G_temp.add_node(
                    i,
                    label=f"I{i}",
                    type="input",
                    layer=0,
                    local_index=i,
                    global_index=i,
                )
            elif i in feature_nodes:
                G_temp.add_node(
                    i,
                    label=f"X{i}",
                    type="feature",
                    layer=0,  # Features are from input layer
                    local_index=i,
                    global_index=i,
                )
            elif i in output_nodes:
                G_temp.add_node(
                    i,
                    label=f"Y{i}",
                    type="output",
                    layer=len(dag.layers),
                    local_index=i - (total_nodes - dag.num_outputs),
                    global_index=i,
                )

    # Add all intermediate nodes and edges based on MLP weights
    for layer_idx, layer in enumerate(dag.layers):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Sequential):
            if isinstance(layer, nn.Sequential):
                layer = layer[1]
                assert isinstance(layer, nn.Linear), (
                    "Sequential block should have a Linear layer in first position"
                )

            weights = layer.weight.data

            # Add intermediate nodes for this layer
            for j in range(weights.shape[0]):
                node_idx = node_index_lookup[(layer_idx + 1, j)]
                if node_idx not in G_temp.nodes():
                    G_temp.add_node(
                        node_idx,
                        label=f"H{layer_idx}_{j}",
                        type="hidden",
                        layer=layer_idx + 1,
                        local_index=j,
                        global_index=node_idx,
                    )

                # Add edges from input nodes based on weights
                for i in range(weights.shape[1]):
                    weight_value = weights[j, i].item()
                    if torch.abs(weights[j, i]) > weight_threshold:
                        G_temp.add_edge(
                            node_index_lookup[(layer_idx, i)],
                            node_index_lookup[(layer_idx + 1, j)],
                            weight=weight_value,
                        )

    # Find all nodes on paths from sources to targets
    sources = set(input_nodes + feature_nodes)
    targets = set(feature_nodes + output_nodes)

    # 1. For each source, find all reachable nodes (forward BFS)
    reachable_from_source = set()
    for source in sources:
        if source in G_temp:
            reachable_from_source.update(nx.descendants(G_temp, source))
            reachable_from_source.add(source)

    # 2. For each target, find all nodes that can reach it (reverse BFS)
    can_reach_target = set()
    G_temp_reversed = G_temp.reverse()
    for target in targets:
        if target in G_temp_reversed:
            can_reach_target.update(nx.descendants(G_temp_reversed, target))
            can_reach_target.add(target)

    # 3. Nodes on any path from a source to a target
    path_nodes = reachable_from_source & can_reach_target

    # Create final graph with only nodes in paths
    G = nx.DiGraph()

    # Add all nodes that are part of paths
    for node in path_nodes:
        G.add_node(node, **G_temp.nodes[node])

    # Add all input, feature, and output nodes even if not in paths
    for node in all_nodes:
        if node not in path_nodes:
            G.add_node(node, **G_temp.nodes[node])

    # Add edges between these nodes
    for u, v, data in G_temp.edges(data=True):
        if u in G and v in G:
            G.add_edge(u, v, **data)
    return G


def get_dag_statistics(dag: nx.DiGraph) -> Dict[str, Any]:
    """
    Get statistics about the constructed DAG.

    Parameters
    ----------
    dag : nx.DiGraph
        The constructed DAG

    Returns
    -------
    Dict[str, Any]
        Dictionary containing various statistics about the DAG
    """
    stats = {
        "total_nodes": dag.number_of_nodes(),
        "total_edges": dag.number_of_edges(),
        "node_types": {},
        "layer_distribution": {},
        "in_degrees": {},
        "out_degrees": {},
    }

    # Count node types
    for node, data in dag.nodes(data=True):
        node_type = data.get("type", "unknown")
        stats["node_types"][node_type] = (
            stats["node_types"].get(node_type, 0) + 1
        )

        # Count layer distribution
        layer = data.get("layer", -1)
        stats["layer_distribution"][layer] = (
            stats["layer_distribution"].get(layer, 0) + 1
        )

    # Calculate degree statistics
    in_degrees = []
    out_degrees = []
    for node in dag.nodes():
        in_deg = dag.in_degree(node)
        out_deg = dag.out_degree(node)
        in_degrees.append(in_deg if isinstance(in_deg, int) else len(in_deg))
        out_degrees.append(
            out_deg if isinstance(out_deg, int) else len(out_deg)
        )

    stats["in_degrees"] = {
        "min": min(in_degrees) if in_degrees else 0,
        "max": max(in_degrees) if in_degrees else 0,
        "mean": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
    }

    stats["out_degrees"] = {
        "min": min(out_degrees) if out_degrees else 0,
        "max": max(out_degrees) if out_degrees else 0,
        "mean": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
    }

    return stats


def get_connectivity_statistics(
    G: nx.DiGraph,
    dag: MLPSCM,
    indices_X: torch.Tensor,
    indices_y: torch.Tensor,
    weight_threshold: float = 1e-6,
) -> Dict[str, Any]:
    """
    Get connectivity statistics about indices_X and targets in the DAG.

    This function analyzes:
    1. How many indices_X are not connected to anything in the graph (not on any path)
    2. How many nodes in the previous layer each target is connected to

    Parameters
    ----------
    dag : nx.DiGraph
        The constructed DAG
    mlp_scm : MLPSCM
        The MLP-based Structural Causal Model
    indices_X : torch.Tensor
        Indices of feature nodes
    indices_y : torch.Tensor
        Indices of output nodes
    weight_threshold : float, default=1e-6
        Threshold for considering a weight as non-zero (creating an edge)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing connectivity statistics with the following structure:
        {
            'indices_X_connectivity': {
                'total_indices_X': int,
                'disconnected_indices_X': int,
                'disconnected_ratio': float,
                'disconnected_indices': List[int]
            },
            'target_connectivity': {
                'target_connections': Dict[int, Dict[str, Any]]
            }
        }
    """
    stats = {"correlates_connectivity": {}, "target_connectivity": {}}

    # 1. Analyze indices_X connectivity
    feature_nodes = indices_X.tolist()
    total_indices_X = len(feature_nodes)
    stats["activations_info"] = get_activations_for_indices(
        dag, indices_X, indices_y
    )

    # For each X node, count incoming edges (from previous layer)
    incoming_edge_counts = {}
    incoming_edge_weights = {}
    disconnected_indices = []
    for node_idx in feature_nodes:
        in_deg = G.in_degree(node_idx)
        incoming_edge_counts[node_idx] = in_deg
        incoming_edge_weights[node_idx] = [
            data["weight"] for _, _, data in G.in_edges(node_idx, data=True)
        ]
        if in_deg == 0:
            disconnected_indices.append(node_idx)

    disconnected_count = len(disconnected_indices)
    disconnected_ratio = (
        disconnected_count / total_indices_X if total_indices_X > 0 else 0.0
    )

    stats["correlates_connectivity"] = {
        "total_correlates": total_indices_X,
        "disconnected_correlates": disconnected_count,
        "correlate_disconnected_ratio": disconnected_ratio,
        "disconnected_indices": disconnected_indices,
        "incoming_edge_counts": incoming_edge_counts,
        "incoming_edge_weights": incoming_edge_weights,
    }

    # 2. Analyze target connectivity
    target_connections = {}

    for target_idx in indices_y.tolist():
        # Get layer and local index for the target
        layer_idx, local_idx = dag.get_local_index_for_node(target_idx)

        if layer_idx == -1:
            # Invalid target index
            target_connections[target_idx] = {
                "layer_idx": -1,
                "local_idx": -1,
                "connections_to_previous_layer": 0,
                "connection_ratio": 0.0,
                "is_reachable": False,
                "error": "Invalid target index",
            }
            continue

        if layer_idx == 0:
            # Target is in input layer
            target_connections[target_idx] = {
                "layer_idx": layer_idx,
                "local_idx": local_idx,
                "connections_to_previous_layer": 0,
                "previous_layer_connection_ratio": 0.0,
                "is_reachable": True,
                "note": "Target is in input layer",
            }
            continue

        # Get weights from previous layer to this target
        if isinstance(dag.layers[layer_idx - 1], nn.Sequential):
            weights = abs(dag.layers[layer_idx - 1][1].weight[local_idx, :])
        else:
            weights = abs(dag.layers[layer_idx - 1].weight[local_idx, :])

        # Count significant connections (similar to is_target_is_reachable)
        significant_connections = torch.sum(
            (weights > weight_threshold).float()
        ).item()
        total_connections = weights.shape[0]
        connection_ratio = (
            significant_connections / total_connections
            if total_connections > 0
            else 0.0
        )

        # Determine if reachable (same logic as is_target_is_reachable)
        is_reachable = connection_ratio >= 0.1

        target_connections[target_idx] = {
            "layer_idx": layer_idx,
            "local_idx": local_idx,
            "connections_to_previous_layer": int(significant_connections),
            "total_possible_connections": total_connections,
            "connection_ratio": connection_ratio,
            "is_reachable": is_reachable,
        }

    stats["target_connectivity"] = target_connections

    return stats


def get_nodes_by_type(dag: nx.DiGraph, node_type: str) -> List[int]:
    """
    Get all nodes of a specific type from the DAG.

    Parameters
    ----------
    dag : nx.DiGraph
        The constructed DAG
    node_type : str
        Type of nodes to retrieve ('input', 'feature', 'output', 'hidden', etc.)

    Returns
    -------
    List[int]
        List of node indices of the specified type
    """
    return [
        node
        for node, data in dag.nodes(data=True)
        if data.get("type") == node_type
    ]


def get_nodes_by_layer(dag: nx.DiGraph, layer: int) -> List[int]:
    """
    Get all nodes from a specific layer in the DAG.

    Parameters
    ----------
    dag : nx.DiGraph
        The constructed DAG
    layer : int
        Layer index to retrieve nodes from

    Returns
    -------
    List[int]
        List of node indices from the specified layer
    """
    return [
        node
        for node, data in dag.nodes(data=True)
        if data.get("layer") == layer
    ]


def get_path_nodes(dag: nx.DiGraph, source: int, target: int) -> List[int]:
    """
    Get all nodes that lie on paths from source to target.

    Parameters
    ----------
    dag : nx.DiGraph
        The constructed DAG
    source : int
        Source node index
    target : int
        Target node index

    Returns
    -------
    List[int]
        List of node indices that lie on paths from source to target
    """
    if source not in dag or target not in dag:
        return []

    # Find all simple paths from source to target
    try:
        paths = list(nx.all_simple_paths(dag, source, target))
    except nx.NetworkXNoPath:
        return []

    # Get all unique nodes that appear in any path
    path_nodes = set()
    for path in paths:
        path_nodes.update(path)

    return list(path_nodes)


def get_activations_for_indices(
    dag: MLPSCM,
    indices_X: torch.Tensor,
    indices_y: torch.Tensor,
) -> Dict[str, Any]:
    """
    Get the activations associated with each index in indices_X and indices_y.

    This function analyzes the MLPSCM structure to determine which activation
    function is applied to each node in indices_X and indices_y.

    Parameters
    ----------
    dag : MLPSCM
        The MLP-based Structural Causal Model
    indices_X : torch.Tensor
        Indices of feature nodes
    indices_y : torch.Tensor
        Indices of output nodes

    Returns
    -------
    Dict[str, Any]
        Dictionary containing activation information with the following structure:
        {
            'feature_activations': Dict[int, str],
            'target_activations': Dict[int, str],
            'layer_activations': Dict[int, str]
        }
    """
    activations_info = {
        "feature_activations": {},
        "target_activations": {},
        "layer_activations": {},
    }

    # Get activation names for each layer
    for layer_idx, layer in enumerate(dag.layers):
        if isinstance(layer, nn.Sequential):
            # For Sequential layers, activation is at index 0
            activation = layer[0]
            assert isinstance(activation, nn.Module), (
                "Activation is not a module"
            )
            activation_name = get_activation_name(activation)
        else:
            # For Linear layers (first layer), no activation
            activation_name = "None"

        activations_info["layer_activations"][layer_idx] = activation_name

    # Get activations for feature nodes (indices_X)
    for idx in indices_X.tolist():
        layer_idx, local_idx = dag.get_local_index_for_node(idx)
        if layer_idx >= 0 and layer_idx < len(dag.layers):
            if layer_idx == 0:
                # Input layer has no activation
                activations_info["feature_activations"][idx] = "None"
            else:
                # Get activation from the layer that produces this node
                layer = dag.layers[
                    layer_idx - 1
                ]  # Previous layer produces this node
                if isinstance(layer, nn.Sequential):
                    activation = layer[0]
                    activations_info["feature_activations"][idx] = (
                        activation.__class__.__name__
                    )
                else:
                    activations_info["feature_activations"][idx] = "None"
        else:
            activations_info["feature_activations"][idx] = "Invalid"

    # Get activations for target nodes (indices_y)
    for idx in indices_y.tolist():
        layer_idx, local_idx = dag.get_local_index_for_node(idx)
        if layer_idx >= 0 and layer_idx < len(dag.layers):
            if layer_idx == 0:
                # Input layer has no activation
                activations_info["target_activations"][idx] = "None"
            else:
                # Get activation from the layer that produces this node
                layer = dag.layers[
                    layer_idx - 1
                ]  # Previous layer produces this node
                if isinstance(layer, nn.Sequential):
                    activation = layer[0]
                    activations_info["target_activations"][idx] = (
                        activation.__class__.__name__
                    )
                else:
                    activations_info["target_activations"][idx] = "None"
        else:
            activations_info["target_activations"][idx] = "Invalid"

    return activations_info
