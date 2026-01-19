from typing import List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from synthefy_pkg.prior.mlp_scm import MLPSCM
from synthefy_pkg.prior.probing import construct_mlp_dag
from synthefy_pkg.prior.tree_scm import TreeSCM


def visualize_full_dag(
    dag: MLPSCM,
    indices_X: torch.Tensor,
    indices_y: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    node_size: int = 2000,
    font_size: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize a DAG including intermediate nodes with different styling.
    Shows all nodes that are part of paths between input, feature, and output nodes.

    Parameters
    ----------
    dag : Union[MLPSCM, TreeSCM]
        The DAG to visualize
    indices_X : torch.Tensor
        Indices of feature nodes
    indices_y : torch.Tensor
        Indices of output nodes
    feature_names : Optional[List[str]], default=None
        Names of the features. If None, will use indices.
    figsize : Tuple[int, int], default=(12, 10)
        Figure size in inches
    node_size : int, default=2000
        Size of nodes in the graph
    font_size : int, default=10
        Font size for node labels
    save_path : Optional[str], default=None
        If provided, save the figure to this path
    show : bool, default=True
        Whether to display the figure
    """
    # Use the refactored construct_mlp_dag function
    G_filtered = construct_mlp_dag(dag, indices_X, indices_y, weight_threshold=1e-6)

    # Get node information for positioning
    input_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n]["type"] in ["input", "input_feature"]]
    feature_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n]["type"] in ["feature", "input_feature"]]
    output_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n]["type"] == "output"]
    hidden_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n]["type"] in ["hidden", "tree"]]
    input_output_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n]["type"] == "input_output"]

    # Create layer mapping for proper node positioning
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

    layer_mapping = []
    for i in range(total_nodes):
        if i < dag.num_causes:
            layer_mapping.append(0)
        elif i < dag.num_causes + (len(dag.layers) - 1) * hidden_size:
            layer_mapping.append(1 + (i - dag.num_causes) // hidden_size)
        else:
            layer_mapping.append(len(dag.layers))

    # Create figure
    plt.figure(figsize=figsize)

    # Create a hierarchical layout with inputs on left and outputs on right
    pos = {}

    # Calculate positions for each node type
    n_inputs = len(input_nodes)
    n_features = len(feature_nodes)
    n_outputs = len(output_nodes)
    n_hidden = len(hidden_nodes)

    # Position output nodes on the right at the top
    for i, node in enumerate(output_nodes):
        layer_pos = 0.1 + 0.9 * (
            layer_mapping[node] / max(1, len(dag.layers) - 1)
        )
        pos[node] = np.array(
            [layer_pos, 0.95 - (i / max(1, n_outputs - 1)) * 0.1]
        )

    # Position feature nodes in the middle with more spread
    for i, node in enumerate(feature_nodes):
        layer_pos = 0.1 + 0.9 * (
            layer_mapping[node] / max(1, len(dag.layers) - 1)
        )
        pos[node] = np.array(
            [layer_pos, 0.8 - (i / max(1, n_features - 1)) * 0.7]
        )

    # Position input nodes on the left with more spread
    for i, node in enumerate(input_nodes):
        pos[node] = np.array([0, 0.8 - (i / max(1, n_inputs - 1)) * 0.7])

    # Position input_output nodes separately
    for i, node in enumerate(input_output_nodes):
        pos[node] = np.array([0, 0.8 - (i / max(1, len(input_output_nodes) - 1)) * 0.7])

    # Position hidden/tree nodes in between
    for i, node in enumerate(hidden_nodes):
        layer_pos = 0.1 + 0.9 * (
            layer_mapping[node] / max(1, len(dag.layers) - 1)
        )
        pos[node] = np.array(
            [layer_pos, 0.5 - (i / max(1, n_hidden - 1)) * 0.4]
        )

    # Draw input nodes
    nx.draw_networkx_nodes(
        G_filtered,
        pos,
        nodelist=input_nodes,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.7,
    )

    # Draw input_output nodes with distinct color
    if input_output_nodes:
        nx.draw_networkx_nodes(
            G_filtered,
            pos,
            nodelist=input_output_nodes,
            node_color="plum",
            node_size=node_size,
            alpha=0.7,
        )

    # Draw feature nodes
    nx.draw_networkx_nodes(
        G_filtered,
        pos,
        nodelist=feature_nodes,
        node_color="lightyellow",
        node_size=node_size,
        alpha=0.7,
    )

    # Draw output nodes
    if output_nodes:
        nx.draw_networkx_nodes(
            G_filtered,
            pos,
            nodelist=output_nodes,
            node_color="lightcoral",
            node_size=node_size,
            alpha=0.7,
        )

    # Draw hidden/tree nodes
    if hidden_nodes:
        nx.draw_networkx_nodes(
            G_filtered,
            pos,
            nodelist=hidden_nodes,
            node_color="lightgray",
            node_size=int(node_size * 0.7),
            alpha=0.5,
        )

    # Draw edges with arrows and fixed width
    nx.draw_networkx_edges(
        G_filtered,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        width=2,
        alpha=0.6,
    )

    # Draw labels
    labels = nx.get_node_attributes(G_filtered, "label")
    nx.draw_networkx_labels(
        G_filtered, pos, labels=labels, font_size=font_size, font_weight="bold"
    )

    # Add title and legend
    dag_type = "MLP" if isinstance(dag, MLPSCM) else "Tree"
    plt.title(
        f"{dag_type} SCM Full Structure (Including All Path Nodes)", pad=20
    )

    # Add legend for node types
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightblue",
            markersize=10,
            label="Input",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="plum",
            markersize=10,
            label="Input/Output",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightyellow",
            markersize=10,
            label="Feature",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightcoral",
            markersize=10,
            label="Output",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightgray",
            markersize=10,
            label="Intermediate",
        ),
    ]
    plt.legend(
        handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1)
    )

    # Remove axis
    plt.axis("off")

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_dag(
    dag: MLPSCM,
    indices_X: torch.Tensor,
    indices_y: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 2000,
    font_size: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize a DAG from an SCM prior using networkx and matplotlib.
    Shows only input, feature, and output nodes with collapsed intermediate connections.

    Parameters
    ----------
    dag : Union[MLPSCM, TreeSCM]
        The DAG to visualize
    indices_X : torch.Tensor
        Indices of feature nodes
    indices_y : torch.Tensor
        Indices of output nodes
    feature_names : Optional[List[str]], default=None
        Names of the features. If None, will use indices.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size in inches
    node_size : int, default=2000
        Size of nodes in the graph
    font_size : int, default=10
        Font size for node labels
    save_path : Optional[str], default=None
        If provided, save the figure to this path
    show : bool, default=True
        Whether to display the figure
    """
    # Create directed graph
    G = nx.DiGraph()

    # Get number of features from the DAG's actual structure
    n_features = dag.num_features
    if isinstance(dag.layers[0], nn.Sequential):
        total_nodes = (
            dag.num_causes
            + (len(dag.layers)) * dag.layers[0][1].weight.shape[0]
        )
        hidden_size = dag.layers[0][1].weight.shape[0]
    else:
        total_nodes = (
            dag.num_causes
            + (len(dag.layers) - 1) * dag.layers[0].weight.shape[0]
            + dag.layers[-1][1].weight.shape[0]
        )
        hidden_size = dag.layers[0].weight.shape[0]

    # Convert indices to lists and handle negative indices for output nodes
    input_nodes = torch.arange(dag.num_causes).tolist()
    feature_nodes = indices_X.tolist()
    # Convert negative indices to positive positions for output nodes
    output_nodes = [n + total_nodes if n < 0 else n for n in indices_y.tolist()]
    all_nodes = input_nodes + feature_nodes + output_nodes

    # Add nodes to main graph
    for i in range(total_nodes):
        if i in all_nodes:
            if i in input_nodes and i in feature_nodes:
                G.add_node(i, label=f"I{i}/X{i}", type="input_feature")
            elif i in input_nodes and i in output_nodes:
                G.add_node(i, label=f"I{i}/Y{i}", type="input_output")
            elif i in input_nodes:
                G.add_node(i, label=f"I{i}", type="input")
            elif i in feature_nodes:
                G.add_node(i, label=f"X{i}", type="feature")
            elif i in output_nodes:
                G.add_node(i, label=f"Y{i}", type="output")

    # Create a temporary graph with all nodes and edges
    G_temp = nx.DiGraph()

    # Add all nodes to temporary graph first
    for node in G.nodes(data=True):
        G_temp.add_node(node[0], **node[1])

    # Add edges based on DAG type
    node_index_lookup = {(0, i): i for i in input_nodes}
    counter = 0
    for layer_idx in range(len(dag.layers)):
        for i in range(hidden_size):
            node_index_lookup[(layer_idx + 1, i)] = len(input_nodes) + counter
            counter += 1

    # if isinstance(dag, MLPSCM):
    # For MLP SCM, add all nodes and edges
    counter = 0
    for layer_idx, layer in enumerate(dag.layers):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Sequential):
            if isinstance(layer, nn.Sequential):
                layer = layer[1]
            weights = layer.weight.data
            # Add intermediate nodes for this layer
            for j in range(weights.shape[0]):
                node_idx = node_index_lookup[(layer_idx + 1, j)]
                if node_idx not in G_temp.nodes():
                    G_temp.add_node(
                        node_idx, label=f"H{layer_idx}_{j}", type="hidden"
                    )
                # Add edges from input nodes
                for i in range(weights.shape[1]):
                    # if node_index_lookup[(layer_idx, i)] in output_nodes:
                    #     print("testing edge ", weights[j, i], node_index_lookup[(layer_idx, i)], node_index_lookup[(layer_idx+1, j)])
                    if torch.abs(weights[j, i]) > 1e-3:
                        # print("adding edge ", node_index_lookup[(layer_idx, i)], node_index_lookup[(layer_idx+1, j)])
                        G_temp.add_edge(
                            node_index_lookup[(layer_idx, i)],
                            node_index_lookup[(layer_idx + 1, j)],
                        )
                counter += 1
    # else:  # TreeSCM
    #     # For Tree SCM, add all tree nodes
    #     for layer in dag.layers:
    #         if hasattr(layer, 'model') and hasattr(layer.model, 'tree_'):
    #             for tree_idx, tree in enumerate(layer.model.trees_):
    #                 feature_indices = tree.tree_.feature
    #                 children_left = tree.tree_.children_left
    #                 children_right = tree.tree_.children_right

    #                 # Add all tree nodes
    #                 for node in range(len(feature_indices)):
    #                     if feature_indices[node] >= 0:
    #                         node_label = f"T{tree_idx}_{node}"
    #                         G_temp.add_node(node + n_features + dag.num_outputs, label=node_label, type='tree')
    #                         # Add edges
    #                         if children_left[node] >= 0:
    #                             G_temp.add_edge(node + n_features + dag.num_outputs, children_left[node] + n_features + dag.num_outputs)
    #                         if children_right[node] >= 0:
    #                             G_temp.add_edge(node + n_features + dag.num_outputs, children_right[node] + n_features + dag.num_outputs)

    # Efficiently find all nodes on any path from sources to targets
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

    # 4. Edges between these nodes
    G = G_temp.subgraph(path_nodes).copy()

    # Create layer mapping for proper node positioning
    layer_mapping = []
    for i in range(total_nodes):
        if i < dag.num_causes:
            layer_mapping.append(0)
        elif i < dag.num_causes + (len(dag.layers) - 1) * hidden_size:
            layer_mapping.append(1 + (i - dag.num_causes) // hidden_size)
        else:
            layer_mapping.append(1 + len(dag.layers))

    # Create figure
    plt.figure(figsize=figsize)

    # Create a hierarchical layout with inputs on left and outputs on right
    pos = {}

    # Calculate positions for each node type
    n_inputs = len(
        [
            n
            for n in G.nodes()
            if G.nodes[n]["type"] in ["input", "input_feature"]
        ]
    )
    n_features = len(
        [
            n
            for n in G.nodes()
            if G.nodes[n]["type"] in ["feature", "input_feature"]
        ]
    )
    n_outputs = len([n for n in G.nodes() if G.nodes[n]["type"] == "output"])

    # Position output nodes on the right at the top
    output_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "output"]
    for i, node in enumerate(output_nodes):
        layer_pos = 0.1 + 0.9 * (
            layer_mapping[node] / max(1, len(dag.layers) - 1)
        )
        pos[node] = np.array(
            [layer_pos, 0.95 - (i / max(1, n_outputs - 1)) * 0.1]
        )

    # Position feature nodes in the middle with more spread
    feature_nodes = [
        n
        for n in G.nodes()
        if G.nodes[n]["type"] in ["feature", "input_feature"]
    ]
    for i, node in enumerate(feature_nodes):
        layer_pos = 0.1 + 0.9 * (
            layer_mapping[node] / max(1, len(dag.layers) - 1)
        )
        pos[node] = np.array(
            [layer_pos, 0.8 - (i / max(1, n_features - 1)) * 0.7]
        )

    # Position input nodes on the left with more spread
    input_nodes = [
        n for n in G.nodes() if G.nodes[n]["type"] in ["input", "input_feature"]
    ]
    for i, node in enumerate(input_nodes):
        pos[node] = np.array([0, 0.8 - (i / max(1, n_inputs - 1)) * 0.7])

    # Position input_output nodes separately
    input_output_nodes = [
        n for n in G.nodes() if G.nodes[n]["type"] == "input_output"
    ]
    for i, node in enumerate(input_output_nodes):
        pos[node] = np.array([0, 0.8 - (i / max(1, len(input_output_nodes) - 1)) * 0.7])

    # Position any remaining nodes at the very bottom
    remaining_nodes = [
        n
        for n in G.nodes()
        if n not in input_nodes + feature_nodes + output_nodes + input_output_nodes
    ]
    for i, node in enumerate(remaining_nodes):
        layer_pos = 0.1 + 0.9 * (
            layer_mapping[node] / max(1, len(dag.layers) - 1)
        )
        pos[node] = np.array(
            [layer_pos, 0.05 + (i / max(1, len(remaining_nodes) - 1)) * 0.05]
        )

    # Draw input nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.7,
    )

    # Draw input_output nodes with distinct color
    if input_output_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=input_output_nodes,
            node_color="plum",
            node_size=node_size,
            alpha=0.7,
        )

    # Draw feature nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=feature_nodes,
        node_color="lightyellow",
        node_size=node_size,
        alpha=0.7,
    )

    # Draw output nodes
    if output_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=output_nodes,
            node_color="lightcoral",
            node_size=node_size,
            alpha=0.7,
        )

    # Draw remaining nodes
    if remaining_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=remaining_nodes,
            node_color="lightgray",
            node_size=int(node_size * 0.7),
            alpha=0.5,
        )

    # Draw edges with arrows and fixed width
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, width=2, alpha=0.6
    )

    # Draw labels
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=font_size, font_weight="bold"
    )

    # Add title and legend
    dag_type = "MLP" if isinstance(dag, MLPSCM) else "Tree"
    plt.title(
        f"{dag_type} SCM Feature Structure (Collapsed Connections)", pad=20
    )

    # Add legend for node types
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightblue",
            markersize=10,
            label="Input",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="plum",
            markersize=10,
            label="Input/Output",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightyellow",
            markersize=10,
            label="Feature",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightcoral",
            markersize=10,
            label="Output",
        ),
    ]
    if remaining_nodes:
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lightgray",
                markersize=10,
                label="Intermediate",
            )
        )
    plt.legend(
        handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1)
    )

    # Remove axis
    plt.axis("off")

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
