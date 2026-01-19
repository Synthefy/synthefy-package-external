from typing import List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from synthefy_pkg.prior.tree_scm import TreeSCM


def visualize_tree_scm(
    tree_scm: TreeSCM,
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
    Visualize a TreeSCM structure showing the tree-based transformations and their connections.

    Parameters
    ----------
    tree_scm : TreeSCM
        The TreeSCM model to visualize
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
    # Create directed graph
    G = nx.DiGraph()

    # Get number of features and total nodes
    n_features = tree_scm.num_features
    total_nodes = tree_scm.num_causes + tree_scm.num_outputs

    # Convert indices to lists and handle negative indices for output nodes
    input_nodes = torch.arange(tree_scm.num_causes).tolist()
    feature_nodes = indices_X.tolist()
    output_nodes = [n + total_nodes if n < 0 else n for n in indices_y.tolist()]
    all_nodes = input_nodes + feature_nodes + output_nodes

    # Add nodes to main graph
    for i in range(total_nodes):
        if i in all_nodes:
            if i in input_nodes and i in feature_nodes:
                G.add_node(i, label=f"I{i}/X{i}", type="input_feature")
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

    # Add tree nodes and their connections
    tree_nodes = []
    for layer_idx, layer in enumerate(tree_scm.layers):
        if hasattr(layer, "model") and hasattr(layer.model, "trees_"):
            for tree_idx, tree in enumerate(layer.model.trees_):
                feature_indices = tree.tree_.feature
                children_left = tree.tree_.children_left
                children_right = tree.tree_.children_right

                # Add all tree nodes
                for node in range(len(feature_indices)):
                    if feature_indices[node] >= 0:
                        node_label = f"T{layer_idx}_{tree_idx}_{node}"
                        node_idx = len(input_nodes) + len(tree_nodes)
                        tree_nodes.append(node_idx)
                        G_temp.add_node(node_idx, label=node_label, type="tree")

                        # Add edges from input/feature nodes to tree nodes
                        if node == 0:  # Root node
                            for input_idx in input_nodes:
                                G_temp.add_edge(input_idx, node_idx)

                        # Add edges between tree nodes
                        if children_left[node] >= 0:
                            child_idx = len(input_nodes) + len(tree_nodes)
                            G_temp.add_edge(node_idx, child_idx)
                        if children_right[node] >= 0:
                            child_idx = len(input_nodes) + len(tree_nodes) + 1
                            G_temp.add_edge(node_idx, child_idx)

    # Add edges from tree nodes to output nodes
    for output_idx in output_nodes:
        for tree_idx in tree_nodes:
            G_temp.add_edge(tree_idx, output_idx)

    # Find all nodes on paths from sources to targets
    sources = set(input_nodes + feature_nodes)
    targets = set(feature_nodes + output_nodes)

    # Find reachable nodes from sources
    reachable_from_source = set()
    for source in sources:
        if source in G_temp:
            reachable_from_source.update(nx.descendants(G_temp, source))
            reachable_from_source.add(source)

    # Find nodes that can reach targets
    can_reach_target = set()
    G_temp_reversed = G_temp.reverse()
    for target in targets:
        if target in G_temp_reversed:
            can_reach_target.update(nx.descendants(G_temp_reversed, target))
            can_reach_target.add(target)

    # Nodes on any path from source to target
    path_nodes = reachable_from_source & can_reach_target

    # Create final graph with only relevant nodes and edges
    G = nx.DiGraph(G_temp.subgraph(path_nodes))

    # Create figure
    plt.figure(figsize=figsize)

    # Create hierarchical layout
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
    n_trees = len([n for n in G.nodes() if G.nodes[n]["type"] == "tree"])

    # Position output nodes on the right
    output_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "output"]
    for i, node in enumerate(output_nodes):
        pos[node] = np.array([0.9, 0.8 - (i / max(1, n_outputs - 1)) * 0.6])

    # Position tree nodes in the middle
    tree_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "tree"]
    for i, node in enumerate(tree_nodes):
        pos[node] = np.array([0.5, 0.8 - (i / max(1, n_trees - 1)) * 0.6])

    # Position feature nodes
    feature_nodes = [
        n
        for n in G.nodes()
        if G.nodes[n]["type"] in ["feature", "input_feature"]
    ]
    for i, node in enumerate(feature_nodes):
        pos[node] = np.array([0.3, 0.8 - (i / max(1, n_features - 1)) * 0.6])

    # Position input nodes on the left
    input_nodes = [
        n for n in G.nodes() if G.nodes[n]["type"] in ["input", "input_feature"]
    ]
    for i, node in enumerate(input_nodes):
        pos[node] = np.array([0.1, 0.8 - (i / max(1, n_inputs - 1)) * 0.6])

    # Draw nodes with different colors based on type
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=feature_nodes,
        node_color="lightyellow",
        node_size=node_size,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=output_nodes,
        node_color="lightcoral",
        node_size=node_size,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=tree_nodes,
        node_color="lightgreen",
        node_size=int(node_size * 0.7),
        alpha=0.5,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, width=2, alpha=0.6
    )

    # Draw labels
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=font_size, font_weight="bold"
    )

    # Add title and legend
    plt.title("TreeSCM Structure Visualization", pad=20)

    # Add legend
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
            markerfacecolor="lightgreen",
            markersize=10,
            label="Tree Node",
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


def visualize_full_tree_scm(
    tree_scm: TreeSCM,
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
    Visualize the complete TreeSCM structure showing all tree nodes and their connections.

    Parameters
    ----------
    tree_scm : TreeSCM
        The TreeSCM model to visualize
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
    # Create directed graph
    G = nx.DiGraph()

    # Get number of features and total nodes
    # n_features = tree_scm.num_features
    total_nodes = tree_scm.num_causes + tree_scm.num_outputs

    # Convert indices to lists and handle negative indices for output nodes
    input_nodes = torch.arange(tree_scm.num_causes).tolist()
    feature_nodes = indices_X.tolist()
    output_nodes = [n + total_nodes if n < 0 else n for n in indices_y.tolist()]
    all_nodes = input_nodes + feature_nodes + output_nodes

    # Add nodes to main graph
    for i in range(total_nodes):
        if i in all_nodes:
            if i in input_nodes and i in feature_nodes:
                G.add_node(i, label=f"I{i}/X{i}", type="input_feature")
            elif i in input_nodes:
                G.add_node(i, label=f"I{i}", type="input")
            elif i in feature_nodes:
                G.add_node(i, label=f"X{i}", type="feature")
            elif i in output_nodes:
                G.add_node(i, label=f"Y{i}", type="output")

    # Add tree nodes and their connections
    tree_nodes = []
    for layer_idx, layer in enumerate(tree_scm.layers):
        if hasattr(layer, "model") and hasattr(layer.model, "trees_"):
            for tree_idx, tree in enumerate(layer.model.trees_):
                feature_indices = tree.tree_.feature
                children_left = tree.tree_.children_left
                children_right = tree.tree_.children_right

                # Add all tree nodes
                for node in range(len(feature_indices)):
                    if feature_indices[node] >= 0:
                        node_label = f"T{layer_idx}_{tree_idx}_{node}"
                        node_idx = len(input_nodes) + len(tree_nodes)
                        tree_nodes.append(node_idx)
                        G.add_node(node_idx, label=node_label, type="tree")

                        # Add edges from input/feature nodes to tree nodes
                        if node == 0:  # Root node
                            for input_idx in input_nodes:
                                G.add_edge(input_idx, node_idx)

                        # Add edges between tree nodes
                        if children_left[node] >= 0:
                            child_idx = len(input_nodes) + len(tree_nodes)
                            G.add_edge(node_idx, child_idx)
                        if children_right[node] >= 0:
                            child_idx = len(input_nodes) + len(tree_nodes) + 1
                            G.add_edge(node_idx, child_idx)

    # Add edges from tree nodes to output nodes
    for output_idx in output_nodes:
        for tree_idx in tree_nodes:
            G.add_edge(tree_idx, output_idx)

    _draw_tree_graph(
        G=G,
        figsize=figsize,
        node_size=node_size,
        font_size=font_size,
        save_path=save_path,
        show=show,
        title="TreeSCM Full Structure Visualization",
    )


def visualize_abbreviated_tree_scm(
    tree_scm: TreeSCM,
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
    Visualize an abbreviated TreeSCM structure showing only essential connections.
    Collapses intermediate tree nodes and shows direct connections between inputs and outputs.

    Parameters
    ----------
    tree_scm : TreeSCM
        The TreeSCM model to visualize
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
    # Create directed graph
    G = nx.DiGraph()

    # Get number of features and total nodes
    # n_features = tree_scm.num_features
    total_nodes = tree_scm.num_causes + tree_scm.num_outputs

    # Convert indices to lists and handle negative indices for output nodes
    input_nodes = torch.arange(tree_scm.num_causes).tolist()
    feature_nodes = indices_X.tolist()
    output_nodes = [n + total_nodes if n < 0 else n for n in indices_y.tolist()]
    all_nodes = input_nodes + feature_nodes + output_nodes

    # Add nodes to main graph
    for i in range(total_nodes):
        if i in all_nodes:
            if i in input_nodes and i in feature_nodes:
                G.add_node(i, label=f"I{i}/X{i}", type="input_feature")
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

    # Add tree nodes and their connections
    tree_nodes = []
    for layer_idx, layer in enumerate(tree_scm.layers):
        if hasattr(layer, "model") and hasattr(layer.model, "trees_"):
            for tree_idx, tree in enumerate(layer.model.trees_):
                feature_indices = tree.tree_.feature
                children_left = tree.tree_.children_left
                children_right = tree.tree_.children_right

                # Add all tree nodes
                for node in range(len(feature_indices)):
                    if feature_indices[node] >= 0:
                        node_label = f"T{layer_idx}_{tree_idx}_{node}"
                        node_idx = len(input_nodes) + len(tree_nodes)
                        tree_nodes.append(node_idx)
                        G_temp.add_node(node_idx, label=node_label, type="tree")

                        # Add edges from input/feature nodes to tree nodes
                        if node == 0:  # Root node
                            for input_idx in input_nodes:
                                G_temp.add_edge(input_idx, node_idx)

                        # Add edges between tree nodes
                        if children_left[node] >= 0:
                            child_idx = len(input_nodes) + len(tree_nodes)
                            G_temp.add_edge(node_idx, child_idx)
                        if children_right[node] >= 0:
                            child_idx = len(input_nodes) + len(tree_nodes) + 1
                            G_temp.add_edge(node_idx, child_idx)

    # Add edges from tree nodes to output nodes
    for output_idx in output_nodes:
        for tree_idx in tree_nodes:
            G_temp.add_edge(tree_idx, output_idx)

    # Find all nodes on paths from sources to targets
    sources = set(input_nodes + feature_nodes)
    targets = set(feature_nodes + output_nodes)

    # Find reachable nodes from sources
    reachable_from_source = set()
    for source in sources:
        if source in G_temp:
            reachable_from_source.update(nx.descendants(G_temp, source))
            reachable_from_source.add(source)

    # Find nodes that can reach targets
    can_reach_target = set()
    G_temp_reversed = G_temp.reverse()
    for target in targets:
        if target in G_temp_reversed:
            can_reach_target.update(nx.descendants(G_temp_reversed, target))
            can_reach_target.add(target)

    # Nodes on any path from source to target
    path_nodes = reachable_from_source & can_reach_target

    # Create final graph with only relevant nodes and edges
    G = nx.DiGraph(G_temp.subgraph(path_nodes))

    _draw_tree_graph(
        G=G,
        figsize=figsize,
        node_size=node_size,
        font_size=font_size,
        save_path=save_path,
        show=show,
        title="TreeSCM Abbreviated Structure Visualization",
    )


def _draw_tree_graph(
    G: nx.DiGraph,
    figsize: Tuple[int, int],
    node_size: int,
    font_size: int,
    save_path: Optional[str],
    show: bool,
    title: str,
) -> None:
    """
    Helper function to draw the tree graph with consistent styling.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to draw
    figsize : Tuple[int, int]
        Figure size in inches
    node_size : int
        Size of nodes in the graph
    font_size : int
        Font size for node labels
    save_path : Optional[str]
        If provided, save the figure to this path
    show : bool
        Whether to display the figure
    title : str
        Title for the plot
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Create hierarchical layout
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
    n_trees = len([n for n in G.nodes() if G.nodes[n]["type"] == "tree"])

    # Position output nodes on the right
    output_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "output"]
    for i, node in enumerate(output_nodes):
        pos[node] = np.array([0.9, 0.8 - (i / max(1, n_outputs - 1)) * 0.6])

    # Position tree nodes in the middle
    tree_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "tree"]
    for i, node in enumerate(tree_nodes):
        pos[node] = np.array([0.5, 0.8 - (i / max(1, n_trees - 1)) * 0.6])

    # Position feature nodes
    feature_nodes = [
        n
        for n in G.nodes()
        if G.nodes[n]["type"] in ["feature", "input_feature"]
    ]
    for i, node in enumerate(feature_nodes):
        pos[node] = np.array([0.3, 0.8 - (i / max(1, n_features - 1)) * 0.6])

    # Position input nodes on the left
    input_nodes = [
        n for n in G.nodes() if G.nodes[n]["type"] in ["input", "input_feature"]
    ]
    for i, node in enumerate(input_nodes):
        pos[node] = np.array([0.1, 0.8 - (i / max(1, n_inputs - 1)) * 0.6])

    # Draw nodes with different colors based on type
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=feature_nodes,
        node_color="lightyellow",
        node_size=node_size,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=output_nodes,
        node_color="lightcoral",
        node_size=node_size,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=tree_nodes,
        node_color="lightgreen",
        node_size=int(node_size * 0.7),
        alpha=0.5,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, width=2, alpha=0.6
    )

    # Draw labels
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=font_size, font_weight="bold"
    )

    # Add title and legend
    plt.title(title, pad=20)

    # Add legend
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
            markerfacecolor="lightgreen",
            markersize=10,
            label="Tree Node",
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
