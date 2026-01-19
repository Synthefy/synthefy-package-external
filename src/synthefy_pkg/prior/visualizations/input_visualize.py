import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from synthefy_pkg.prior.mlp_scm import MLPSCM


def visualize_dag_signals(
    dag: MLPSCM,
    num_samples: int = 1000,
    output_dir: str = "figures/dags",
    batch_idx: int = 100,
) -> None:
    """Visualizes the signal flow through a DAG by plotting each input node and its connected nodes.

    Parameters
    ----------
    dag : MLPSCM
        The DAG model to visualize
    num_samples : int, default=1000
        Number of samples to generate for visualization
    """
    # Get intermediate outputs
    dag_sample = dag.xsampler.sample(return_signal_types=True)
    assert isinstance(dag_sample, tuple), "must be 2-tensor"
    causes, signal_types = dag_sample
    print(signal_types, dag.xsampler, dag.xsampler.sampling)

    assert isinstance(causes, torch.Tensor), "All outputs must be tensors"
    outputs = [causes]
    for layer in dag.layers:
        assert isinstance(outputs[-1], torch.Tensor), (
            "All outputs must be tensors"
        )
        outputs.append(layer(outputs[-1]))
    outputs = outputs  # Skip first layer as in forward()

    # Flatten all intermediate outputs
    outputs_flat = torch.cat(outputs, dim=-1)

    # Get weight matrices to determine connections
    weight_matrices = dag.weight_matrices

    # For each input node, create a figure
    for input_node, signal_type in zip(range(dag.num_causes), signal_types):
        # Find connected nodes in first hidden layer
        first_layer_weights = weight_matrices[0]
        connected_nodes = (
            torch.where(first_layer_weights[input_node] > 1e-3)[0].cpu().numpy()
            + dag.num_causes
        )
        assert isinstance(connected_nodes, np.ndarray), (
            "Connected nodes must be a numpy array"
        )

        if len(connected_nodes) == 0:
            continue

        # Select one connected node from first layer
        connected_node = connected_nodes[
            np.random.randint(0, len(connected_nodes))
        ]

        # Find a random downstream node that's connected to our input
        # by checking subsequent weight matrices
        downstream_node = input_node
        last_layer_size = dag.num_causes
        at = 0
        for layer_idx, weights in enumerate(weight_matrices, 0):
            # Get nodes in this layer that are connected to our connected_node
            next_connected = torch.where(
                weights[:, downstream_node - at] > 1e-3
            )[0]
            if len(next_connected) == 0 or np.random.rand() < 0.3:
                break
            at = at + last_layer_size
            last_layer_size = weights.shape[0]
            downstream_node = (
                next_connected[np.random.randint(0, len(next_connected))] + at
            )

        if downstream_node is None:
            continue

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

        # Sort data by input node value if using fixed inputs
        sort_idx_np = None
        if dag.sampling == "mixed":
            sort_idx = torch.argsort(outputs_flat[:, input_node])
            sort_idx_np = sort_idx.cpu().numpy()

        # Plot input node
        input_values = outputs_flat[:, input_node].detach().cpu().numpy()
        ax1.plot(
            input_values[sort_idx_np]
        ) if sort_idx_np is not None else ax1.plot(input_values)
        ax1.set_title(f"Input Node {input_node} Signal ({signal_type})")
        ax1.set_xlabel("Sorted Index")
        ax1.set_ylabel("Value")

        # Plot connected node from first layer
        connected_values = (
            outputs_flat[:, connected_node].detach().cpu().numpy()
        )
        ax2.plot(
            connected_values[sort_idx_np]
        ) if sort_idx_np is not None else ax2.plot(connected_values)
        ax2.set_title(f"Connected Node {connected_node} (First Layer)")
        ax2.set_xlabel("Sorted Index")
        ax2.set_ylabel("Value")

        # Plot downstream node
        downstream_values = (
            outputs_flat[:, downstream_node].detach().cpu().numpy()
        )
        ax3.plot(
            downstream_values[sort_idx_np]
        ) if sort_idx_np is not None else ax3.plot(downstream_values)
        ax3.set_title(
            f"Downstream Node {downstream_node} (Layer {layer_idx + 1})"
        )
        ax3.set_xlabel("Sorted Index")
        ax3.set_ylabel("Value")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"input_signal_flow_{batch_idx}_{input_node}.png"
            )
        )
        plt.close()
