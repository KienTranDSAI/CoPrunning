"""
Visualization utilities for pruning analysis.

Provides functions to visualize weight distributions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_layer_value_distribution(layer_weights, layer_name, save_dir="assets/layer_analysis",
                                       suffix="", layer_idx=None):
    """
    Visualize the value distribution of a layer's weights.

    Creates a multi-panel visualization showing:
    1. Histogram of all weights
    2. Histogram of non-zero weights only
    3. Statistics summary
    4. Per-sublayer statistics

    Args:
        layer_weights: Weight tensors (dict of {sublayer_name: tensor})
        layer_name: Name/identifier for the layer
        save_dir: Directory to save the visualization
        suffix: Suffix to add to filename (e.g., "before", "after")
        layer_idx: Optional layer index for better naming

    Returns:
        Path to the saved figure
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if layer_idx is not None:
        filename = f"layer_{layer_idx:03d}_{suffix}.png" if suffix else f"layer_{layer_idx:03d}.png"
    else:
        filename = f"{layer_name}_{suffix}.png" if suffix else f"{layer_name}.png"

    full_path = save_path / filename

    # Flatten all weights from all sublayers
    weights_flat = []
    sublayer_stats = {}

    for sublayer_name in layer_weights:
        w = layer_weights[sublayer_name].detach().cpu().numpy().flatten()
        weights_flat.extend(w)

        # Compute per-sublayer stats
        w_nonzero = w[w != 0]
        sparsity = (w == 0).sum() / len(w)
        sublayer_stats[sublayer_name] = {
            'total': len(w),
            'nonzero': len(w_nonzero),
            'sparsity': sparsity,
            'mean': w.mean(),
            'std': w.std(),
            'mean_nonzero': w_nonzero.mean() if len(w_nonzero) > 0 else 0,
            'std_nonzero': w_nonzero.std() if len(w_nonzero) > 0 else 0,
        }

    weights_flat = np.array(weights_flat)

    # Filter out zeros for better visualization of non-zero weights
    weights_nonzero = weights_flat[weights_flat != 0]

    # Calculate overall statistics
    sparsity = (weights_flat == 0).sum() / len(weights_flat)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    title_suffix = f" ({suffix})" if suffix else ""
    fig.suptitle(f'Layer {layer_idx if layer_idx is not None else layer_name} - Weight Distribution{title_suffix}',
                 fontsize=16, fontweight='bold')

    # 1. Histogram of all weights
    ax = axes[0, 0]
    ax.hist(weights_flat, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'All Weights\nSparsity: {sparsity:.2%}', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    ax.legend()

    # 2. Histogram of non-zero weights only
    ax = axes[0, 1]
    if len(weights_nonzero) > 0:
        ax.hist(weights_nonzero, bins=100, alpha=0.7, edgecolor='black', color='forestgreen')
        ax.set_xlabel('Weight Value (non-zero only)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Non-Zero Weights Only\n({len(weights_nonzero):,} values)', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'All weights are zero', ha='center', va='center', fontsize=14)
        ax.axis('off')

    # 3. Overall statistics summary
    ax = axes[1, 0]
    ax.axis('off')

    stats_text = f"""
Weight Distribution Statistics:

OVERALL:
  Total weights:        {len(weights_flat):,}
  Non-zero weights:     {len(weights_nonzero):,}
  Zero weights:         {len(weights_flat) - len(weights_nonzero):,}
  Sparsity:             {sparsity:.2%}

  Mean (all):           {weights_flat.mean():.6f}
  Std (all):            {weights_flat.std():.6f}
  Min:                  {weights_flat.min():.6f}
  Max:                  {weights_flat.max():.6f}

  Mean (non-zero):      {weights_nonzero.mean() if len(weights_nonzero) > 0 else 0:.6f}
  Std (non-zero):       {weights_nonzero.std() if len(weights_nonzero) > 0 else 0:.6f}
"""

    ax.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes)

    # 4. Per-sublayer statistics
    ax = axes[1, 1]
    ax.axis('off')

    sublayer_text = "PER-SUBLAYER STATISTICS:\n\n"
    for name, stats in sorted(sublayer_stats.items()):
        sublayer_text += f"{name}:\n"
        sublayer_text += f"  Sparsity: {stats['sparsity']:.2%}\n"
        sublayer_text += f"  Mean:     {stats['mean']:.6f}\n"
        sublayer_text += f"  Non-zero: {stats['nonzero']:,}/{stats['total']:,}\n\n"

    ax.text(0.05, 0.95, sublayer_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout()

    # Save figure
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"  Saved weight distribution to {full_path}")

    plt.close(fig)

    return full_path
