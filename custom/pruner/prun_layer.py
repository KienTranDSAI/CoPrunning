"""
Prune a single layer and analyze activation differences before/after pruning.

This script reuses existing pruning infrastructure from the pruning module.

Usage examples:

  # Print all layers and their indices
  python prun_layer.py --print_layers

  # Prune layer 0 with 50% unstructured sparsity (no images saved)
  python prun_layer.py --layer_idx 0 --sparsity_ratio 0.5 --nsamples 10

  # Prune layer 5 with 2:4 structured sparsity (no images saved)
  python prun_layer.py --layer_idx 5 --sparsity_type 2:4 --nsamples 10

  # Save activation difference plot to custom location
  python prun_layer.py --layer_idx 0 --save_plot results/layer0_analysis.png

  # Save weight distribution visualizations (before & after pruning)
  python prun_layer.py --layer_idx 0 --save_visualization

  # Save both activation plot and weight distributions
  python prun_layer.py --layer_idx 0 --save_plot results/activation.png --save_visualization
"""




import argparse
import sys
from pathlib import Path

# Add the custom/pruner directory to Python path
PRUNER_DIR = Path(__file__).parent.resolve()
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from utils.model_utils import load_model, find_layers, prepare_calibration_input
from utils.dataset_loader import get_loaders
from utils.activation_capture import ActivationCapture
from utils.visualization_utils import visualize_layer_value_distribution
from pruning.sparsity_patterns import UnstructuredSparsity, NMSparsity
from pruning.wanda import WandaPruner
from recovery.weight_redistribution import (
    WeightRedistributor,
    InverseWandaStrategy,
)


def print_model_layers(model):
    """
    Print all transformer layers in the model with their indices.

    This helps identify which layer_idx to use for pruning.

    Args:
        model: The loaded model
    """
    layers = model.model.layers
    num_layers = len(layers)

    print("\n" + "="*80)
    print(f"MODEL LAYER STRUCTURE ({num_layers} transformer layers)")
    print("="*80)

    for i, layer in enumerate(layers):
        # Get layer type name
        layer_type = type(layer).__name__

        # Count sublayers (Linear layers)
        sublayers = find_layers(layer)
        num_sublayers = len(sublayers)

        print(f"Layer {i:3d}: {layer_type:30s} ({num_sublayers} Linear sublayers)")

        # Print first few sublayer names as examples
        if num_sublayers > 0:
            sample_names = list(sublayers.keys())[:3]
            for name in sample_names:
                print(f"         ├─ {name}")
            if num_sublayers > 3:
                print(f"         └─ ... and {num_sublayers - 3} more")

    print("="*80)
    print(f"\nTo prune a specific layer, use: --layer_idx <0-{num_layers-1}>")
    print("="*80 + "\n")


def capture_layer_activations(layer, inputs, attention_mask, position_ids, nsamples):
    """
    Run inputs through a layer and capture output activations (WX).

    Args:
        layer: Transformer layer
        inputs: Input activations (nsamples, seqlen, hidden_size)
        attention_mask: Attention mask
        position_ids: Position IDs
        nsamples: Number of samples

    Returns:
        Tensor of stacked activation outputs (nsamples, seqlen, hidden_size)
    """
    # Pre-allocate output buffer (memory efficient)
    outputs = torch.zeros_like(inputs)

    with torch.no_grad():
        for j in range(nsamples):
            layer_kwargs = {
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }

            # Forward pass through the layer - write directly to buffer
            outputs[j] = layer(inputs[j].unsqueeze(0), **layer_kwargs)[0]

    return outputs


def prune_single_layer(pruner, layer_idx, sparsity_ratio, sparsity_pattern,
                       dataloader, nsamples, device, save_visualization=False):
    """
    Prune a single layer using existing BasePruner infrastructure.

    Args:
        pruner: WandaPruner instance
        layer_idx: Index of layer to prune
        sparsity_ratio: Target sparsity
        sparsity_pattern: SparsityPattern instance
        dataloader: Calibration data loader
        nsamples: Number of calibration samples
        device: Device for computation
        save_visualization: Whether to save weight distribution visualizations

    Returns:
        Tuple of (inputs, outputs, attention_mask, position_ids, stats)
    """
    model = pruner.model

    # Save cache setting
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Prepare calibration inputs (intercept first layer)
    print("Preparing calibration inputs...")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    # For layers before target layer, just forward pass (no pruning)
    layers = model.model.layers
    for i in range(layer_idx):
        print(f"Forwarding through layer {i} (no pruning)...")
        layer = layers[i]

        # Handle multi-GPU
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)

        # Forward pass
        for j in range(nsamples):
            with torch.no_grad():
                layer_kwargs = {
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Swap buffers
        inps, outs = outs, inps

    # Now prune the target layer using BasePruner's _prune_layer method
    print(f"\nPruning target layer {layer_idx}...")
    layer = layers[layer_idx]

    # Save weights BEFORE pruning for visualization (if enabled)
    subset = find_layers(layer)
    if save_visualization:
        weights_before = {}
        for name in subset:
            weights_before[name] = subset[name].weight.data.clone()

        # Visualize BEFORE pruning
        print(f"\n[Visualization] Layer {layer_idx} BEFORE pruning:")
        visualize_layer_value_distribution(
            weights_before,
            layer_name=f"layer_{layer_idx}",
            save_dir="../../assets/layer_analysis",
            suffix="before_pruning",
            layer_idx=layer_idx
        )

    # Use BasePruner's _prune_layer method (reuse existing logic!)
    inps, outs = pruner._prune_layer(
        layer_idx, layer, inps, outs,
        attention_mask, position_ids,
        sparsity_ratio, sparsity_pattern,
        nsamples
    )

    # Save weights AFTER pruning for visualization (if enabled)
    if save_visualization:
        weights_after = {}
        for name in subset:
            weights_after[name] = subset[name].weight.data.clone()

        # Visualize AFTER pruning
        print(f"\n[Visualization] Layer {layer_idx} AFTER pruning:")
        visualize_layer_value_distribution(
            weights_after,
            layer_name=f"layer_{layer_idx}",
            save_dir="../../assets/layer_analysis",
            suffix="after_pruning",
            layer_idx=layer_idx
        )

    # Restore cache setting
    model.config.use_cache = use_cache

    # Return the inputs that were used (outputs become inputs after swap)
    return inps, outs, attention_mask, position_ids


def analyze_activation_differences(pre_activations, post_activations, nsamples):
    """
    Analyze differences between pre- and post-pruning activations.

    Args:
        pre_activations: Pre-pruning activation tensor (nsamples, seqlen, hidden_size)
        post_activations: Post-pruning activation tensor (nsamples, seqlen, hidden_size)
        nsamples: Number of samples

    Returns:
        Tuple of (stats_dict, all_diffs_tensor)
    """
    print("\nAnalyzing activation differences...")

    # Debug: Check for NaN or Inf in input tensors
    if torch.isnan(pre_activations).any():
        print("Warning: pre_activations contains NaN values")
    if torch.isnan(post_activations).any():
        print("Warning: post_activations contains NaN values")
    if torch.isinf(pre_activations).any():
        print("Warning: pre_activations contains Inf values")
    if torch.isinf(post_activations).any():
        print("Warning: post_activations contains Inf values")

    # Compute differences directly on tensors (memory efficient)
    all_diffs = torch.abs(pre_activations - post_activations)

    # Compute per-sample L2 norms
    per_sample_l2 = []
    for j in range(nsamples):
        sample_norm = torch.norm(all_diffs[j]).item()
        per_sample_l2.append(sample_norm)

    # Compute statistics
    stats = {
        'mean_abs_diff': all_diffs.mean().item(),
        'max_abs_diff': all_diffs.max().item(),
        'std_abs_diff': all_diffs.std().item(),
        'l2_norm_diff': torch.norm(all_diffs).item(),
        'per_sample_l2': per_sample_l2,
    }

    return stats, all_diffs


def plot_activation_analysis(stats, all_diffs, save_path=None):
    """
    Create visualization of activation differences.

    Args:
        stats: Statistics dictionary
        all_diffs: Tensor of absolute differences
        save_path: Path to save plot (if None, plot is not created)
    """
    if not save_path:
        print("\nSkipping activation difference plot (no save path specified)")
        return

    print("\nCreating activation difference visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Histogram of absolute differences
    ax = axes[0]
    diffs_flat = all_diffs.flatten().cpu().numpy()
    ax.hist(diffs_flat, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Difference')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Absolute Activation Differences')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Per-sample L2 norm
    ax = axes[1]
    sample_indices = list(range(len(stats['per_sample_l2'])))
    ax.bar(sample_indices, stats['per_sample_l2'], alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('L2 Norm of Difference')
    ax.set_title('Per-Sample L2 Norm of Activation Differences')
    ax.grid(True, alpha=0.3)

    # 3. Statistics summary (text)
    ax = axes[2]
    ax.axis('off')

    summary_text = f"""
Activation Difference Statistics:

Mean Absolute Diff:    {stats['mean_abs_diff']:.6f}
Max Absolute Diff:     {stats['max_abs_diff']:.6f}
Std Absolute Diff:     {stats['std_abs_diff']:.6f}

Total L2 Norm Diff:    {stats['l2_norm_diff']:.6f}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved activation difference plot to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Prune a single layer and analyze activation differences"
    )

    # Model configuration
    parser.add_argument(
        '--model', type=str, default='openbmb/MiniCPM-2B-sft-bf16',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--cache_dir', type=str, default='../../llm_weights',
        help='Directory to cache model weights'
    )
    parser.add_argument(
        '--seqlen', type=int, default=2048,
        help='Sequence length'
    )

    # Layer selection
    parser.add_argument(
        '--layer_idx', type=int, default=0,
        help='Index of layer to prune (0-indexed)'
    )

    # Pruning configuration
    parser.add_argument(
        '--sparsity_ratio', type=float, default=0.5,
        help='Target sparsity ratio (0.0-1.0)'
    )
    parser.add_argument(
        '--sparsity_type', type=str, default='unstructured',
        choices=['unstructured', '2:4', '4:8'],
        help='Sparsity pattern type'
    )

    # Calibration
    parser.add_argument(
        '--dataset', type=str, default='wikitext2',
        choices=['wikitext2', 'c4'],
        help='Calibration dataset'
    )
    parser.add_argument(
        '--nsamples', type=int, default=10,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )

    # Weight Redistribution
    parser.add_argument(
        '--use_recovery', action='store_true',
        help='Apply weight redistribution after pruning'
    )
    parser.add_argument(
        '--recovery_strategy', type=str, default='inverse_wanda',
        choices=['inverse_wanda', 'proportional_buffer', 'uniform', 'magnitude_weighted'],
        help='Strategy for redistributing lost signal'
    )
    parser.add_argument(
        '--protected_tier_pct', type=float, default=0.2,
        help='Percentage of surviving weights to protect (proportional_buffer only)'
    )
    parser.add_argument(
        '--inverse_wanda_update_fraction', type=float, default=0.3,
        help='Fraction of survivors to update for inverse_wanda (0.0-1.0, default: 0.3)'
    )
    parser.add_argument(
        '--inverse_wanda_max_relative_update', type=float, default=2.0,
        help='Max update relative to weight magnitude for inverse_wanda (default: 2.0)'
    )

    # Output
    parser.add_argument(
        '--save_plot', type=str, default=None,
        help='Path to save analysis plot (optional)'
    )
    parser.add_argument(
        '--save_visualization', action='store_true',
        help='Save weight distribution visualizations before and after pruning'
    )

    # Utility options
    parser.add_argument(
        '--print_layers', action='store_true',
        help='Print all model layers and their indices, then exit'
    )

    args = parser.parse_args()

    # Print configuration
    print("="*80)
    print("Single Layer Pruning and Activation Analysis")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layer index: {args.layer_idx}")
    print(f"Sparsity: {args.sparsity_ratio:.1%} ({args.sparsity_type})")
    print(f"Calibration: {args.dataset} ({args.nsamples} samples)")
    print("="*80)

    # Load model and tokenizer
    print("\n[1/6] Loading model and tokenizer...")
    model = load_model(args.model, args.cache_dir, args.seqlen)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If --print_layers flag is set, print layer structure and exit
    if args.print_layers:
        print_model_layers(model)
        return

    # Check layer index
    num_layers = len(model.model.layers)
    if args.layer_idx >= num_layers:
        print(f"Error: layer_idx={args.layer_idx} but model only has {num_layers} layers")
        print(f"\nUse --print_layers to see all available layers")
        return

    # Select sparsity pattern
    print(f"\n[2/6] Initializing {args.sparsity_type} sparsity pattern...")
    if args.sparsity_type == 'unstructured':
        sparsity_pattern = UnstructuredSparsity()
    elif args.sparsity_type == '2:4':
        if abs(args.sparsity_ratio - 0.5) > 0.01:
            print(f"Warning: 2:4 sparsity requires sparsity_ratio=0.5, setting to 0.5.")
            args.sparsity_ratio = 0.5
        sparsity_pattern = NMSparsity(n=2, m=4)
    elif args.sparsity_type == '4:8':
        if abs(args.sparsity_ratio - 0.5) > 0.01:
            print(f"Warning: 4:8 sparsity requires sparsity_ratio=0.5, setting to 0.5.")
            args.sparsity_ratio = 0.5
        sparsity_pattern = NMSparsity(n=4, m=8)

    # Load calibration data
    print(f"\n[3/6] Loading calibration data...")
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, tokenizer=tokenizer
    )

    # Create redistributor if enabled
    redistributor = None
    if args.use_recovery:
        if args.recovery_strategy == 'inverse_wanda':
            strategy = InverseWandaStrategy(
                update_fraction=args.inverse_wanda_update_fraction,
                max_relative_update=args.inverse_wanda_max_relative_update
            )
        elif args.recovery_strategy == 'proportional_buffer':
            strategy = ProportionalBufferStrategy(args.protected_tier_pct)
        elif args.recovery_strategy == 'uniform':
            strategy = UniformStrategy()
        elif args.recovery_strategy == 'magnitude_weighted':
            strategy = MagnitudeWeightedStrategy()
        else:
            raise ValueError(f"Unknown recovery strategy: {args.recovery_strategy}")
        redistributor = WeightRedistributor(strategy)

    # Create WandaPruner (reuses existing infrastructure)
    print(f"\n[4/6] Creating WandaPruner...")
    if args.use_recovery:
        print(f"  Using recovery strategy: {args.recovery_strategy}")
    pruner = WandaPruner(model, tokenizer, device, redistributor)

    # Capture PRE-pruning activations
    print(f"\n[5/6] Capturing PRE-pruning activations...")

    # Prepare inputs for the target layer
    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    # Forward through layers before target layer
    layers = model.model.layers
    for i in range(args.layer_idx):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)

        for j in range(args.nsamples):
            with torch.no_grad():
                layer_kwargs = {
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

    # Capture pre-pruning activations at target layer
    target_layer = layers[args.layer_idx]
    if f"model.layers.{args.layer_idx}" in model.hf_device_map:
        dev = model.hf_device_map[f"model.layers.{args.layer_idx}"]
        inps = inps.to(dev)
        attention_mask = attention_mask.to(dev)
        position_ids = position_ids.to(dev)

    pre_activations = capture_layer_activations(
        target_layer, inps, attention_mask, position_ids, args.nsamples
    )
    print(f"Captured pre-pruning activation tensor: {pre_activations.shape}")

    # Move pre_activations to CPU to free GPU memory
    pre_activations = pre_activations.cpu()
    print("Moved pre-pruning activations to CPU")

    # Clean up pre-pruning forward pass
    del inps, outs, target_layer, layers
    del pruner  # Delete pruner which holds reference to model
    torch.cuda.empty_cache()

    # Reload model for pruning (to start fresh and avoid any gradient accumulation)
    print(f"\n[6/6] Pruning layer {args.layer_idx} and capturing POST-pruning activations...")
    del model  # Delete old model first
    torch.cuda.empty_cache()

    model = load_model(args.model, args.cache_dir, args.seqlen)
    # Recreate redistributor if needed
    redistributor_post = None
    if args.use_recovery:
        if args.recovery_strategy == 'inverse_wanda':
            strategy = InverseWandaStrategy(
                update_fraction=args.inverse_wanda_update_fraction,
                max_relative_update=args.inverse_wanda_max_relative_update
            )
        elif args.recovery_strategy == 'proportional_buffer':
            strategy = ProportionalBufferStrategy(args.protected_tier_pct)
        elif args.recovery_strategy == 'uniform':
            strategy = UniformStrategy()
        elif args.recovery_strategy == 'magnitude_weighted':
            strategy = MagnitudeWeightedStrategy()
        else:
            raise ValueError(f"Unknown recovery strategy: {args.recovery_strategy}")
        redistributor_post = WeightRedistributor(strategy)

    pruner = WandaPruner(model, tokenizer, device, redistributor_post)

    # Prune and get post-pruning inputs
    inps_post, _, attention_mask_post, position_ids_post = prune_single_layer(
        pruner, args.layer_idx, args.sparsity_ratio,
        sparsity_pattern, dataloader, args.nsamples, device,
        save_visualization=args.save_visualization
    )

    # Capture post-pruning activations
    target_layer_post = model.model.layers[args.layer_idx]
    post_activations = capture_layer_activations(
        target_layer_post, inps_post, attention_mask_post, position_ids_post, args.nsamples
    )
    print(f"Captured post-pruning activation tensor: {post_activations.shape}")

    # Move post_activations to CPU for analysis (free GPU memory)
    post_activations = post_activations.cpu()
    print("Moved post-pruning activations to CPU")

    model.config.use_cache = use_cache

    # Display recovery statistics if recovery was applied
    if args.use_recovery and pruner.recovery_stats:
        print("\n" + "="*80)
        print("RECOVERY STATISTICS")
        print("="*80)
        recovery_summary = pruner.get_recovery_summary()
        if recovery_summary:
            print(f"Strategy: {args.recovery_strategy}")
            print(f"Mean activation error:   {recovery_summary.get('mean_relative_error', 0):.6f}")
            print(f"Max activation error:    {recovery_summary.get('max_relative_error', 0):.6f}")
            print(f"Weights updated:         {recovery_summary.get('total_weights_updated', 0):,}")
            print(f"Mean max update:         {recovery_summary.get('mean_max_update', 0):.6f}")

            # Display new sum statistics
            if 'sum_of_errors' in recovery_summary:
                print(f"\nSum of errors (before):  {recovery_summary.get('sum_of_errors', 0):.6f}")
            if 'sum_of_errors_after' in recovery_summary:
                print(f"Sum of errors (after):   {recovery_summary.get('sum_of_errors_after', 0):.6f}")
            if 'sum_of_updated_error' in recovery_summary:
                print(f"Sum of updated error:    {recovery_summary.get('sum_of_updated_error', 0):.6f}")

            # Calculate improvement if both before and after are available
            if 'sum_of_errors' in recovery_summary and 'sum_of_errors_after' in recovery_summary:
                sum_before = recovery_summary['sum_of_errors']
                sum_after = recovery_summary['sum_of_errors_after']
                if sum_before > 0:
                    improvement = sum_before - sum_after
                    improvement_pct = (improvement / sum_before) * 100
                    print(f"\nSum error improvement:   {improvement:.6f} ({improvement_pct:.2f}%)")

            print(f"Layers processed:        {recovery_summary.get('num_layers', 0)}")
        print("="*80)

    # Clean up GPU memory before analysis
    del inps_post, target_layer_post
    torch.cuda.empty_cache()

    # Analyze differences (both tensors now on CPU)
    diff_stats, all_diffs = analyze_activation_differences(
        pre_activations, post_activations, args.nsamples
    )

    # Print analysis
    print("\n" + "="*80)
    print("ACTIVATION DIFFERENCE ANALYSIS")
    print("="*80)
    print(f"Mean Absolute Difference:     {diff_stats['mean_abs_diff']:.6f}")
    print(f"Max Absolute Difference:      {diff_stats['max_abs_diff']:.6f}")
    print(f"Std Absolute Difference:      {diff_stats['std_abs_diff']:.6f}")
    print()
    print(f"Total L2 Norm Difference:     {diff_stats['l2_norm_diff']:.6f}")
    print("="*80)

    # Create visualization
    plot_activation_analysis(diff_stats, all_diffs, args.save_plot)

    print("\nDone!")


if __name__ == "__main__":
    main()
