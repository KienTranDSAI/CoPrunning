"""
Base pruner class for all pruning methods.

Provides common infrastructure for layer-wise pruning with activation-aware metrics.
Reference: wanda/lib/prune.py:127-222
"""

import torch
from abc import ABC, abstractmethod

# Import from utils modules
import sys
from pathlib import Path
PRUNER_DIR = Path(__file__).parent.parent
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))

from utils.model_utils import find_layers, prepare_calibration_input
from utils.dataset_loader import get_loaders
from utils.activation_capture import ActivationCapture


class BasePruner(ABC):
    """
    Abstract base class for pruning methods.

    Handles the common pruning pipeline:
    1. Load calibration data
    2. Prepare calibration inputs
    3. Prune layer by layer:
       - Capture activation statistics
       - Compute pruning metric (method-specific)
       - Apply sparsity pattern
       - Zero out masked weights

    Subclasses only need to implement compute_pruning_metric().
    """

    def __init__(self, model, tokenizer, device, redistributor=None):
        """
        Initialize pruner.

        Args:
            model: Model to prune
            tokenizer: Associated tokenizer
            device: Device for computation
            redistributor: Optional WeightRedistributor instance for post-pruning weight redistribution
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.redistributor = redistributor
        self.recovery_stats = []  # Store recovery statistics

        # Storage for pruning state (for recovery without re-pruning)
        self.pruning_masks = {}  # Store masks per layer/sublayer
        self.activation_captures = {}  # Store activation stats per layer/sublayer
        self.original_weights = {}  # Store original weights before pruning

    def prune(self, sparsity_ratio, sparsity_pattern, nsamples=128,
              dataset="c4", seed=0):
        """
        Main pruning pipeline.

        Reference: wanda/lib/prune.py:127-222

        Args:
            sparsity_ratio: Fraction of weights to prune (0.0-1.0)
            sparsity_pattern: SparsityPattern instance defining pruning structure
            nsamples: Number of calibration samples
            dataset: Dataset for calibration ('wikitext2' or 'c4')
            seed: Random seed for reproducibility
        """
        print(f"Starting pruning with {sparsity_pattern}")
        print(f"Sparsity ratio: {sparsity_ratio}")
        print(f"Calibration dataset: {dataset}, samples: {nsamples}")

        # Save original cache setting
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        # Load calibration data
        print("Loading calibration data...")
        dataloader, _ = get_loaders(
            dataset, nsamples=nsamples, seed=seed,
            seqlen=self.model.seqlen, tokenizer=self.tokenizer
        )

        # Prepare calibration inputs (intercept first layer)
        print("Preparing calibration inputs...")
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                self.model, dataloader, self.device
            )

        # Prune layer by layer
        layers = self.model.model.layers
        for i in range(len(layers)):
            print(f"\n{'='*60}")
            print(f"Pruning layer {i}/{len(layers)}")
            print(f"{'='*60}")

            inps, outs = self._prune_layer(
                i, layers[i], inps, outs,
                attention_mask, position_ids,
                sparsity_ratio, sparsity_pattern,
                nsamples
            )

        # Restore cache setting
        self.model.config.use_cache = use_cache
        torch.cuda.empty_cache()

        print("\nPruning complete!")

    @abstractmethod
    def compute_pruning_metric(self, layer, activation_stats):
        """
        Compute importance metric for determining which weights to prune.

        This is the key method that differentiates pruning algorithms:
        - Magnitude pruning: metric = |weight|
        - Wanda: metric = |weight| * sqrt(activation_norm)
        - Random: metric = random values

        Args:
            layer: Layer to compute metric for
            activation_stats: ActivationCapture instance with accumulated statistics

        Returns:
            Tensor of importance scores (higher = more important)
            Shape: (out_features, in_features)
        """
        pass

    def get_recovery_summary(self):
        """
        Get summary statistics of weight redistribution recovery.

        Returns:
            dict: Summary with mean/max relative errors, or None if no recovery was applied
        """
        if not self.recovery_stats:
            return None

        relative_errors = [s['relative_error'] for s in self.recovery_stats]
        num_weights = [s['num_weights_updated'] for s in self.recovery_stats]
        max_updates = [s['max_update_magnitude'] for s in self.recovery_stats]

        return {
            'mean_relative_error': sum(relative_errors) / len(relative_errors),
            'max_relative_error': max(relative_errors),
            'total_weights_updated': sum(num_weights),
            'mean_max_update': sum(max_updates) / len(max_updates),
            'num_layers': len(self.recovery_stats)
        }

    def apply_recovery(self, redistributor):
        """
        Apply weight redistribution to an already-pruned model.

        This method uses stored pruning masks and activation statistics from
        a previous pruning run, allowing recovery to be applied without re-pruning.

        Args:
            redistributor: WeightRedistributor instance to apply

        Returns:
            dict: Summary statistics with both before and after recovery stats
        """
        if not self.pruning_masks:
            raise RuntimeError("No pruning state stored. Must run prune() first.")

        print("\nApplying weight redistribution to pruned model...")
        print(f"Layers to recover: {len(self.pruning_masks)}")

        # Clear previous recovery stats
        self.recovery_stats = []
        before_recovery_stats = []

        # Iterate through stored pruning state
        for layer_key in sorted(self.pruning_masks.keys()):
            # Move stored tensors from CPU to GPU
            mask = self.pruning_masks[layer_key].to(self.device)
            activation_stats = self.activation_captures[layer_key]
            mean_activations = activation_stats['mean_activations'].to(self.device)
            scaler_row = activation_stats['scaler_row'].to(self.device)
            W_dense = self.original_weights[layer_key].to(self.device)

            # Parse layer key to get layer reference
            # Format: "layer_{layer_idx}_{sublayer_name}"
            parts = layer_key.split('_', 2)
            layer_idx = int(parts[1])
            sublayer_name = parts[2]

            # Get the actual layer
            layer = self.model.model.layers[layer_idx]
            subset = {sublayer_name: getattr(layer, sublayer_name.split('.')[0])}
            if '.' in sublayer_name:
                for attr in sublayer_name.split('.')[1:]:
                    subset = {sublayer_name: getattr(list(subset.values())[0], attr)}

            sublayer = list(subset.values())[0]

            print(f"  Recovering {layer_key}...")

            # Compute error statistics BEFORE recovery
            W_sparse_before = sublayer.weight.data.clone()
            before_stats = redistributor.compute_error_stats(
                W_dense,
                W_sparse_before,
                mask,
                mean_activations
            )
            print(f"    Before recovery - Relative error: {before_stats['relative_error']:.6f}")

            # Apply recovery
            recovery_stats = redistributor.apply(
                W_dense,
                sublayer,
                mask,
                mean_activations,
                scaler_row
            )

            print(f"    After recovery  - Relative error: {recovery_stats['relative_error']:.6f}")

            # Store both before and after stats
            before_recovery_stats.append({
                'layer': layer_key,
                'relative_error': before_stats['relative_error'],
                'total_lost_signal': before_stats['total_lost_signal'],
            })

            self.recovery_stats.append({
                'layer': layer_key,
                'relative_error': recovery_stats['relative_error'],
                'num_weights_updated': recovery_stats['num_weights_updated'],
                'max_update_magnitude': recovery_stats['max_update_magnitude']
            })

        print(f"Recovery complete!")

        # Compute summary for both before and after
        before_summary = self._compute_error_summary(before_recovery_stats)
        after_summary = self.get_recovery_summary()

        return {
            'before_recovery': before_summary,
            'after_recovery': after_summary
        }

    def _compute_error_summary(self, stats_list):
        """
        Compute summary statistics from a list of error stats.

        Args:
            stats_list: List of dictionaries with 'relative_error' and other fields

        Returns:
            dict: Summary with mean/max relative errors
        """
        if not stats_list:
            return None

        relative_errors = [s['relative_error'] for s in stats_list]

        summary = {
            'mean_relative_error': sum(relative_errors) / len(relative_errors),
            'max_relative_error': max(relative_errors),
            'num_layers': len(stats_list)
        }

        # Add optional fields if present
        if 'total_lost_signal' in stats_list[0]:
            lost_signals = [s['total_lost_signal'] for s in stats_list]
            summary['mean_lost_signal'] = sum(lost_signals) / len(lost_signals)

        return summary

    def _prune_layer(self, layer_idx, layer, inps, outs,
                     attention_mask, position_ids,
                     sparsity_ratio, sparsity_pattern, nsamples):
        """
        Prune a single transformer layer.

        Args:
            layer_idx: Layer index
            layer: The layer to prune
            inps: Input activations (nsamples, seqlen, hidden_size)
            outs: Output buffer (nsamples, seqlen, hidden_size)
            attention_mask: Attention mask from forward pass
            position_ids: Position IDs from forward pass
            sparsity_ratio: Fraction of weights to prune
            sparsity_pattern: SparsityPattern instance
            nsamples: Number of calibration samples

        Returns:
            Tuple of (new_inps, new_outs) for next layer
            Buffers are swapped: outputs become inputs
        """
        # Handle multi-GPU case
        if f"model.layers.{layer_idx}" in self.model.hf_device_map:
            dev = self.model.hf_device_map[f"model.layers.{layer_idx}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)
        else:
            dev = self.device

        # Find all Linear layers in this transformer layer
        subset = find_layers(layer)

        # Wrap layers to capture activations
        activation_captures = {}
        for name in subset:
            activation_captures[name] = ActivationCapture(subset[name])

        # Register forward hooks to collect activation statistics
        def make_hook(name):
            def hook(_, inp, out):
                activation_captures[name].add_batch(inp[0].data, out.data)
            return hook

        handles = []
        for name in activation_captures:
            handles.append(subset[name].register_forward_hook(make_hook(name)))

        # Forward pass through layer to collect activations
        for j in range(nsamples):
            with torch.no_grad():
                # Build kwargs for layer forward pass
                layer_kwargs = {
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }

                # Forward pass: hooks will capture activations
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Remove hooks
        for h in handles:
            h.remove()

        # Prune each sublayer
        for name in subset:
            print(f"  Pruning sublayer: {name}")

            # Compute importance metric (method-specific)
            metric = self.compute_pruning_metric(
                subset[name],
                activation_captures[name]
            )

            # Create pruning mask based on sparsity pattern
            mask = sparsity_pattern.create_mask(metric, sparsity_ratio)

            # BEFORE pruning: Save original weights for lost signal calculation
            W_dense = subset[name].weight.data.clone()

            # Store pruning state for potential recovery later (on CPU to save GPU memory)
            layer_key = f"layer_{layer_idx}_{name}"
            self.pruning_masks[layer_key] = mask.clone().cpu()
            # Store only the needed statistics, not the whole ActivationCapture object
            self.activation_captures[layer_key] = {
                'mean_activations': activation_captures[name].get_mean_activations().cpu(),
                'scaler_row': activation_captures[name].get_scaler().cpu()
            }
            self.original_weights[layer_key] = W_dense.clone().cpu()

            # Apply mask: zero out pruned weights
            subset[name].weight.data[mask] = 0

            # AFTER pruning: Apply weight redistribution if redistributor is enabled
            if self.redistributor is not None:
                recovery_stats = self.redistributor.apply(
                    W_dense,  # Original weights before pruning
                    subset[name],  # Layer (weights modified in-place)
                    mask,  # Pruning mask
                    activation_captures[name].get_mean_activations(),  # E[x]
                    activation_captures[name].get_scaler()  # L2-norm squared for Wanda
                )
                print(f"    Recovery - Relative error: {recovery_stats['relative_error']:.6f}")
                # Store stats for later aggregation
                self.recovery_stats.append({
                    'layer': f"layer_{layer_idx}_{name}",
                    'relative_error': recovery_stats['relative_error'],
                    'num_weights_updated': recovery_stats['num_weights_updated'],
                    'max_update_magnitude': recovery_stats['max_update_magnitude']
                })

            # Report sparsity for this sublayer
            total_weights = mask.numel()
            pruned_weights = mask.sum().item()
            sublayer_sparsity = pruned_weights / total_weights
            print(f"    Pruned {pruned_weights}/{total_weights} weights "
                  f"({sublayer_sparsity:.2%})")

        # Forward pass again with pruned weights to compute outputs for next layer
        for j in range(nsamples):
            with torch.no_grad():
                layer_kwargs = {
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }

                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Swap buffers: outputs of this layer become inputs to next layer
        return outs, inps
