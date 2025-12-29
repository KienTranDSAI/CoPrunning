"""
Base pruner class for all pruning methods.

Provides common infrastructure for layer-wise pruning with activation-aware metrics.
Reference: wanda/lib/prune.py:127-222
"""

import torch
from abc import ABC, abstractmethod
from ..utils.model_utils import find_layers, prepare_calibration_input
from ..utils.dataset_loader import get_loaders
from ..utils.activation_capture import ActivationCapture


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

    def __init__(self, model, tokenizer, device):
        """
        Initialize pruner.

        Args:
            model: Model to prune
            tokenizer: Associated tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

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
                layer_kwargs = {}
                if attention_mask is not None:
                    layer_kwargs['attention_mask'] = attention_mask
                if position_ids is not None:
                    layer_kwargs['position_ids'] = position_ids

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

            # Apply mask: zero out pruned weights
            subset[name].weight.data[mask] = 0

            # Report sparsity for this sublayer
            total_weights = mask.numel()
            pruned_weights = mask.sum().item()
            sublayer_sparsity = pruned_weights / total_weights
            print(f"    Pruned {pruned_weights}/{total_weights} weights "
                  f"({sublayer_sparsity:.2%})")

        # Forward pass again with pruned weights to compute outputs for next layer
        for j in range(nsamples):
            with torch.no_grad():
                layer_kwargs = {}
                if attention_mask is not None:
                    layer_kwargs['attention_mask'] = attention_mask
                if position_ids is not None:
                    layer_kwargs['position_ids'] = position_ids

                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Swap buffers: outputs of this layer become inputs to next layer
        return outs, inps
