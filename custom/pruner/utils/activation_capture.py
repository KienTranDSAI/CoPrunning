"""
Activation statistics collection for pruning.

Standalone reimplementation of wanda/lib/layerwrapper.py
"""

import torch
import torch.nn as nn


class ActivationCapture:
    """
    Captures and accumulates activation statistics for a layer.

    Used by Wanda pruning to compute activation-aware importance metrics.
    This is a standalone reimplementation of WrappedGPT from wanda/lib/layerwrapper.py:5-35

    The key insight: we track the L2-norm of activations per input feature.
    Higher activation norms indicate more important features that should be preserved.
    """

    def __init__(self, layer):
        """
        Initialize activation capture for a layer.

        Args:
            layer: PyTorch layer to capture activations from (typically nn.Linear)
        """
        self.layer = layer
        self.device = layer.weight.device
        self.rows = layer.weight.data.shape[0]  # Output features
        self.columns = layer.weight.data.shape[1]  # Input features

        # scaler_row: L2-norm squared of activations per input feature
        # Shape: (columns,) = (input_features,)
        self.scaler_row = torch.zeros((self.columns), device=self.device)

        # mean_activations: Mean activation value per input feature
        # Used for weight redistribution after pruning
        # Shape: (columns,) = (input_features,)
        self.mean_activations = torch.zeros((self.columns), device=self.device)

        self.nsamples = 0

    def add_batch(self, inp, out):
        """
        Accumulate activation statistics from a batch.

        This method is called via a forward hook during calibration passes.

        Reference: wanda/lib/layerwrapper.py:22-35

        Args:
            inp: Input tensor to the layer
                 Shape: (batch, seq_len, input_dim) or (batch*seq_len, input_dim)
            out: Output tensor from the layer (not used, but part of hook signature)
        """
        # Handle 2D input: add batch dimension
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        batch_size = inp.shape[0]

        # For Linear layers: reshape to 2D if needed
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                # Flatten batch and sequence dimensions
                # (batch, seq_len, input_dim) -> (batch*seq_len, input_dim)
                inp = inp.reshape(-1, inp.shape[-1])

            # Transpose to (input_dim, batch*seq_len)
            inp = inp.t()

        # Running average update (numerically stable)
        # Update scaler_row proportionally to preserve cumulative average
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        # Accumulate L2-norm squared per input feature
        # Convert to float32 for numerical stability
        inp = inp.type(torch.float32)

        # torch.norm(inp, p=2, dim=1) computes L2-norm across the batch dimension
        # Shape: (input_dim,)
        # We square it to get ||activation||^2 per feature
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

        # Accumulate mean activation per input feature
        # inp is (input_dim, batch*seq_len), so mean across dim=1 gives mean per feature
        batch_mean = torch.mean(inp, dim=1)
        # Running average update
        self.mean_activations = (
            (self.mean_activations * (self.nsamples - batch_size) + batch_mean * batch_size) /
            self.nsamples
        )

    def get_scaler(self):
        """
        Get the accumulated activation statistics.

        Returns:
            Tensor of shape (columns,) containing L2-norm squared per input feature
        """
        return self.scaler_row

    def get_mean_activations(self):
        """
        Get the accumulated mean activations.

        Returns:
            Tensor of shape (columns,) containing mean activation per input feature
        """
        return self.mean_activations

    def reset(self):
        """Reset accumulated statistics."""
        self.scaler_row.zero_()
        self.mean_activations.zero_()
        self.nsamples = 0
