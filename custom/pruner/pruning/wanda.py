"""
Wanda (Pruning by Weights AND Activations) implementation.

Reference: wanda/lib/prune.py:127-222
Paper: https://arxiv.org/abs/2306.11695
"""

import torch
import sys
from pathlib import Path

# Ensure pruning module can import base_pruner
PRUNER_DIR = Path(__file__).parent.parent
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))

from pruning.base_pruner import BasePruner


class WandaPruner(BasePruner):
    """
    Wanda pruning: prune weights based on |weight| × sqrt(||activation||^2).

    Key insight: A weight's importance depends on both:
    1. Its magnitude (large weights have more impact on output)
    2. Activation norm (weights on frequently-activated features are more important)

    Combining these gives a more accurate importance metric than magnitude alone.

    The Wanda metric for each weight w_ij is:
        importance(w_ij) = |w_ij| × sqrt(||activation_j||^2)

    where:
    - |w_ij| is the absolute value of the weight
    - ||activation_j||^2 is the L2-norm squared of activations for input feature j

    Reference: wanda/lib/prune.py:172
    """

    def compute_pruning_metric(self, layer, activation_stats):
        """
        Compute Wanda importance metric.

        Args:
            layer: Layer to compute metric for
            activation_stats: ActivationCapture instance with accumulated statistics

        Returns:
            Tensor of importance scores (higher = more important)
            Shape: (out_features, in_features)
        """
        # Get weight matrix
        weight = layer.weight.data

        # Get activation statistics: L2-norm squared per input feature
        # Shape: (in_features,)
        scaler_row = activation_stats.scaler_row

        # Compute Wanda metric: |W| × sqrt(||activation||^2)
        # Broadcasting: weight is (out_features, in_features)
        #               scaler_row.reshape(1, -1) is (1, in_features)
        # Result: (out_features, in_features)
        metric = torch.abs(weight) * torch.sqrt(scaler_row.reshape(1, -1))

        return metric
