"""
Weight redistribution for sparse neural networks.

Flexible framework for experimenting with different weight redistribution strategies
after pruning. Uses strategy pattern to support multiple hypotheses.
"""

import torch
from abc import ABC, abstractmethod


# ============================================================================
# Abstract Base: Redistribution Strategy
# ============================================================================

class RedistributionStrategy(ABC):
    """
    Abstract strategy for computing redistribution coefficients.

    Subclasses implement different hypotheses for how to redistribute
    lost signal from pruned weights to surviving weights.
    """

    @abstractmethod
    def compute_coefficients(self, W_sparse, prune_mask, mean_activations, scaler_row=None):
        """
        Compute redistribution coefficient matrix C.

        Args:
            W_sparse: Sparse weight matrix (after pruning), shape (out_features, in_features)
            prune_mask: Boolean mask (True = pruned), same shape as W_sparse
            mean_activations: Mean activations E[x], shape (in_features,)
            scaler_row: L2-norm squared per input feature (in_features,)
                       Optional, required for Wanda-based strategies

        Returns:
            C: Coefficient matrix, shape (out_features, in_features)
               - C[i,j] = fraction of lost signal ε_i that weight w_ij should absorb
               - Constraint: sum(C[i,:]) = 1.0 for each row i
               - C[i,j] = 0 for pruned weights
        """
        pass


# ============================================================================
# Inverse Wanda Strategy
# ============================================================================

class InverseWandaStrategy(RedistributionStrategy):
    """
    Redistribute lost signal inversely proportional to Wanda scores.

    Key principle: Weights with LOWER Wanda scores receive MORE update.
    This is based on the hypothesis that low-Wanda weights are "cheaper" to modify
    without significantly impacting model behavior.

    Algorithm:
    1. Compute Wanda scores for all surviving weights: |W| × sqrt(||activation||²)
    2. For each output neuron, select top-k survivors with LOWEST Wanda scores
    3. Compute inverse coefficients: 1 / (wanda_score + epsilon)
    4. Normalize to sum to 1.0 per row

    Args:
        update_fraction: Fraction of surviving weights to update (0.0-1.0)
                        Default: 0.3 (30% of survivors receive updates)
        max_relative_update: Maximum update relative to weight magnitude
                           Default: 2.0 (update can be at most 2x weight value)
        min_wanda_epsilon: Small constant to prevent division by zero
                         Default: 1e-8
    """

    def __init__(self, update_fraction=0.3, max_relative_update=2.0, min_wanda_epsilon=1e-8):
        """
        Initialize inverse Wanda redistribution strategy.

        Args:
            update_fraction: Fraction of survivors to update (0.3 = 30%)
            max_relative_update: Cap update at this multiple of weight magnitude
            min_wanda_epsilon: Epsilon for numerical stability in division
        """
        # Validate parameters
        if not 0.0 < update_fraction <= 1.0:
            raise ValueError(f"update_fraction must be in (0, 1], got {update_fraction}")
        if max_relative_update <= 0:
            raise ValueError(f"max_relative_update must be positive, got {max_relative_update}")

        self.update_fraction = update_fraction
        self.max_relative_update = max_relative_update
        self.min_wanda_epsilon = min_wanda_epsilon

    def compute_coefficients(self, W_sparse, prune_mask, mean_activations, scaler_row=None):
        """
        Compute redistribution coefficients inversely proportional to Wanda scores.

        Args:
            W_sparse: Sparse weight matrix (out_features, in_features)
            prune_mask: Boolean mask (True = pruned)
            mean_activations: Mean activations E[x] (in_features,)
            scaler_row: L2-norm squared of activations (in_features,)
                       REQUIRED for InverseWandaStrategy

        Returns:
            C: Coefficient matrix (out_features, in_features)
               - C[i,j] = fraction of error ε_i that w_ij should absorb
               - Constraint: sum(C[i,:]) = 1.0 for each row
        """
        # Validate scaler_row is provided
        if scaler_row is None:
            raise ValueError(
                "InverseWandaStrategy requires scaler_row (activation norms). "
                "Ensure BasePruner passes activation_captures[name].get_scaler()."
            )

        device = W_sparse.device
        dtype = W_sparse.dtype

        # Ensure scaler_row matches device and dtype
        scaler_row = scaler_row.to(device=device, dtype=dtype)

        # 1. Compute Wanda scores for all weights
        # Same formula as pruning: |W| × sqrt(||activation||^2)
        wanda_scores = torch.abs(W_sparse) * torch.sqrt(scaler_row.reshape(1, -1))

        # 2. Create survivor mask
        survivor_mask = ~prune_mask

        # 3. Initialize coefficient matrix (match W_sparse dtype)
        C = torch.zeros_like(W_sparse, dtype=dtype)

        # 4. Process each output neuron independently
        for i in range(W_sparse.shape[0]):
            row_coeffs = self._compute_row_coefficients(
                W_sparse[i, :],
                wanda_scores[i, :],
                survivor_mask[i, :]
            )
            C[i, :] = row_coeffs

        return C

    def _compute_row_coefficients(self, W_row, wanda_row, survivor_mask_row):
        """
        Compute redistribution coefficients for a single output neuron.

        Args:
            W_row: Weight row (in_features,)
            wanda_row: Wanda scores for this row (in_features,)
            survivor_mask_row: Boolean mask of survivors (in_features,)

        Returns:
            Coefficient vector (in_features,) that sums to 1.0
        """
        dtype = W_row.dtype

        # Get survivor indices
        survivor_indices = torch.where(survivor_mask_row)[0]

        # Edge case: no survivors
        if survivor_indices.numel() == 0:
            return torch.zeros_like(W_row, dtype=dtype)

        # Edge case: only one survivor (gets all the update)
        if survivor_indices.numel() == 1:
            coeffs = torch.zeros_like(W_row, dtype=dtype)
            coeffs[survivor_indices[0]] = 1.0
            return coeffs

        # Extract survivor Wanda scores
        survivor_wanda = wanda_row[survivor_indices]

        # Select top-k weights with LOWEST Wanda scores
        num_survivors = survivor_indices.numel()
        num_to_update = max(1, int(num_survivors * self.update_fraction))

        # Sort by Wanda score (ascending) and take first num_to_update
        sorted_indices = torch.argsort(survivor_wanda)
        update_indices_local = sorted_indices[:num_to_update]
        update_indices_global = survivor_indices[update_indices_local]

        # Compute inverse Wanda coefficients
        selected_wanda = survivor_wanda[update_indices_local]

        # Inverse proportionality: 1 / (wanda + epsilon)
        inverse_coeffs = 1.0 / (selected_wanda + self.min_wanda_epsilon)

        # Normalize to sum to 1.0
        normalized_coeffs = inverse_coeffs / inverse_coeffs.sum()

        # Assign to output vector
        coeffs = torch.zeros_like(W_row, dtype=dtype)
        coeffs[update_indices_global] = normalized_coeffs

        return coeffs


# ============================================================================
# Weight Redistributor
# ============================================================================

class WeightRedistributor:
    """
    Applies weight redistribution after pruning.

    Uses a RedistributionStrategy to compute coefficients, then updates
    surviving weights to compensate for pruned weights.

    Algorithm:
    1. Compute lost signal per output neuron due to pruning
    2. Get redistribution coefficients from strategy
    3. Compute weight updates based on coefficients and lost signal
    4. Apply update cap to prevent instability
    5. Update weights in-place
    6. Verify pruned weights remain zero
    """

    def __init__(self, strategy):
        """
        Initialize redistributor.

        Args:
            strategy: RedistributionStrategy instance
        """
        self.strategy = strategy

    def apply(self, W_dense, layer, prune_mask, mean_activations, scaler_row=None):
        """
        Apply weight redistribution to a pruned layer.

        Args:
            W_dense: Original weights before pruning (out_features, in_features)
            layer: Layer with pruned weights (modified in-place)
            prune_mask: Boolean mask (True = pruned)
            mean_activations: Mean activations E[x] (in_features,)
            scaler_row: L2-norm squared of activations (in_features,)

        Returns:
            Dictionary of recovery statistics
        """
        # Get sparse weights
        W_sparse = layer.weight.data
        device = W_sparse.device
        dtype = W_sparse.dtype

        # Ensure all tensors are on same device and dtype
        mean_activations = mean_activations.to(device=device, dtype=dtype)
        W_dense = W_dense.to(device=device, dtype=dtype)
        if scaler_row is not None:
            scaler_row = scaler_row.to(device=device, dtype=dtype)

        # 1. Compute lost signal per output neuron
        # ε_i = (W_dense[i, :] - W_sparse[i, :]) @ E[x]
        delta_W = W_dense - W_sparse  # Difference due to pruning
        lost_signal = torch.matmul(delta_W, mean_activations)  # (out_features,)

        # 2. Compute redistribution coefficients
        C = self.strategy.compute_coefficients(
            W_sparse, prune_mask, mean_activations, scaler_row
        )

        # 3. Compute weight updates
        # For each weight w_ij: Δw_ij = C[i,j] * ε_i
        # where ε_i is the lost signal already computed
        # Broadcasting: lost_signal (out_features,) → (out_features, 1)
        delta_weights = C * lost_signal.reshape(-1, 1)

        # 4. Apply update cap (if strategy has this attribute)
        if hasattr(self.strategy, 'max_relative_update'):
            max_update = self.strategy.max_relative_update
            # Cap: |Δw_ij| ≤ max_relative_update * |w_ij|
            cap = max_update * torch.abs(W_sparse)
            # For zero weights, use a small default cap
            cap = torch.where(cap > 0, cap, torch.ones_like(cap) * 0.01)
            delta_weights = torch.clamp(delta_weights, -cap, cap)

        # 5. Apply updates to weights (in-place)
        layer.weight.data += delta_weights

        # 6. Verify: pruned weights should still be zero
        layer.weight.data[prune_mask] = 0

        # 7. Compute recovery statistics
        W_recovered = layer.weight.data
        recovered_signal = torch.matmul(W_recovered, mean_activations)
        original_signal = torch.matmul(W_dense, mean_activations)

        relative_error = (
            torch.norm(recovered_signal - original_signal) /
            (torch.norm(original_signal) + 1e-8)
        ).item()

        stats = {
            'relative_error': relative_error,
            'total_lost_signal': torch.norm(lost_signal).item(),
            'num_weights_updated': (C != 0).sum().item(),
            'max_update_magnitude': torch.abs(delta_weights).max().item(),
        }

        return stats
