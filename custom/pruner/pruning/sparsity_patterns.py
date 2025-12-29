"""
Sparsity patterns for pruning.

Implements unstructured and N:M structured sparsity patterns.
Reference: wanda/lib/prune.py:175-206
"""

import torch
from abc import ABC, abstractmethod


class SparsityPattern(ABC):
    """
    Abstract base class for sparsity patterns.

    A sparsity pattern defines HOW weights should be pruned (which weights form a group).
    The pruning metric defines WHICH weights within that group should be pruned.
    """

    @abstractmethod
    def create_mask(self, metric, sparsity_ratio):
        """
        Create a binary pruning mask based on the importance metric.

        Args:
            metric: Tensor of importance scores (higher = more important)
                    Shape: (out_features, in_features)
            sparsity_ratio: Fraction of weights to prune (0.0-1.0)

        Returns:
            Boolean mask where True = prune this weight
            Shape: same as metric
        """
        pass


class UnstructuredSparsity(SparsityPattern):
    """
    Unstructured sparsity: prune any weights independently.

    Prunes the lowest-importance weights globally across the entire weight matrix.

    Reference: wanda/lib/prune.py:204-206
    """

    def create_mask(self, metric, sparsity_ratio):
        """
        Create unstructured pruning mask.

        Args:
            metric: Importance scores (out_features, in_features)
            sparsity_ratio: Fraction of weights to prune

        Returns:
            Boolean mask (True = prune)
        """
        # Initialize mask (all False = keep all weights)
        W_mask = torch.zeros_like(metric, dtype=torch.bool)

        # Sort weights by importance (stable sort preserves order of equal elements)
        sort_res = torch.sort(metric, dim=-1, stable=True)

        # Get indices of smallest sparsity_ratio fraction of weights
        num_to_prune = int(metric.shape[1] * sparsity_ratio)
        indices = sort_res[1][:, :num_to_prune]

        # Mark these indices for pruning
        W_mask.scatter_(1, indices, True)

        return W_mask


class NMSparsity(SparsityPattern):
    """
    N:M structured sparsity: prune N out of every M consecutive weights.

    This pattern is hardware-friendly and supported by modern GPUs (e.g., NVIDIA A100).
    Common patterns:
    - 2:4 = keep 2, prune 2 out of every 4 weights (50% sparsity)
    - 4:8 = keep 4, prune 4 out of every 8 weights (50% sparsity)

    Note: For N:M sparsity, sparsity_ratio must be 0.5 (enforced by caller)

    Reference: wanda/lib/prune.py:175-180
    """

    def __init__(self, n, m):
        """
        Initialize N:M sparsity pattern.

        Args:
            n: Number of weights to prune in each group
            m: Group size
        """
        self.n = n
        self.m = m

        # Validate: n must be <= m
        if n > m:
            raise ValueError(f"Invalid N:M pattern: n={n} must be <= m={m}")

        # For N:M sparsity, sparsity ratio should be n/m
        self.expected_sparsity = n / m

    def create_mask(self, metric, sparsity_ratio):
        """
        Create N:M structured pruning mask.

        Processes the weight matrix in m-width blocks along the column dimension.
        Within each block, finds the n smallest weights and marks them for pruning.

        Args:
            metric: Importance scores (out_features, in_features)
            sparsity_ratio: Fraction of weights to prune (should be n/m)

        Returns:
            Boolean mask (True = prune)
        """
        # Validate sparsity ratio matches N:M pattern
        if abs(sparsity_ratio - self.expected_sparsity) > 0.01:
            print(f"Warning: sparsity_ratio={sparsity_ratio:.2f} doesn't match "
                  f"{self.n}:{self.m} pattern (expected {self.expected_sparsity:.2f})")

        # Initialize mask (all False = keep all weights)
        W_mask = torch.zeros_like(metric, dtype=torch.bool)

        # Process m-width blocks along columns
        for col_idx in range(metric.shape[1]):
            # Only process at the start of each m-width block
            if col_idx % self.m == 0:
                # Extract m-width block
                # Shape: (out_features, m)
                block_end = min(col_idx + self.m, metric.shape[1])
                block = metric[:, col_idx:block_end].float()

                # Find n smallest weights in each row of the block
                # topk with largest=False gives us the smallest values
                if block.shape[1] >= self.n:
                    smallest_indices = torch.topk(block, self.n, dim=1, largest=False)[1]

                    # Mark these indices for pruning
                    # Add col_idx offset to get global column indices
                    W_mask.scatter_(1, col_idx + smallest_indices, True)

        return W_mask

    def __repr__(self):
        return f"NMSparsity(n={self.n}, m={self.m})"
