# Inverse Wanda Weight Redistribution - Usage Example

## Quick Start

```python
from custom.pruner.recovery import InverseWandaStrategy, WeightRedistributor
from custom.pruner.pruning import WandaPruner

# 1. Create the inverse Wanda strategy
strategy = InverseWandaStrategy(
    update_fraction=0.3,        # Update 30% of survivors with lowest Wanda scores
    max_relative_update=2.0     # Cap updates at 2x weight magnitude
)

# 2. Create the redistributor
redistributor = WeightRedistributor(strategy)

# 3. Create pruner with redistribution enabled
pruner = WandaPruner(
    model=model,
    tokenizer=tokenizer,
    device='cuda',
    redistributor=redistributor  # Enable weight redistribution
)

# 4. Run pruning
pruner.prune(
    dataloader=calibration_dataloader,
    sparsity_ratio=0.5,
    sparsity_type='unstructured'
)
```

## Parameter Guide

### update_fraction (default: 0.3)
- **Range**: 0.0 to 1.0
- **Meaning**: Fraction of surviving weights that receive redistribution updates
- **Example**: 0.3 means update the 30% of survivors with lowest Wanda scores
- **Conservative**: 0.2 (fewer weights updated, safer)
- **Aggressive**: 0.5 (more weights updated, more recovery)

### max_relative_update (default: 2.0)
- **Range**: > 0.0
- **Meaning**: Maximum allowed update as a multiple of the original weight magnitude
- **Example**: 2.0 means |Δw_ij| ≤ 2.0 × |w_ij|
- **Conservative**: 1.0 (smaller updates, more stable)
- **Aggressive**: 5.0 (larger updates, more recovery potential)

## How It Works

1. **After pruning**, some weights are zeroed out, causing activation drift
2. **Compute lost signal**: ε_i = sum(pruned_weights × activations)
3. **Select update candidates**: Choose survivors with LOWEST Wanda scores
   - Wanda score = |w_ij| × sqrt(||activation_j||²)
   - Lower score = less important = cheaper to modify
4. **Redistribute inversely**:
   - Compute coefficient: c_ij = 1 / (wanda_score + ε)
   - Normalize: c_ij / sum(c_ij) = 1.0 per row
5. **Update weights**: w_ij_new = w_ij_old + c_ij × ε_i / activation_j
6. **Apply cap**: Ensure |Δw_ij| ≤ max_relative_update × |w_ij|

## Expected Output

During pruning, you'll see recovery statistics:
```
Processing layer: model.layers.0
  Sublayer: self_attn.q_proj
    Pruned 2048/4096 weights (50.00%)
    Recovery - Relative error: 0.023456
```

**relative_error**: Lower is better (< 0.1 is good, < 0.05 is excellent)

## Configuration Examples

### Balanced (Recommended)
```python
strategy = InverseWandaStrategy(
    update_fraction=0.3,
    max_relative_update=2.0
)
```

### Conservative (Safer, less recovery)
```python
strategy = InverseWandaStrategy(
    update_fraction=0.2,
    max_relative_update=1.0
)
```

### Aggressive (More recovery, higher risk)
```python
strategy = InverseWandaStrategy(
    update_fraction=0.5,
    max_relative_update=5.0
)
```

## Verification

After pruning with redistribution:
- ✅ Pruned weights remain exactly zero
- ✅ Selected survivors have their values modified
- ✅ No NaN or Inf values in weights
- ✅ Relative error reduced compared to no redistribution

## CLI Usage

Run pruning with inverse Wanda redistribution:

```bash
cd custom/pruner

# Basic usage (default parameters)
python wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --use_recovery

# Custom parameters (conservative)
python wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --use_recovery \
    --inverse_wanda_update_fraction 0.2 \
    --inverse_wanda_max_relative_update 1.0

# Custom parameters (aggressive)
python wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --use_recovery \
    --inverse_wanda_update_fraction 0.5 \
    --inverse_wanda_max_relative_update 5.0

# With 2:4 structured sparsity
python wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_type 2:4 \
    --use_recovery
```

## Integration with Existing Code

The redistributor integrates seamlessly with existing pruners:
- Works with any `BasePruner` subclass
- Compatible with all sparsity types (unstructured, 2:4, 4:8)
- No changes needed to model architecture
- Layer-by-layer application (memory efficient)
