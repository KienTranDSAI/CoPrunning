# Wanda Pruning Method

## Overview
Wanda (Pruning by Weights and Activations) is a one-shot pruning method that combines weight magnitudes with activation norms to identify unimportant weights.

## Logical Steps

### 1. Preparation
- Load pre-trained model
- Load calibration dataset (WikiText2 or C4)
- Set target sparsity ratio and sparsity type (unstructured or N:M)

### 2. Calibration Input Collection
- Run forward passes on calibration data
- Capture input activations for each transformer layer
- Store attention masks and position IDs

### 3. Layer-wise Pruning (iterate for each layer)

**Step 3.1: Activation Statistics Collection**
- Wrap linear layers with hooks
- Run `nsamples` forward passes (default: 128)
- Collect activation norms for each layer: `scaler_row = Σ(activation²)`

**Step 3.2: Compute Pruning Metric**
```
W_metric = |W| × √(scaler_row)
```
- `|W|`: Weight magnitude (importance from parameters)
- `√(scaler_row)`: Activation norm (importance from data)

**Step 3.3: Determine Pruning Mask**
- **Unstructured**: Sort weights by metric, prune lowest X% globally
- **N:M structured**: For every M consecutive weights, prune N with lowest metric
- **Variant (optional)**: Use adaptive threshold based on cumulative metric

**Step 3.4: Apply Pruning**
- Set selected weights to zero: `W[mask] = 0`

**Step 3.5: Update Activations**
- Run forward passes through pruned layer
- Update inputs for next layer

### 4. Evaluation
- Compute perplexity on WikiText2 test set
- Optionally run zero-shot task evaluation
- Verify actual sparsity achieved

## Key Insights

1. **One-shot pruning**: No iterative retraining required
2. **Data-aware**: Uses activations from calibration data, not just weight magnitudes
3. **Efficient**: Single forward pass per layer for calibration
4. **Layer-wise**: Each layer pruned independently, reducing memory requirements

## Comparison with Other Methods

| Method | Metric | Complexity |
|--------|--------|------------|
| Magnitude | \|W\| | Lowest |
| Wanda | \|W\| × √(activation) | Low |
| SparseGPT | Hessian-based OBS | High |

## Implementation Reference
- Main logic: `wanda/lib/prune.py:prune_wanda()`
- Activation capture: `wanda/lib/layerwrapper.py:WrappedGPT`
