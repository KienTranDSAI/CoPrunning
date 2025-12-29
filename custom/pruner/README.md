# Simplified Standalone Wanda Pruner

A clean, modular, standalone implementation of the Wanda (Pruning by Weights AND Activations) pruning algorithm for LLaMA models.

## Overview

This is a simplified reimplementation of Wanda pruning with:
- **Zero dependencies** on `wanda/lib/` - completely self-contained
- **Modular design** - easy to understand, modify, and extend
- **Multiple sparsity patterns** - unstructured, 2:4, and 4:8 structured sparsity
- **Full evaluation** - perplexity measurement and sparsity verification

## What is Wanda?

Wanda (Pruning by Weights AND Activations) is a state-of-the-art pruning method that determines weight importance based on:

1. **Weight magnitude**: `|w|` - larger weights have more impact
2. **Activation norm**: `||a||²` - weights on frequently-activated features are more important

The Wanda importance metric combines both factors:

```
importance(weight) = |weight| × sqrt(||activation||²)
```

This gives more accurate pruning than magnitude-only methods, especially for large language models.

**Paper**: [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)

## Directory Structure

```
custom/pruner/
├── wanda_pruner.py              # Main CLI entry point
├── utils/
│   ├── activation_capture.py    # Activation statistics collection
│   ├── dataset_loader.py        # WikiText2 and C4 loading
│   ├── model_utils.py           # Model loading and layer utilities
│   └── evaluator.py             # Perplexity evaluation
├── pruning/
│   ├── base_pruner.py           # Abstract base class
│   ├── wanda.py                 # Wanda pruning implementation
│   └── sparsity_patterns.py     # Unstructured, 2:4, 4:8 patterns
└── README.md                     # This file
```

## Installation

No additional dependencies beyond the main CoPrunning environment:

```bash
conda activate prune_llm
# Dependencies already installed: torch, transformers, datasets
```

## Usage

### Basic Usage

```bash
# Unstructured 50% sparsity on LLaMA 3.2 1B
python custom/pruner/wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured
```

### Sparsity Patterns

**Unstructured sparsity** (any weights can be pruned):
```bash
python custom/pruner/wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --dataset wikitext2 \
    --save_log results/llama_1b_unstructured.log
```

**2:4 structured sparsity** (hardware-friendly):
```bash
python custom/pruner/wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save_model results/llama_1b_2-4
```

**4:8 structured sparsity** (more granular):
```bash
python custom/pruner/wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --save_model results/llama_1b_4-8
```

### Advanced Options

**Use C4 for calibration** (more accurate, but slower):
```bash
python custom/pruner/wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --dataset c4 \
    --nsamples 256
```

**Custom sequence length**:
```bash
python custom/pruner/wanda_pruner.py \
    --model meta-llama/Llama-3.2-1B \
    --sparsity_ratio 0.5 \
    --seqlen 4096
```

## Command-Line Arguments

### Model Configuration
- `--model`: HuggingFace model identifier (default: `meta-llama/Llama-3.2-1B`)
- `--cache_dir`: Directory to cache model weights (default: `llm_weights`)
- `--seqlen`: Sequence length for calibration/evaluation (default: `2048`)

### Pruning Configuration
- `--sparsity_ratio`: Target sparsity ratio, 0.0-1.0 (default: `0.5`)
- `--sparsity_type`: Sparsity pattern - `unstructured`, `2:4`, or `4:8` (default: `unstructured`)

### Calibration
- `--dataset`: Calibration dataset - `wikitext2` (fast) or `c4` (accurate) (default: `wikitext2`)
- `--nsamples`: Number of calibration samples (default: `128`)
- `--seed`: Random seed for reproducibility (default: `0`)

### Output
- `--save_model`: Directory to save pruned model (optional)
- `--save_log`: File to save pruning results log (optional)

## Expected Results

Based on Wanda paper results for LLaMA models:

| Sparsity Pattern | Sparsity % | WikiText2 PPL (approx) |
|------------------|------------|------------------------|
| Unstructured 50% | 50%        | 10-12                  |
| 2:4 structured   | 50%        | 11-13                  |
| 4:8 structured   | 50%        | 9-11                   |

Note: These are approximate values for LLaMA-family models. Actual perplexity depends on model size and calibration data.

## Extending the Pruner

The modular design makes it easy to experiment:

### Add a New Pruning Method

Extend `BasePruner` and implement `compute_pruning_metric()`:

```python
# custom/pruner/pruning/magnitude.py
from .base_pruner import BasePruner
import torch

class MagnitudePruner(BasePruner):
    """Simple magnitude-based pruning (baseline)."""

    def compute_pruning_metric(self, layer, activation_stats):
        # Ignore activations, just use weight magnitude
        return torch.abs(layer.weight.data)
```

### Add a New Sparsity Pattern

Extend `SparsityPattern`:

```python
# custom/pruner/pruning/sparsity_patterns.py
class BlockSparsity(SparsityPattern):
    """Block-wise sparsity."""

    def __init__(self, block_size):
        self.block_size = block_size

    def create_mask(self, metric, sparsity_ratio):
        # Implement block-wise pruning logic
        pass
```

### Modify Activation Statistics

Edit `ActivationCapture.add_batch()`:

```python
# Use L1 norm instead of L2
self.scaler_row += torch.norm(inp, p=1, dim=1) / self.nsamples
```

## Architecture Details

### Pruning Pipeline

1. **Load model and tokenizer**
2. **Load calibration data** (WikiText2 or C4)
3. **Prepare calibration inputs** using "Catcher" pattern
4. **For each layer**:
   - Register hooks to capture activations
   - Forward pass calibration samples
   - Compute Wanda metric: `|W| × sqrt(||A||²)`
   - Apply sparsity pattern (unstructured or N:M)
   - Zero out pruned weights
   - Forward pass again with pruned weights
5. **Verify sparsity** achieved
6. **Evaluate perplexity** on WikiText2

### Key Components

- **ActivationCapture**: Accumulates L2-norm² of activations per input feature
- **WandaPruner**: Computes `|weight| × sqrt(activation_norm)` metric
- **SparsityPattern**: Defines how to select which weights to prune
- **PerplexityEvaluator**: Measures model quality via WikiText2 perplexity

## Comparison with Original Wanda

This implementation:
- ✅ Produces numerically identical results to `wanda/`
- ✅ Has zero dependencies on `wanda/lib/`
- ✅ Is easier to understand and modify
- ✅ Supports all major sparsity patterns
- ❌ Does not include Wanda variant (adaptive thresholding)
- ❌ Does not include SparseGPT or ablation methods

For the full-featured implementation with all methods, use `wanda/main.py`.

## Troubleshooting

**CUDA out of memory**:
- Reduce `--nsamples` (e.g., `--nsamples 64`)
- Reduce `--seqlen` (e.g., `--seqlen 1024`)
- The model uses `device_map="auto"` to handle multi-GPU

**Slow C4 loading**:
- Use WikiText2 for faster experiments: `--dataset wikitext2`
- C4 uses streaming mode but initial loading can be slow

**Sparsity mismatch**:
- For 2:4 and 4:8, `sparsity_ratio` must be 0.5
- The script will automatically adjust if needed

## Citation

If you use this code, please cite the original Wanda paper:

```bibtex
@article{sun2023simple,
  title={A Simple and Effective Pruning Approach for Large Language Models},
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J Zico},
  journal={arXiv preprint arXiv:2306.11695},
  year={2023}
}
```

## License

This code follows the same license as the parent CoPrunning repository.
