# Wanda Library Documentation

This directory contains the core implementation of various neural network pruning algorithms, including Wanda (Pruning by Weights and Activations), SparseGPT, and magnitude-based pruning methods for large language models.

## Table of Contents

1. [ablate.py](#ablatepy)
2. [data.py](#datapy)
3. [eval.py](#evalpy)
4. [layerwrapper.py](#layerwrapperpy)
5. [prune_opt.py](#prune_optpy)
6. [prune.py](#prunepy)
7. [sparsegpt.py](#sparsegptpy)

---

## ablate.py

**Purpose**: Implements the `AblateGPT` class for neural network pruning with reconstruction error minimization.

### Key Components

#### Class: `AblateGPT`

A pruning implementation that combines weight magnitude with activation statistics (similar to Wanda) and uses Optimal Brain Surgeon (OBS) methodology to minimize reconstruction error.

**Initialization** (`__init__`):
- Takes a neural network layer as input
- Extracts weight dimensions (rows, columns)
- Initializes Hessian matrix `H` for storing second-order information
- Initializes `scaler_row` to store activation norms per input feature
- Supports `nn.Linear`, `nn.Conv2d`, and `transformers.Conv1D` layers

**Methods**:

1. **`add_batch(inp, out)`** - Lines 28-43
   - Accumulates input activation statistics during calibration
   - Updates the Hessian matrix `H = inp @ inp.T` (outer product of inputs)
   - Computes and updates `scaler_row`: L2 norm of inputs per feature dimension
   - Uses incremental averaging to combine statistics from multiple batches

2. **`get_wanda_mask(sparsity, prunen, prunem)`** - Lines 45-58
   - Generates pruning mask using Wanda metric: `|W| * sqrt(scaler_row)`
   - Combines weight magnitude with activation importance
   - Supports both:
     - **Unstructured pruning**: Remove individual weights based on global threshold
     - **Structured N:M sparsity**: Keep N non-zero weights in every M consecutive weights
   - Returns binary mask indicating which weights to prune

3. **`get_mag_mask(sparsity, prunen, prunem)`** - Lines 60-73
   - Generates pruning mask using only weight magnitude: `|W|`
   - Simpler baseline compared to Wanda
   - Supports same sparsity patterns as `get_wanda_mask`

4. **`fasterprune(args, sparsity, mask, prune_n, prune_m, blocksize, percdamp)`** - Lines 75-158
   - Main pruning algorithm using Optimal Brain Surgeon (OBS) methodology
   - Processes weights in blocks of size `blocksize` (default: 128)
   - For each block:
     - Computes inverse Hessian with damping for numerical stability
     - Determines which weights to prune (using Wanda/magnitude metric)
     - Sets pruned weights to zero
     - Redistributes pruning error to remaining weights using Hessian information
   - This reconstruction approach minimizes output perturbation
   - Supports both sequential (one-shot) and iterative pruning via mask parameter

**Configuration**:
- `percdamp`: Damping factor (default 0.01) for Hessian regularization
- `blocksize`: Number of columns processed at once (default 128)
- Disables TF32 for numerical precision in matrix operations

---

## data.py

**Purpose**: Dataset loading utilities for calibration and evaluation of language models.

### Key Functions

1. **`set_seed(seed)`** - Lines 9-11
   - Sets random seeds for NumPy and PyTorch for reproducibility
   - Ensures consistent sampling across runs

2. **`TokenizerWrapper`** - Lines 14-16
   - Simple wrapper class to hold tokenized input IDs
   - Used for validation data formatting

3. **`get_wikitext2(nsamples, seed, seqlen, tokenizer)`** - Lines 19-38
   - Loads WikiText-2 dataset (raw version) from HuggingFace datasets
   - **Training data**:
     - Joins all text with spaces
     - Randomly samples `nsamples` sequences of length `seqlen`
     - Creates (input, target) pairs for language modeling
     - Targets have `-100` for all but last token (standard LM objective)
   - **Test data**:
     - Joins all text with double newlines
     - Returns full tokenized test set
   - Used for perplexity evaluation and calibration

4. **`get_c4(nsamples, seed, seqlen, tokenizer)`** - Lines 41-66
   - Loads C4 dataset (Colossal Clean Crawled Corpus) from HuggingFace
   - Uses first shard of training data (`c4-train.00000-of-01024.json.gz`)
   - **Training data**:
     - Randomly samples documents until finding ones longer than `seqlen`
     - Extracts random subsequences of length `seqlen`
     - More diverse than WikiText-2
   - **Validation data**:
     - Uses first 1100 examples from validation set
     - Truncates to `256 * seqlen` tokens
   - Preferred for calibration due to domain diversity

5. **`get_loaders(name, nsamples, seed, seqlen, tokenizer)`** - Lines 69-73
   - Router function that selects appropriate dataset loader
   - Returns (train_loader, test_encoder) tuple
   - Supports "wikitext2" and "c4" dataset names

**Default Parameters**:
- `nsamples=128`: Number of calibration samples
- `seed=0`: Random seed for sampling
- `seqlen=2048`: Sequence length (context window)

---

## eval.py

**Purpose**: Evaluation functions for measuring model performance through perplexity and zero-shot task accuracy.

### Key Functions

1. **`eval_ppl(args, model, tokenizer, device)`** - Lines 14-29
   - Main entry point for perplexity evaluation
   - Currently hardcoded to evaluate on WikiText-2 test set
   - Loads test data with model's sequence length
   - Calls `eval_ppl_wikitext` with batch size of 1
   - Returns perplexity score (lower is better)

2. **`eval_ppl_wikitext_train(model, trainloader, bs, device)`** - Lines 32-80
   - Evaluates perplexity on training data samples
   - Useful for checking calibration data fit
   - **Process**:
     - Iterates through training samples in batches
     - Computes model logits for each sequence
     - Calculates cross-entropy loss between predictions and actual next tokens
     - Accumulates negative log-likelihoods
     - Computes perplexity as `exp(total_nll / total_tokens)`
   - Progress printed every 50 samples
   - Clears CUDA cache after evaluation

3. **`eval_ppl_wikitext(model, testenc, bs, device)`** - Lines 83-129
   - Evaluates perplexity on WikiText-2 test set
   - **Process**:
     - Splits test set into non-overlapping sequences of `model.seqlen`
     - For each sequence:
       - Forward pass through model
       - Shift logits and labels for next-token prediction
       - Compute cross-entropy loss
       - Accumulate negative log-likelihood
     - Final perplexity = `exp(sum(nll) / (nsamples * seqlen))`
   - Standard metric for language model evaluation
   - Batch size typically 1 to save memory

4. **`eval_zero_shot(model_name, model, tokenizer, task_list, num_fewshot, use_accelerate, add_special_tokens)`** - Lines 132-165
   - Evaluates model on zero-shot or few-shot downstream tasks
   - Uses `lm-eval-harness` library for standardized evaluation
   - **Default tasks**:
     - `boolq`: Boolean question answering
     - `rte`: Recognizing textual entailment
     - `hellaswag`: Commonsense reasoning
     - `winogrande`: Pronoun resolution
     - `arc_challenge` & `arc_easy`: Science QA
     - `openbookqa`: Open book question answering
   - **Parameters**:
     - `num_fewshot=0`: Number of in-context examples (0 = zero-shot)
     - `use_accelerate=False`: Whether to use HF Accelerate for distributed inference
     - `add_special_tokens=False`: Tokenization behavior
   - Limits evaluation to 2000 samples for 65B/70B models to save time
   - Returns detailed results dictionary with accuracy metrics

**Helper**:
- `pattern_match(patterns, source_list)` - Lines 135-140: Matches task names using glob patterns

---

## layerwrapper.py

**Purpose**: Provides `WrappedGPT` class for wrapping neural network layers to collect activation statistics.

### Class: `WrappedGPT`

A lightweight wrapper for collecting activation norms during forward passes. Simpler than `AblateGPT` - only tracks activation statistics without Hessian information.

**Initialization** (`__init__`):
- Takes a layer, optional layer ID, and layer name
- Extracts weight dimensions and device
- Initializes `scaler_row` to store L2 norms of input activations
- Tracks number of samples seen (`nsamples`)

**Method: `add_batch(inp, out)`** - Lines 22-35
- Called during forward pass via PyTorch hooks
- **Process**:
  - Handles 2D and 3D input tensors
  - Reshapes and transposes to get activations in correct format
  - Updates `scaler_row` using incremental averaging
  - Computes L2 norm (squared) per input feature dimension
  - Formula: `scaler_row = (scaler_row * old_n + new_norms) / new_n`

**Usage Pattern**:
```python
wrapped = WrappedGPT(layer)
# Register as forward hook
handle = layer.register_forward_hook(
    lambda _, inp, out: wrapped.add_batch(inp[0], out)
)
# Run calibration data through model
# Access statistics via wrapped.scaler_row
handle.remove()
```

**Difference from AblateGPT**:
- No Hessian computation (lighter memory footprint)
- Only tracks activation norms for Wanda metric
- Used in `prune_wanda` for simple one-shot pruning

---

## prune_opt.py

**Purpose**: Main pruning orchestration for OPT-family models (Facebook's Open Pretrained Transformers). Contains implementations of all pruning strategies.

### Utility Functions

1. **`find_layers(module, layers, name)`** - Lines 11-30
   - Recursively traverses PyTorch module tree
   - Finds all layers of specified types (default: `nn.Linear`)
   - Returns dictionary mapping layer names to layer objects
   - Used to identify which layers to prune (typically all linear projections)

2. **`check_sparsity(model)`** - Lines 32-56
   - Computes actual sparsity (percentage of zero weights) in model
   - Iterates through all decoder layers
   - Reports per-layer and overall sparsity
   - Used to verify pruning achieved target sparsity ratio

3. **`prepare_calibration_input(model, dataloader, device)`** - Lines 58-92
   - Crucial preprocessing step for layer-wise pruning
   - **Process**:
     - Temporarily replaces first decoder layer with `Catcher` module
     - Runs calibration data through model
     - `Catcher` intercepts and stores:
       - Layer inputs (`inps`)
       - Attention masks
       - Position IDs (missing in this version, see prune.py for correct version)
     - Restores original first layer
   - Returns calibration inputs needed for layer-wise pruning
   - Handles multi-GPU scenarios via `hf_device_map`

4. **`return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)`** - Lines 94-100
   - Helper for Wanda variant algorithm
   - Computes pruning threshold based on alpha parameter
   - Uses cumulative sum of sorted weights
   - Returns mask and achieved sparsity ratio

### Pruning Algorithms

#### 1. **`prune_magnitude(args, model, tokenizer, device, prune_n, prune_m)`** - Lines 102-122

**Type**: Baseline method (no calibration data needed)

**Process**:
- Iterates through all decoder layers
- For each linear layer:
  - Computes weight magnitude: `|W|`
  - Creates pruning mask based on magnitude
  - Sets smallest weights to zero

**Sparsity Patterns**:
- **Unstructured** (`prune_n=0`): Global threshold based on `args.sparsity_ratio`
- **Structured N:M** (`prune_n>0`): Keep largest N weights in every M consecutive weights

**Advantages**: Fast, no calibration needed
**Disadvantages**: Ignores activation statistics, often worse performance

#### 2. **`prune_wanda(args, model, tokenizer, device, prune_n, prune_m)`** - Lines 124-187

**Type**: Activation-aware pruning (requires calibration)

**Process**:
1. Load C4 calibration data (128 samples by default)
2. Prepare calibration inputs for first layer
3. For each decoder layer:
   - Wrap all linear layers with `WrappedGPT`
   - Register forward hooks to collect activation statistics
   - Run calibration data through layer
   - For each linear sublayer:
     - Compute Wanda metric: `|W| * sqrt(scaler_row)`
     - Create pruning mask based on metric
     - Set pruned weights to zero
   - Run calibration data through pruned layer (for next layer)
4. Swap inputs/outputs and proceed to next layer

**Key Features**:
- Uses `WrappedGPT` for lightweight activation collection
- Supports unstructured and N:M structured sparsity
- Layer-wise pruning maintains calibration data flow
- Handles multi-GPU models

#### 3. **`prune_sparsegpt(args, model, tokenizer, dev, prune_n, prune_m)`** - Lines 189-275

**Type**: Second-order pruning with reconstruction (requires calibration)

**Process**:
1. Load C4 calibration data
2. Capture layer inputs using `Catcher` module
3. For each decoder layer:
   - Wrap all linear layers with `SparseGPT` class
   - Collect Hessian information via forward hooks
   - For each sublayer:
     - Run `fasterprune()` with second-order information
     - Minimizes reconstruction error
     - Frees Hessian memory
   - Recompute outputs for next layer

**Key Features**:
- Uses full Hessian matrix (memory intensive)
- Redistributes pruning error to unpruned weights
- Better preservation of model accuracy
- Slower than Wanda but higher quality

**Source**: Based on [SparseGPT paper](https://github.com/IST-DASLab/sparsegpt)

#### 4. **`prune_ablate(args, model, tokenizer, dev, prune_n, prune_m)`** - Lines 277-372

**Type**: Hybrid approach combining selection and reconstruction

**Process**:
1. Load C4 calibration data
2. Capture layer inputs
3. For each decoder layer:
   - Wrap layers with `AblateGPT` class
   - Collect both Hessian and activation statistics
   - For each sublayer:
     - **Selection phase**: Choose pruning mask using:
       - `ablate_wanda_seq`: Wanda metric (sequential)
       - `ablate_mag_seq`: Magnitude metric (sequential)
       - `ablate_*_iter`: Iterative methods (no pre-computed mask)
     - **Reconstruction phase**: Use `fasterprune()` to minimize error
   - Recompute outputs

**Key Features**:
- Combines benefits of Wanda (smart selection) and SparseGPT (reconstruction)
- Supports multiple pruning strategies via `args.prune_method`
- More flexible than pure Wanda or SparseGPT

**Note**: Missing position_ids tracking (see prune.py for complete version)

---

## prune.py

**Purpose**: Similar to `prune_opt.py` but designed for LLaMA-family models and other modern architectures. Contains more complete implementations with position IDs support.

### Key Differences from prune_opt.py

1. **Model Architecture**:
   - References `model.model.layers` (LLaMA structure)
   - vs `model.model.decoder.layers` (OPT structure)

2. **Position IDs Tracking**:
   - Properly captures and propagates `position_ids` through layers
   - Critical for RoPE (Rotary Position Embeddings) used in LLaMA
   - Lines 80, 92, 135, 240, 253, 331, 344

3. **Wanda Variant Support**:
   - Includes Wanda variant algorithm (Lines 178-196)
   - Uses binary search to find alpha parameter
   - Achieves exact target sparsity via cumulative weight threshold
   - Activated via `args.use_variant` flag

### Updated Functions

All functions have same structure as `prune_opt.py` but with corrections:

1. **`prepare_calibration_input`** - Lines 58-95
   - Returns `(inps, outs, attention_mask, position_ids)` (4-tuple vs 3-tuple)

2. **`prune_wanda`** - Lines 127-210
   - Accepts and propagates position_ids
   - Includes Wanda variant logic for dynamic thresholding

3. **`prune_sparsegpt`** - Lines 214-300
   - Properly handles position_ids in forward passes

4. **`prune_ablate`** - Lines 304-398
   - Properly handles position_ids in forward passes

### Wanda Variant Algorithm (Lines 178-196)

**Purpose**: Dynamic thresholding for exact sparsity

**Algorithm**:
1. Sort weights by Wanda metric: `|W| * sqrt(scaler_row)`
2. Compute cumulative sum of sorted metrics
3. Use binary search to find alpha such that:
   - `sum(metrics where metric <= alpha * total_sum) ≈ target_sparsity`
4. Creates more balanced pruning across rows

**Parameters**:
- Initial alpha = 0.4
- Search bounds = [0.0, 0.8]
- Convergence threshold = 0.001 (0.1% sparsity tolerance)

---

## sparsegpt.py

**Purpose**: Implements the SparseGPT algorithm for neural network pruning based on second-order optimization.

### Class: `SparseGPT`

A sophisticated pruning method that uses the Hessian of the loss with respect to weights to minimize reconstruction error.

**Theoretical Foundation**:
- Based on Optimal Brain Surgeon (OBS) methodology
- Minimizes: `||Y - f(X; W_pruned)||²` where Y = f(X; W_original)
- Uses second-order Taylor expansion around original weights

**Initialization** (`__init__`):
- Takes a neural network layer
- Initializes Hessian matrix `H` (columns × columns)
- Supports `nn.Linear`, `nn.Conv2d`, `transformers.Conv1D`
- No activation tracking (unlike `AblateGPT`)

**Methods**:

1. **`add_batch(inp, out)`** - Lines 27-38
   - Accumulates Hessian information during calibration
   - Computes: `H = inp @ inp.T` (Fisher information approximation)
   - Uses incremental averaging with scaling factor `sqrt(2/nsamples)`
   - Assumes squared error loss

2. **`fasterprune(sparsity, prune_n, prune_m, blocksize, percdamp)`** - Lines 40-116
   - Main pruning algorithm
   - **Process**:
     1. **Hessian preprocessing**:
        - Add damping: `H[i,i] += damp` for numerical stability
        - Compute inverse via Cholesky decomposition
        - Handle dead neurons (zero diagonal elements)
     2. **Block-wise pruning** (blocks of size `blocksize`):
        - For each block:
          - Determine pruning mask using metric: `W² / diag(H⁻¹)²`
          - For unstructured: global threshold
          - For N:M structured: per-group top-k
        - For each column in block:
          - Set pruned weights to zero
          - Compute reconstruction error
          - Redistribute error to remaining columns using Hessian
          - Formula: `W[:, j:] -= (W[:,i] - Q[:,i]) / H⁻¹[i,i] * H⁻¹[i,j:]`
     3. **Update layer weights** with pruned version

**Key Features**:
- Block-wise processing reduces memory (don't need full inverse)
- Error redistribution minimizes output perturbation
- Supports both unstructured and N:M structured sparsity
- More accurate than magnitude but computationally expensive

**Configuration**:
- `blocksize=128`: Trade-off between memory and accuracy
- `percdamp=0.01`: Damping factor for numerical stability
- Disables TF32 for precision

**Complexity**:
- Time: O(columns² * rows / blocksize)
- Space: O(columns²) for Hessian

**Reference**: [SparseGPT GitHub](https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa)

---

## Algorithm Comparison

| Algorithm | Calibration Data | Hessian | Activation Stats | Complexity | Quality |
|-----------|-----------------|---------|------------------|------------|---------|
| **Magnitude** | ✗ | ✗ | ✗ | O(params) | Baseline |
| **Wanda** | ✓ | ✗ | ✓ | O(params) | Good |
| **SparseGPT** | ✓ | ✓ | ✗ | O(params²) | Best |
| **Ablate** | ✓ | ✓ | ✓ | O(params²) | Best |

**Recommendations**:
- **Quick baseline**: Use `prune_magnitude`
- **Best speed/quality**: Use `prune_wanda`
- **Best quality**: Use `prune_sparsegpt` or `prune_ablate`
- **Exact sparsity**: Use `prune_wanda` with `use_variant=True`

## Common Parameters

- `args.sparsity_ratio`: Target proportion of weights to prune (e.g., 0.5 = 50%)
- `args.nsamples`: Number of calibration samples (default: 128)
- `args.seed`: Random seed for reproducibility
- `prune_n`, `prune_m`: For N:M structured sparsity (N non-zeros per M weights)
- `args.prune_method`: Specifies pruning algorithm variant

## Architecture Support

- **prune_opt.py**: OPT, GPT-2, and decoder-only models using `model.model.decoder.layers`
- **prune.py**: LLaMA, LLaMA-2, Mistral, and models using `model.model.layers` with RoPE

## References

- **Wanda**: [Paper Link - Add if available]
- **SparseGPT**: [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)
- **Optimal Brain Surgeon**: Classic work on second-order pruning
