# Model Evaluation Metrics

This directory contains scripts for evaluating language model performance, particularly perplexity measurements for comparing original and pruned models.

## Scripts

### 1. `perplexity.py`
Calculate perplexity for a single model on WikiText2 or C4 datasets.

**Usage:**
```bash
python perplexity.py --model <MODEL_PATH> [OPTIONS]
```

**Arguments:**
- `--model`: **Required** - Model name (HuggingFace) or local path
- `--dataset`: Dataset to use - `wikitext2` (default) or `c4`
- `--seqlen`: Sequence length for evaluation (default: 2048)
- `--cache_dir`: Directory to cache model weights (default: llm_weights)
- `--output`: Output JSON file path (default: auto-generated)
- `--device`: Device to use - `cuda` (default) or `cpu`

**Examples:**

```bash
# Evaluate original model on WikiText2
python perplexity.py \
    --model meta-llama/Llama-3.2-1B \
    --dataset wikitext2 \
    --seqlen 2048

# Evaluate pruned model
python perplexity.py \
    --model ../wanda/out/llama_3.2_1b_pruned_model/ \
    --dataset wikitext2 \
    --output results/pruned_model_ppl.json

# Evaluate on C4 dataset
python perplexity.py \
    --model meta-llama/Llama-3.2-1B \
    --dataset c4 \
    --seqlen 2048
```

**Output:**
- Console output with perplexity score
- JSON file with detailed results:
  ```json
  {
    "perplexity": 12.34,
    "dataset": "wikitext2",
    "seqlen": 2048,
    "model_name": "meta-llama/Llama-3.2-1B",
    "timestamp": "2025-12-27T..."
  }
  ```

---

### 2. `compare_models.py`
Compare perplexity and other metrics across multiple models (e.g., original vs pruned versions).

**Usage:**
```bash
python compare_models.py --models <MODEL1> <MODEL2> ... [OPTIONS]
```

**Arguments:**
- `--models`: **Required** - List of model paths to compare (space-separated)
- `--names`: Display names for models (optional, default: "Model 1", "Model 2", ...)
- `--dataset`: Dataset to use - `wikitext2` (default) or `c4`
- `--seqlen`: Sequence length for evaluation (default: 2048)
- `--cache_dir`: Directory to cache model weights (default: llm_weights)
- `--output`: Output JSON file path (default: results/model_comparison.json)
- `--device`: Device to use - `cuda` (default) or `cpu`

**Examples:**

```bash
# Compare original vs pruned model
python compare_models.py \
    --models meta-llama/Llama-3.2-1B ../wanda/out/llama_3.2_1b_pruned_50/ \
    --names "Original" "Pruned-50%" \
    --dataset wikitext2

# Compare multiple pruning ratios
python compare_models.py \
    --models \
        meta-llama/Llama-3.2-1B \
        ../wanda/out/llama_3.2_1b_pruned_50/ \
        ../wanda/out/llama_3.2_1b_pruned_60/ \
        ../wanda/out/llama_3.2_1b_pruned_70/ \
    --names "Dense" "50% Sparse" "60% Sparse" "70% Sparse" \
    --dataset wikitext2 \
    --output results/sparsity_comparison.json
```

**Output:**
- Comparison table showing:
  - Total parameters
  - Non-zero parameters
  - Sparsity percentage
  - Model size (MB)
  - Perplexity
- Improvement table relative to baseline (first model)
- JSON file with all results

**Example Output:**
```
╒═══════════════╤════════════════╤══════════════════╤═══════════════╤═══════════╤═════════════╕
│ Model         │ Total Params   │ Non-Zero Params  │ Sparsity (%)  │ Size (MB) │ Perplexity  │
╞═══════════════╪════════════════╪══════════════════╪═══════════════╪═══════════╪═════════════╡
│ Original      │ 1,235,814,400  │ 1,235,814,400    │ 0.00          │ 2,471.63  │ 11.2345     │
├───────────────┼────────────────┼──────────────────┼───────────────┼───────────┼─────────────┤
│ Pruned-50%    │ 1,235,814,400  │ 617,907,200      │ 50.00         │ 1,235.82  │ 12.5678     │
╘═══════════════╧════════════════╧══════════════════╧═══════════════╧═══════════╧═════════════╛
```

---

## Output Directory Structure

Results are saved to the `results/` directory:

```
results/
├── perplexity_meta-llama_Llama-3.2-1B_wikitext2.json
├── perplexity_pruned_model_wikitext2.json
└── model_comparison.json
```

---

## Perplexity Metric

**Perplexity (PPL)** measures how well a language model predicts a sample of text:
- **Lower is better** - indicates better model performance
- PPL = exp(average negative log-likelihood)
- Common baseline perplexities:
  - LLaMA-7B on WikiText2: ~5.68
  - LLaMA-13B on WikiText2: ~5.09
  - LLaMA-3.2-1B on WikiText2: ~10-12 (approximate)

**Interpreting Results:**
- PPL increase of 1-2 points after pruning: Acceptable quality
- PPL increase of 2-5 points: Moderate quality degradation
- PPL increase > 5 points: Significant quality loss

---

## Dependencies

Required packages (included in main environment):
```bash
pip install transformers torch datasets tabulate
```

---

## Integration with Wanda Pruning

These scripts integrate seamlessly with the Wanda pruning workflow:

```bash
# Step 1: Prune model
cd ../wanda
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda_50/ \
    --save_model out/llama_3.2_1b_pruned_50/

# Step 2: Evaluate perplexity
cd ../metrics
python perplexity.py \
    --model ../wanda/out/llama_3.2_1b_pruned_50/ \
    --dataset wikitext2

# Step 3: Compare with original
python compare_models.py \
    --models meta-llama/Llama-3.2-1B ../wanda/out/llama_3.2_1b_pruned_50/ \
    --names "Original" "Wanda-50%"
```

---

## Notes

- For 16GB GPUs, use `--seqlen 2048` or smaller
- WikiText2 evaluation is faster than C4 (~2-5 minutes vs 10-20 minutes)
- The first model in comparison is used as the baseline
- Results are automatically saved with timestamps
- GPU memory is cleared between model evaluations in comparison mode

---

## Troubleshooting

**Out of Memory Error:**
```bash
# Use shorter sequences
python perplexity.py --model <MODEL> --seqlen 1024
```

**Model Loading Error:**
```bash
# Specify correct cache directory
python perplexity.py --model <MODEL> --cache_dir /path/to/cache
```

**Tokenizer Error:**
```bash
# For pruned models, the tokenizer is loaded from the original model path
# This is handled automatically in compare_models.py
```
