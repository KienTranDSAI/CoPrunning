# Layer Analysis Tools

This directory contains tools for analyzing layers in LLaMA models before and during pruning.

## Scripts

### 1. `analysis_model_layer.py`
Lists all layer types in the model with their counts and paths.

**Usage:**
```bash
python analysis_model_layer.py
```

**Output:** Summary of all unique layer types (Linear, Embedding, RMSNorm, etc.) with example paths.

---

### 2. `list_prunable_layers.py`
Lists all Linear layers that can be pruned, grouped by type.

**Usage:**
```bash
python list_prunable_layers.py --model meta-llama/Llama-3.2-1B
```

**Output:**
- Grouped list of all Linear layers
- Attention layers (Q, K, V, O projections)
- MLP layers (Gate, Up, Down projections)
- Example commands for visualization

---

### 3. `visualize_layer_metrics.py`
Visualizes weight magnitude, activation norm, and Wanda metric for a specific layer.

**Usage:**
```bash
python visualize_layer_metrics.py --layer <LAYER_NAME>
```

**Arguments:**
- `--model`: Model name (default: meta-llama/Llama-3.2-1B)
- `--layer`: **Required** - Full layer name to analyze
- `--cache_dir`: Cache directory for model weights (default: llm_weights)
- `--nsamples`: Number of calibration samples (default: 128)

**Example Commands:**
```bash
# Analyze the query projection in the first layer
python visualize_layer_metrics.py --layer model.layers.0.self_attn.q_proj

# Analyze the gate projection in layer 10
python visualize_layer_metrics.py --layer model.layers.10.mlp.gate_proj

# Analyze with custom number of samples
python visualize_layer_metrics.py --layer model.layers.0.self_attn.q_proj --nsamples 256
```

**Output:**
- 9-panel visualization showing:
  - **Weight Magnitude**: Heatmap, histogram, and statistics
  - **Activation Norm**: Bar plot, histogram, and statistics
  - **Wanda Metric**: Heatmap, histogram, and statistics
- Saved PNG file: `layer_analysis_<layer_name>.png`

---

## Workflow

1. **Discover layer types:**
   ```bash
   python analysis_model_layer.py
   ```

2. **List prunable layers:**
   ```bash
   python list_prunable_layers.py
   ```

3. **Visualize specific layers:**
   ```bash
   python visualize_layer_metrics.py --layer model.layers.0.self_attn.q_proj
   ```

---

## Understanding the Visualizations

### Weight Magnitude
- Shows the absolute value of each weight in the layer
- Lower magnitude weights are typically pruned first in magnitude-based pruning
- Heatmap shows the full weight matrix structure

### Activation Norm
- L2 norm of input activations per output dimension
- Computed from calibration data (C4 dataset)
- Indicates which output dimensions are more "active"

### Wanda Metric
- Computed as: `|Weight| × √||Activation||`
- Combines both weight importance and activation patterns
- Wanda pruning removes weights with the **lowest** Wanda metric values
- This metric considers both structural importance (weights) and data-driven importance (activations)

---

## Notes

- All scripts require GPU for loading the model
- Calibration data is downloaded automatically from the C4 dataset
- Visualizations are saved to the current directory
- The Wanda metric computation matches the implementation in `wanda/lib/prune.py`
