# Memory Requirements Estimation for Wanda Pruning

This guide helps you estimate GPU memory requirements before running Wanda pruning experiments.

## Quick Reference Table

### LLaMA Models on WikiText2 Dataset

| Model Size | Parameters | Model (fp16) | Activations | Total Est. | Min GPU | Recommended GPU |
|------------|-----------|--------------|-------------|------------|---------|-----------------|
| LLaMA 1B   | 1.2B      | ~2.5 GB      | ~2-3 GB     | ~5-6 GB    | 8 GB    | 12-16 GB        |
| LLaMA 3B   | 3.2B      | ~6.5 GB      | ~4-5 GB     | ~11-12 GB  | 16 GB   | 24 GB           |
| LLaMA 7B   | 7B        | ~14 GB       | ~6-8 GB     | ~20-22 GB  | 24 GB   | 40 GB           |
| LLaMA 13B  | 13B       | ~26 GB       | ~8-10 GB    | ~34-36 GB  | 40 GB   | 48 GB           |
| LLaMA 30B  | 30B       | ~60 GB       | ~12-15 GB   | ~72-75 GB  | 80 GB   | Multi-GPU       |
| LLaMA 70B  | 70B       | ~140 GB      | ~20-25 GB   | ~160-165 GB| 2x80GB  | Multi-GPU       |

**Note:** These estimates are for `--dataset wikitext2` with default settings (`--nsamples 128`, `--seqlen 2048`).

---

## Detailed Memory Breakdown

### 1. Model Weights (Static)

Model weights in fp16 (half precision):

```
Memory = (Number of Parameters × 2 bytes) / (1024³)
```

**Examples:**
- LLaMA 1B: 1,235,814,400 params × 2 bytes = ~2.5 GB
- LLaMA 7B: 6,738,415,616 params × 2 bytes = ~14 GB
- LLaMA 13B: 13,015,864,320 params × 2 bytes = ~26 GB

### 2. Activation Memory (Dynamic)

Activations depend on:
- Sequence length (`--seqlen`)
- Number of calibration samples (`--nsamples`)
- Batch size (typically 1 for pruning)
- Model hidden dimensions

**Estimation Formula:**

```python
# Per layer activation (approximate)
hidden_dim = 2048  # For LLaMA 1B
num_layers = 16    # For LLaMA 1B
seq_len = 2048
nsamples = 128

# Attention activations
attn_memory = seq_len * hidden_dim * 4 * 2  # Q, K, V, O projections in fp16
attn_memory_gb = (attn_memory * num_layers) / (1024**3)

# MLP activations
ffn_dim = hidden_dim * 4  # Typical expansion
mlp_memory = seq_len * ffn_dim * 3 * 2  # gate, up, down in fp16
mlp_memory_gb = (mlp_memory * num_layers) / (1024**3)

# Total activation memory per sample
per_sample_gb = (attn_memory_gb + mlp_memory_gb)

# For nsamples stored in memory
total_activation_gb = per_sample_gb * min(nsamples, 10)  # Usually ~10 samples in memory at once
```

**Typical Ranges:**
- **LLaMA 1B** (seqlen=2048): ~2-3 GB
- **LLaMA 7B** (seqlen=2048): ~6-8 GB
- **LLaMA 13B** (seqlen=2048): ~8-10 GB

### 3. Calibration Data (WikiText2)

WikiText2 dataset memory footprint:

```
Dataset size: ~2 MB (negligible)
Tokenized data (cached): ~10-50 MB
Active batch in memory: ~(seqlen × 2 bytes) = ~4 KB per sample
```

**Impact:** Nearly zero - WikiText2 is tiny compared to model and activations.

### 4. Gradient Buffers & Optimizer State

For pruning (no backpropagation):
- **Gradient buffers:** Not needed (using `torch.no_grad()`)
- **Optimizer state:** Not needed
- **Additional overhead:** ~500 MB - 1 GB for PyTorch/CUDA overhead

### 5. Intermediate Pruning Buffers

During pruning, temporary buffers are needed:

```python
# Per layer buffers
W_metric = layer_out_dim × layer_in_dim × 4 bytes (fp32)
W_mask = layer_out_dim × layer_in_dim × 1 byte (bool)

# For LLaMA 1B attention layer (2048 × 2048)
W_metric = 2048 × 2048 × 4 = ~16.8 MB
W_mask = 2048 × 2048 × 1 = ~4.2 MB

# Total per layer: ~21 MB (negligible)
```

---

## Memory Optimization Strategies

### 1. Reduce Sequence Length

```bash
# Instead of default 2048
python main.py --seqlen 1024  # Cuts activation memory by ~50%
python main.py --seqlen 512   # Cuts activation memory by ~75%
```

**Impact:** Minimal effect on pruning quality for unstructured sparsity.

### 2. Reduce Number of Samples

```bash
# Instead of default 128
python main.py --nsamples 64   # Reduces calibration memory
python main.py --nsamples 32   # Further reduction
```

**Impact:** May slightly affect pruning quality, but often acceptable.

### 3. Use Smaller Dataset (Already Default)

```bash
# WikiText2 (default) - tiny dataset
python main.py --dataset wikitext2  # ~2 MB

# C4 - much larger (avoid if memory constrained)
python main.py --dataset c4  # ~500 GB download
```

### 4. Clear Cache Between Layers

The code already includes:
```python
torch.cuda.empty_cache()  # Called after each layer
```

### 5. Use CPU Offloading (for very large models)

```python
# Modify model loading in main.py
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically distributes across GPU/CPU
    offload_folder="offload",  # CPU offload directory
    low_cpu_mem_usage=True
)
```

---

## Real-World Examples

### Example 1: LLaMA 3.2 1B on 16GB GPU

**Configuration:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --dataset wikitext2 \
    --nsamples 128 \
    --seqlen 2048
```

**Memory Breakdown:**
- Model (fp16): ~2.5 GB
- Activations: ~2.5 GB
- Calibration data: ~10 MB
- PyTorch overhead: ~800 MB
- Pruning buffers: ~100 MB
- **Total:** ~5.9 GB
- **GPU Available:** 16 GB
- **Status:** ✅ **Safe** - ~10 GB headroom

### Example 2: LLaMA 7B on 24GB GPU

**Configuration:**
```bash
python main.py \
    --model meta-llama/Llama-7B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --dataset wikitext2 \
    --nsamples 128 \
    --seqlen 2048
```

**Memory Breakdown:**
- Model (fp16): ~14 GB
- Activations: ~7 GB
- Calibration data: ~10 MB
- PyTorch overhead: ~1 GB
- Pruning buffers: ~200 MB
- **Total:** ~22.2 GB
- **GPU Available:** 24 GB
- **Status:** ⚠️ **Tight** - Only ~2 GB headroom

**Optimization:**
```bash
# Use smaller sequence length
python main.py \
    --model meta-llama/Llama-7B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --dataset wikitext2 \
    --nsamples 64 \
    --seqlen 1024
```

**After Optimization:**
- Model (fp16): ~14 GB
- Activations: ~3.5 GB (reduced)
- Calibration data: ~10 MB
- PyTorch overhead: ~1 GB
- **Total:** ~18.5 GB
- **Status:** ✅ **Safe** - ~5.5 GB headroom

### Example 3: LLaMA 13B on 16GB GPU (Will Fail)

**Configuration:**
```bash
python main.py \
    --model meta-llama/Llama-13B \
    --sparsity_ratio 0.5 \
    --dataset wikitext2
```

**Memory Breakdown:**
- Model (fp16): ~26 GB
- **Status:** ❌ **FAILS** - Model alone exceeds GPU memory

**Solutions:**
1. Use a GPU with more memory (40GB or 48GB)
2. Use multi-GPU setup with `device_map="auto"`
3. Use CPU offloading (very slow)

---

## Memory Calculation Tool

Use this formula to estimate memory for your setup:

```python
def estimate_memory(model_params_billions, seqlen=2048, nsamples=128):
    """
    Estimate GPU memory requirements in GB

    Args:
        model_params_billions: Model size in billions (e.g., 1.2 for LLaMA 1B)
        seqlen: Sequence length
        nsamples: Number of calibration samples

    Returns:
        Estimated memory in GB
    """
    # Model weights in fp16
    model_memory = model_params_billions * 2

    # Activation memory (empirical formula)
    activation_memory = model_params_billions * 2 * (seqlen / 2048)

    # Overhead
    overhead = 1.0

    # Total
    total = model_memory + activation_memory + overhead

    return {
        'model': model_memory,
        'activations': activation_memory,
        'overhead': overhead,
        'total': total,
        'recommended_gpu': total * 1.5  # 50% headroom
    }

# Example
result = estimate_memory(1.2, seqlen=2048)
print(f"Estimated total: {result['total']:.1f} GB")
print(f"Recommended GPU: {result['recommended_gpu']:.1f} GB")
```

---

## Troubleshooting OOM Errors

### Error: `CUDA out of memory`

**Solution 1:** Reduce sequence length
```bash
python main.py --seqlen 1024  # or 512
```

**Solution 2:** Reduce number of samples
```bash
python main.py --nsamples 32
```

**Solution 3:** Enable memory fragmentation handling
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py ...
```

**Solution 4:** Check what's using GPU memory
```bash
nvidia-smi
# Kill other processes using GPU
```

**Solution 5:** Use gradient checkpointing (modify code)
```python
model.gradient_checkpointing_enable()
```

---

## Comparison: WikiText2 vs C4

| Aspect | WikiText2 | C4 |
|--------|-----------|-----|
| Dataset size | ~2 MB | ~500 GB |
| Download time | Seconds | Hours |
| Memory impact | Negligible | Moderate (+1-2 GB) |
| Pruning quality | Good | Slightly better |
| Use case | Quick experiments | Paper reproduction |

**Recommendation:** Always start with WikiText2 for development and testing.

---

## Hardware Recommendations

### For Development/Testing (LLaMA 1B-3B)
- **Minimum:** RTX 3060 (12 GB)
- **Recommended:** RTX 3090 (24 GB) or RTX 4090 (24 GB)
- **Optimal:** A5000 (24 GB) or A100 (40 GB)

### For Research (LLaMA 7B-13B)
- **Minimum:** RTX A6000 (48 GB)
- **Recommended:** A100 (40-80 GB)
- **Optimal:** H100 (80 GB)

### For Large Models (LLaMA 30B-70B)
- **Required:** Multi-GPU setup (2-4 × A100 80GB)
- **Or:** Cluster with distributed pruning

---

## Best Practices

1. **Always check available memory first:**
   ```bash
   nvidia-smi
   ```

2. **Start with conservative settings:**
   ```bash
   --dataset wikitext2 --nsamples 32 --seqlen 1024
   ```

3. **Monitor memory during execution:**
   ```bash
   watch -n 1 nvidia-smi  # In another terminal
   ```

4. **Scale up gradually:**
   - Start with small model (1B)
   - Verify memory usage
   - Then try larger models

5. **Use the estimation formula** before running large experiments

---

## Summary

**Key Takeaways:**
- WikiText2 has **minimal memory impact** (~2 MB vs ~500 GB for C4)
- Main memory consumers: **Model weights** + **Activations**
- For 16GB GPU: Safe for LLaMA 1B-3B with default settings
- For 24GB GPU: Safe for LLaMA 7B with optimization
- For 40GB+ GPU: Safe for LLaMA 13B and above
- **Tune `--seqlen` and `--nsamples`** to fit your GPU

**Quick Decision Matrix:**
- 8-12 GB GPU → LLaMA 1B with `--seqlen 1024`
- 16 GB GPU → LLaMA 1B-3B with default settings
- 24 GB GPU → LLaMA 7B with `--seqlen 1024`
- 40-48 GB GPU → LLaMA 13B with default settings
- 80 GB GPU → LLaMA 30B with optimization
- Multi-GPU → LLaMA 70B
