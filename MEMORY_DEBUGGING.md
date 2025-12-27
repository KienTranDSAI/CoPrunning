# Memory Debugging Guide

This guide shows you how to track and debug memory usage during pruning experiments.

## Quick Fix for Your Issue

**Problem:** The model was trying to allocate 64 GB for calibration input because `model.seqlen` was set to 131072 (model's max position embeddings).

**Solution:** Now defaults to 2048. You can control it with `--seqlen`:

```bash
# Use default seqlen (2048) - FIXED!
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda/

# Or explicitly specify
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048 \
    --save out/llama_3.2_1b/unstructured/wanda/
```

---

## Memory Logging Feature

### Enable Memory Tracking

Add the `--log_memory` flag to track memory usage at each stage:

```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048 \
    --save out/llama_3.2_1b/unstructured/wanda/ \
    --log_memory
```

### What You'll See

```
================================================================================
Memory Checkpoint: 01_start
================================================================================
GPU Memory:
  Allocated: 0.00 GB
  Reserved:  0.00 GB
  Peak:      0.00 GB
  Free:      15.47 GB
  Total:     15.47 GB
CPU Memory:
  Used:      2.15 GB (5.4%)
================================================================================

================================================================================
Memory Checkpoint: 02_model_loaded
================================================================================
GPU Memory:
  Allocated: 2.30 GB
  Reserved:  2.50 GB
  Peak:      2.30 GB
  Free:      13.17 GB
  Total:     15.47 GB
CPU Memory:
  Used:      2.45 GB (6.1%)
================================================================================

... (more checkpoints) ...

================================================================================
MEMORY USAGE SUMMARY
================================================================================
Checkpoint                     GPU Alloc    GPU Peak     CPU Used
--------------------------------------------------------------------------------
01_start                            0.00 GB       0.00 GB       2.15 GB
02_model_loaded                     2.30 GB       2.30 GB       2.45 GB
03_before_pruning                   2.30 GB       2.30 GB       2.45 GB
04_after_pruning                    4.85 GB       5.20 GB       2.60 GB
05_before_eval                      4.85 GB       5.20 GB       2.60 GB
06_after_eval                       4.85 GB       5.50 GB       2.65 GB
09_end                              4.85 GB       5.50 GB       2.65 GB
================================================================================

Peak GPU Memory: 5.50 GB
Peak CPU Memory: 2.65 GB
================================================================================
```

### Memory Log CSV File

A CSV file is automatically saved to `{save_dir}/memory_log.csv`:

```csv
Timestamp,Checkpoint,GPU_Allocated_GB,GPU_Reserved_GB,GPU_Peak_GB,CPU_Used_GB,CPU_Percent
2025-12-28T00:15:30,01_start,0.000,0.000,0.000,2.150,5.40
2025-12-28T00:15:45,02_model_loaded,2.300,2.500,2.300,2.450,6.10
2025-12-28T00:15:46,03_before_pruning,2.300,2.500,2.300,2.450,6.10
2025-12-28T00:18:23,04_after_pruning,4.850,5.000,5.200,2.600,6.50
...
```

You can analyze this with pandas or Excel to identify memory spikes.

---

## Memory Checkpoints Explained

| Checkpoint | What's Happening | Expected Memory Increase |
|------------|------------------|-------------------------|
| `01_start` | Initial state | Baseline |
| `02_model_loaded` | Model loaded to GPU | +2-15 GB (depends on model size) |
| `03_before_pruning` | Ready to prune | Minimal |
| `04_after_pruning` | Pruning complete | +1-5 GB (activations cached) |
| `05_before_eval` | Ready to evaluate | Minimal |
| `06_after_eval` | Evaluation complete | +0.5-2 GB (evaluation data) |
| `07_before_save_model` | Ready to save | Minimal |
| `08_after_save_model` | Model saved | Minimal |
| `09_end` | Finished | Final state |

---

## Debugging OOM Errors

### Step 1: Check Current Memory Usage

Before running pruning:
```bash
nvidia-smi
```

Look for:
- Total memory available
- Memory already in use
- Free memory

### Step 2: Run with Memory Logging

```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048 \
    --save out/test/ \
    --log_memory
```

### Step 3: Analyze the Logs

Look for the checkpoint where memory spikes:

**Example Analysis:**
```
02_model_loaded:   2.30 GB  ← Model loaded fine
03_before_pruning: 2.30 GB  ← Still fine
04_after_pruning:  ERROR!   ← OOM happened during pruning
```

This tells you the OOM occurred during the pruning phase.

### Step 4: Apply Fixes

Based on where OOM occurred:

#### OOM at Model Loading (checkpoint 02)
```bash
# Model too large for GPU
# Solutions:
# 1. Use smaller model
# 2. Use multi-GPU
# 3. Use CPU offloading (very slow)
```

#### OOM During Pruning (checkpoint 04)
```bash
# Reduce sequence length
python main.py --seqlen 1024  # or 512

# Reduce calibration samples
python main.py --nsamples 64  # or 32

# Use both
python main.py --seqlen 1024 --nsamples 64
```

#### OOM During Evaluation (checkpoint 06)
```bash
# Use smaller evaluation batch
# Edit lib/eval.py to use smaller batch size
```

---

## Manual Memory Monitoring

### Quick Memory Check in Python

Add to your code:

```python
from lib.memory_utils import print_gpu_memory_summary

print_gpu_memory_summary()
```

Output:
```
================================================================================
GPU MEMORY STATUS
================================================================================
Allocated: 4.25 GB
Reserved:  4.50 GB
Peak:      5.10 GB
Free:      11.22 GB
Total:     15.47 GB
Usage:     27.5%
================================================================================
```

### Watch Memory in Real-Time

In a separate terminal:
```bash
watch -n 1 nvidia-smi
```

This updates every second showing:
- GPU utilization
- Memory usage
- Running processes

---

## Memory Optimization Tips

### 1. Reduce Sequence Length (Biggest Impact)

```bash
# Default (may cause OOM on 16GB GPU with larger models)
--seqlen 2048

# Reduced (safer)
--seqlen 1024

# Minimal (very safe)
--seqlen 512
```

**Impact on Pruning Quality:** Minimal for unstructured sparsity

### 2. Reduce Calibration Samples

```bash
# Default
--nsamples 128

# Reduced
--nsamples 64

# Minimal
--nsamples 32
```

**Impact on Pruning Quality:** Small reduction, usually acceptable

### 3. Use WikiText2 Instead of C4

```bash
# Fast and small
--dataset wikitext2

# Large and slow (avoid if memory constrained)
--dataset c4
```

**Impact:** Negligible - WikiText2 is tiny

### 4. Clear Cache Between Operations

The code already includes:
```python
torch.cuda.empty_cache()  # After each layer
```

### 5. Enable Memory Fragmentation Handling

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py ...
```

---

## Example: Debugging Your OOM Issue

**Original Command (Failed):**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda/
```

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 GiB.
```

**Root Cause:**
```python
# In lib/prune.py line 68:
inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
# model.seqlen was 131072 instead of 2048
# 128 * 131072 * 2048 * 2 bytes = 68 GB
```

**Fix Applied:**
- Set `model.seqlen = min(seqlen, model.config.max_position_embeddings)`
- Default `seqlen=2048` in arguments

**Now Works:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048 \
    --save out/llama_3.2_1b/unstructured/wanda/
```

**Memory Usage:**
```
128 * 2048 * 2048 * 2 bytes = 1.07 GB  ✅ Fits in 16GB GPU
```

---

## Calculating Expected Memory

### Formula

```python
# Model memory (fp16)
model_memory_gb = (num_parameters * 2) / (1024**3)

# Calibration input memory
calibration_memory_gb = (nsamples * seqlen * hidden_dim * 2) / (1024**3)

# Total estimated
total_gb = model_memory_gb + calibration_memory_gb + 1.0  # +1 GB overhead
```

### Example: LLaMA 1B

```python
# Parameters
num_params = 1.2e9
hidden_dim = 2048
nsamples = 128
seqlen = 2048

# Calculate
model_gb = (1.2e9 * 2) / (1024**3) = 2.24 GB
calib_gb = (128 * 2048 * 2048 * 2) / (1024**3) = 1.07 GB
total = 2.24 + 1.07 + 1.0 = 4.31 GB

# Fits in 16GB GPU? YES ✅
```

---

## Summary

**Key Commands:**
```bash
# Enable memory logging
--log_memory

# Control sequence length
--seqlen 2048

# Reduce calibration samples
--nsamples 64

# Use small dataset
--dataset wikitext2
```

**For 16GB GPU (LLaMA 1B-3B):**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048 \
    --nsamples 128 \
    --dataset wikitext2 \
    --save out/test/ \
    --log_memory
```

**Expected Memory:** ~5-6 GB total (safe for 16GB GPU)
