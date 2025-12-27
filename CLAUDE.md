# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoPrunning is a repository for pruning Large Language Models (LLMs) using the **Wanda** (Pruning by Weights and Activations) method and other pruning techniques. The project focuses on creating sparse models while maintaining performance, primarily targeting LLaMA-family models.

## Environment Setup

```bash
# Create conda environment
conda create -n prune_llm python=3.9
conda activate prune_llm

# Install PyTorch and dependencies
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
```

**Note**: There are known issues with the transformers library when loading LLaMA tokenizers. See https://github.com/huggingface/transformers/issues/22222

## Common Commands

### Running Pruning Experiments

All pruning commands should be run from the `wanda/` directory.

**Basic Wanda pruning (50% unstructured sparsity):**
```bash
cd wanda
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --dataset wikitext2 \
    --save out/llama_3.2_1b/unstructured/wanda/
```

**Note:** Use `--dataset wikitext2` (default, faster) for quick testing, or `--dataset c4` to match the paper setup.

**Structured N:M sparsity (2:4 pattern):**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_3.2_1b/2-4/wanda/
```

**Save pruned model weights:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda/ \
    --save_model out/llama_3.2_1b_pruned_model/
```

**Zero-shot evaluation:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda/ \
    --eval_zero_shot
```

### Running Batch Experiments

Pre-configured batch scripts are available in `wanda/scripts/`:
- `llama_7b.sh` - Run all pruning methods on LLaMA-7B
- `llama_13b.sh`, `llama_30b.sh`, `llama_65b.sh` - For larger models
- `ablate_weight_update.sh` - Ablation studies on weight update strategies

```bash
cd wanda
chmod +x scripts/llama_7b.sh
./scripts/llama_7b.sh
```

## Architecture

### Main Entry Points

- **`wanda/main.py`**: Primary script for pruning LLaMA-family models
  - Loads model and tokenizer
  - Applies pruning method (wanda/magnitude/sparsegpt)
  - Evaluates perplexity on WikiText
  - Optionally evaluates zero-shot tasks
  - Saves results and pruned model

- **`wanda/main_opt.py`**: Variant for pruning OPT models

### Core Library (`wanda/lib/`)

- **`prune.py`**: Core pruning implementations
  - `prune_wanda()`: Wanda pruning using weight magnitudes × activation norms
  - `prune_magnitude()`: Simple magnitude-based pruning
  - `prune_sparsegpt()`: SparseGPT pruning method
  - `prune_ablate()`: Ablation study variants with OBS weight updates
  - `prepare_calibration_input()`: Prepares calibration data for activation-aware pruning
  - `check_sparsity()`: Verifies actual sparsity achieved

- **`data.py`**: Dataset loading utilities for C4 and WikiText datasets

- **`eval.py`**: Evaluation functions
  - `eval_ppl()`: WikiText perplexity evaluation
  - `eval_zero_shot()`: Zero-shot task evaluation (requires modified lm-evaluation-harness)

- **`layerwrapper.py`**: `WrappedGPT` class for capturing activation statistics during forward passes

- **`sparsegpt.py`**: `SparseGPT` class implementing the SparseGPT algorithm

- **`ablate.py`**: `AblateGPT` class for weight update ablation studies

### Key Arguments

- `--model`: HuggingFace model identifier (e.g., `meta-llama/Llama-3.2-1B`)
- `--prune_method`: Pruning method - `wanda`, `magnitude`, `sparsegpt`, or ablation variants
- `--sparsity_ratio`: Target sparsity (0.0-1.0); must be 0.5 for N:M sparsity
- `--sparsity_type`: `unstructured`, `2:4`, or `4:8`
- `--dataset`: Calibration dataset - `wikitext2` (default, faster) or `c4` (paper setup)
- `--nsamples`: Number of calibration samples (default: 128)
- `--cache_dir`: Directory to cache model weights (default: `llm_weights`)
- `--save`: Directory to save pruning results/logs
- `--save_model`: Directory to save pruned model weights
- `--eval_zero_shot`: Enable zero-shot evaluation (requires modified lm-evaluation-harness)
- `--use_variant`: Use Wanda variant from appendix (adaptive thresholding)

### Pruning Flow

1. **Model Loading**: Load pre-trained model from HuggingFace with `torch.float16` and `device_map="auto"`
2. **Calibration**: Load calibration data (C4 dataset) and perform forward passes to capture activations
3. **Layer-wise Pruning**: Iterate through transformer layers:
   - For Wanda: Compute metric = |weight| × sqrt(activation_norm)
   - For magnitude: Compute metric = |weight|
   - For SparseGPT: Use second-order information with Hessian
   - Determine pruning mask based on metric and sparsity type
   - Zero out selected weights
4. **Evaluation**: Compute WikiText perplexity and optionally zero-shot task performance
5. **Saving**: Save results log and optionally save pruned model

### Additional Components

- **`image_classifiers/`**: Pruning for vision transformers (separate workflow)
- **`lora_ft/`**: LoRA fine-tuning scripts for pruned models
- **`dense_ft/`**: Dense fine-tuning with custom trainers

## Development Workflow

**Important**: This repository is used for code development and updates. After making code changes here, the code will be run on a GPU server. When modifying code, ensure it is compatible with GPU execution environments and consider the deployment context.

This is server config:
nvidia-smi   
Sat Dec 27 14:12:42 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |



## Important Notes

- For structured N:M sparsity, `sparsity_ratio` must be 0.5
- **Dataset choice**: Use `--dataset wikitext2` (default, ~2MB, fast) for quick experiments, or `--dataset c4` (hundreds of GB, slow download) to match the paper setup
- Calibration uses 128 samples by default
- Zero-shot evaluation requires downloading a modified version of EleutherAI LM Harness
- Multi-GPU support via `device_map="auto"` for 30B/65B/70B models
- Results are saved in text format with actual sparsity and perplexity
