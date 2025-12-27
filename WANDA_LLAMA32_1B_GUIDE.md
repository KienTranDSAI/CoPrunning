# Wanda Pruning Guide for Llama 3.2 1B

This guide provides instructions for running Wanda pruning on Llama 3.2 1B models.

## Prerequisites

### 1. Environment Setup

Create and activate a conda environment:
```bash
conda create -n prune_llm python=3.9
conda activate prune_llm
```

### 2. Install Dependencies

```bash
# Install PyTorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install other required packages
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
```

**Note**: There are known issues with the transformers library when loading LLaMA tokenizers. If you encounter problems, refer to [this GitHub issue](https://github.com/huggingface/transformers/issues/22222).

### 3. Model Access

Ensure you have access to the Llama 3.2 1B model on Hugging Face. You may need to:
- Accept the model's license agreement on Hugging Face
- Login to Hugging Face CLI: `huggingface-cli login`

## Quick Start

Navigate to the Wanda directory:
```bash
cd wanda
```

### Basic Wanda Pruning (50% Unstructured Sparsity)

```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda/
```

## Pruning Options

### 1. Unstructured Sparsity

Prune with different sparsity ratios:

**50% Sparsity:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda_50/
```

**60% Sparsity:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.6 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda_60/
```

**70% Sparsity:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda_70/
```

### 2. Structured N:M Sparsity

**2:4 Sparsity (50%):**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_3.2_1b/2-4/wanda/
```

**4:8 Sparsity (50%):**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --save out/llama_3.2_1b/4-8/wanda/
```

**Note:** For structured N:M sparsity, the sparsity ratio must be 0.5.

### 3. Saving the Pruned Model

To save the pruned model weights:
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda/ \
    --save_model out/llama_3.2_1b_pruned_model/
```

### 4. Zero-Shot Evaluation

To include zero-shot task evaluation (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA):
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/wanda/ \
    --eval_zero_shot
```

**Note:** Zero-shot evaluation requires a modified version of the EleutherAI LM Harness. Download it from [this link](https://drive.google.com/file/d/1zugbLyGZKsH1L19L9biHLfaGGFnEc7XL/view?usp=sharing).

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Hugging Face model identifier | Required |
| `--prune_method` | Pruning method: `wanda`, `magnitude`, `sparsegpt` | Required |
| `--sparsity_ratio` | Percentage of weights to prune (0.0 to 1.0) | 0.0 |
| `--sparsity_type` | Type of sparsity: `unstructured`, `2:4`, `4:8` | Required |
| `--cache_dir` | Directory to cache model weights | `llm_weights` |
| `--nsamples` | Number of calibration samples | 128 |
| `--seed` | Random seed for reproducibility | 0 |
| `--use_variant` | Use Wanda variant from appendix | False |
| `--save` | Directory to save pruning results/logs | None |
| `--save_model` | Directory to save pruned model | None |
| `--eval_zero_shot` | Perform zero-shot evaluation | False |

## Understanding the Output

After running the pruning command, you will see:

1. **Model Loading**: Confirmation of model and tokenizer loading
2. **Pruning Progress**: Layer-by-layer pruning information
3. **Sparsity Check**: Actual sparsity ratio achieved
4. **WikiText Perplexity**: Perplexity score on WikiText dataset (lower is better)
5. **Results File**: A log file saved in the specified `--save` directory

Example output:
```
pruning starts
...
******************************
sparsity sanity check 0.5000
******************************
wikitext perplexity 12.34
```

## Batch Script Example

Create a shell script to run multiple experiments:

```bash
#!/bin/bash

model="meta-llama/Llama-3.2-1B"
cuda_device=0

export CUDA_VISIBLE_DEVICES=$cuda_device

# Unstructured sparsity experiments
for sparsity in 0.5 0.6 0.7; do
    echo "Running Wanda with ${sparsity} unstructured sparsity"
    python main.py \
        --model $model \
        --prune_method wanda \
        --sparsity_ratio $sparsity \
        --sparsity_type unstructured \
        --save out/llama_3.2_1b/unstructured/wanda_${sparsity}/
done

# Structured sparsity experiments
for sparsity_type in "2:4" "4:8"; do
    echo "Running Wanda with ${sparsity_type} structured sparsity"
    python main.py \
        --model $model \
        --prune_method wanda \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --save out/llama_3.2_1b/${sparsity_type}/wanda/
done
```

Save this as `llama_3.2_1b_wanda.sh` and run with:
```bash
chmod +x llama_3.2_1b_wanda.sh
./llama_3.2_1b_wanda.sh
```

## Comparison with Other Methods

You can compare Wanda with magnitude pruning and SparseGPT:

**Magnitude Pruning:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method magnitude \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/magnitude/
```

**SparseGPT:**
```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_3.2_1b/unstructured/sparsegpt/
```

## Tips and Best Practices

1. **GPU Memory**: Llama 3.2 1B should fit comfortably on a single GPU with 16GB+ VRAM
2. **Reproducibility**: Set `--seed` to a fixed value for reproducible results
3. **Calibration Samples**: The default 128 samples (`--nsamples 128`) works well; increasing may improve quality slightly
4. **Start Conservative**: Begin with 50% sparsity and increase gradually
5. **Results Directory**: Organize your experiments with descriptive save paths

## Troubleshooting

**Issue**: CUDA out of memory
- Solution: Ensure no other processes are using GPU memory

**Issue**: Tokenizer loading errors
- Solution: Update transformers or follow [this guide](https://github.com/huggingface/transformers/issues/22222)

**Issue**: Model download fails
- Solution: Check Hugging Face authentication and internet connection

## Citation

If you use Wanda in your research, please cite:

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models},
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

## Additional Resources

- [Wanda Paper](https://arxiv.org/abs/2306.11695)
- [Project Page](https://eric-mingjie.github.io/wanda/home.html)
- [Original Wanda Repository](https://github.com/locuslab/wanda)
