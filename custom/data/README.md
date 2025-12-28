# Dataset Viewer

Simple utility to view samples from calibration datasets.

## Usage

### View WikiText2 samples
```bash
python view_dataset_samples.py --dataset wikitext2 --n_samples 5
```

### View C4 samples
```bash
python view_dataset_samples.py --dataset c4 --n_samples 3
```

### View both datasets
```bash
python view_dataset_samples.py --dataset both --n_samples 3
```

### View tokenized samples
```bash
# WikiText2 with LLaMA tokenizer
python view_dataset_samples.py --dataset wikitext2 --tokenized --n_samples 2

# C4 with custom model tokenizer
python view_dataset_samples.py --dataset c4 --tokenized --model meta-llama/Llama-3.2-1B
```

## Options

- `--dataset`: Choose `wikitext2`, `c4`, or `both`
- `--n_samples`: Number of samples to display (default: 3)
- `--tokenized`: Show tokenized version with token IDs
- `--model`: Model name for tokenizer (default: meta-llama/Llama-3.2-1B)

## Output

- Raw text samples with character counts
- For C4: includes URL and timestamp metadata
- With `--tokenized`: shows token IDs and decoded tokens
