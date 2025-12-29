"""
Dataset loading utilities for calibration and evaluation.

Standalone reimplementation of wanda/lib/data.py
"""

import random
import torch
from datasets import load_dataset


class TokenizerWrapper:
    """Wrapper for tokenized input IDs (for C4 validation data)."""

    def __init__(self, input_ids):
        self.input_ids = input_ids


def load_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load WikiText2 dataset for calibration and evaluation.

    WikiText2 is small (~2MB) and fast to load, making it ideal for quick experiments.

    Reference: wanda/lib/data.py:19-38

    Args:
        nsamples: Number of calibration samples
        seed: Random seed for reproducibility
        seqlen: Sequence length
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (trainloader, testenc)
        - trainloader: List of (input_ids, targets) tuples for calibration
        - testenc: Test data with .input_ids attribute for evaluation
    """
    print("Loading WikiText2 dataset...")

    # Load train and test splits
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Sample random sequences for calibration
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Mask all but last token for target
        trainloader.append((inp, tar))

    print(f"WikiText2 loaded: {nsamples} calibration samples")
    return trainloader, testenc


def load_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load C4 dataset for calibration and evaluation.

    C4 is large (hundreds of GB) so we use streaming mode to avoid downloading
    the entire dataset. This matches the paper setup for more accurate calibration.

    Reference: wanda/lib/data.py:41-86

    Args:
        nsamples: Number of calibration samples
        seed: Random seed for reproducibility
        seqlen: Sequence length
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (trainloader, valenc)
        - trainloader: List of (input_ids, targets) tuples for calibration
        - valenc: Validation data with .input_ids attribute for evaluation
    """
    print("Loading C4 dataset (streaming mode)...")

    # Load train and validation datasets in streaming mode
    traindata = load_dataset(
        'allenai/c4',
        'en',
        split='train',
        streaming=True
    )
    valdata = load_dataset(
        'allenai/c4',
        'en',
        split='validation',
        streaming=True
    )

    # Generate calibration samples from training set
    random.seed(seed)
    trainloader = []

    # Iterate through shuffled stream until we have enough samples
    for sample in traindata.shuffle(seed=seed, buffer_size=10000):
        trainenc = tokenizer(sample['text'], return_tensors='pt')

        # Only use samples longer than seqlen
        if trainenc.input_ids.shape[1] > seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

            if len(trainloader) >= nsamples:
                break

    # Prepare validation dataset (first 1100 samples)
    print("Preparing validation data...")
    val_samples = []
    for i, sample in enumerate(valdata):
        if i >= 1100:
            break
        val_samples.append(sample['text'])

    valenc = tokenizer(' '.join(val_samples), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    print(f"C4 loaded: {nsamples} calibration samples, {len(val_samples)} validation samples")
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Get appropriate data loaders based on dataset name.

    Reference: wanda/lib/data.py:89-93

    Args:
        name: Dataset name ('wikitext2' or 'c4')
        nsamples: Number of calibration samples
        seed: Random seed
        seqlen: Sequence length
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (trainloader, testloader/valloader)

    Raises:
        ValueError: If dataset name is not recognized
    """
    if 'wikitext2' in name:
        return load_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif 'c4' in name:
        return load_c4(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'wikitext2' or 'c4'.")
