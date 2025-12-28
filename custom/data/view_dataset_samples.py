"""
Simple script to view samples from calibration datasets (WikiText2, C4)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../wanda'))

from datasets import load_dataset
from transformers import AutoTokenizer

def view_wikitext2_samples(n_samples=3):
    """View samples from WikiText2 dataset"""
    print("=" * 80)
    print("WikiText2 Dataset Samples")
    print("=" * 80)

    # Load dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    print(f"\nTrain size: {len(traindata)} samples")
    print(f"Test size: {len(testdata)} samples")

    print(f"\n--- First {n_samples} Training Samples ---")
    for i in range(min(n_samples, len(traindata))):
        text = traindata[i]['text']
        if text.strip():  # Only show non-empty samples
            print(f"\n[Sample {i}]")
            print(f"Length: {len(text)} chars")
            print(f"Text: {text}")  # First 300 chars

    print(f"\n--- First {n_samples} Test Samples ---")
    for i in range(min(n_samples, len(testdata))):
        text = testdata[i]['text']
        if text.strip():
            print(f"\n[Sample {i}]")
            print(f"Length: {len(text)} chars")
            print(f"Text: {text}")

def view_c4_samples(n_samples=3):
    """View samples from C4 dataset"""
    print("\n" + "=" * 80)
    print("C4 Dataset Samples")
    print("=" * 80)

    # Load dataset (only one file for speed)
    print("\nLoading C4 dataset (this may take a while)...")
    traindata = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train'
    )

    print(f"Train size: {len(traindata)} samples")

    print(f"\n--- First {n_samples} Training Samples ---")
    for i in range(min(n_samples, len(traindata))):
        text = traindata[i]['text']
        url = traindata[i].get('url', 'N/A')
        timestamp = traindata[i].get('timestamp', 'N/A')

        print(f"\n[Sample {i}]")
        print(f"URL: {url}")
        print(f"Timestamp: {timestamp}")
        print(f"Length: {len(text)} chars")
        print(f"Text: {text}")

def view_tokenized_samples(dataset_name='wikitext2', model_name='meta-llama/Llama-3.2-1B', n_samples=2):
    """View tokenized samples with a specific tokenizer"""
    print("\n" + "=" * 80)
    print(f"Tokenized Samples - {dataset_name.upper()} with {model_name}")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Load dataset
    if dataset_name == 'wikitext2':
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    elif dataset_name == 'c4':
        print("Loading C4 dataset...")
        data = load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train'
        )

    # Show tokenized samples
    print(f"\n--- Tokenized Samples ---")
    for i in range(min(n_samples, len(data))):
        text = data[i]['text']
        if not text.strip():
            continue

        # Tokenize
        tokens = tokenizer(text, return_tensors='pt')
        input_ids = tokens.input_ids[0]

        print(f"\n[Sample {i}]")
        print(f"Original text ({len(text)} chars):")
        print(f"{text}")
        print(f"\nTokenized ({len(input_ids)} tokens):")
        print(f"Token IDs: {input_ids[:50].tolist()}...")  # First 50 tokens
        print(f"\nDecoded tokens (first 20):")
        for j in range(min(20, len(input_ids))):
            token_id = input_ids[j].item()
            token_text = tokenizer.decode([token_id])
            print(f"  {j:3d}: {token_id:6d} -> '{token_text}'")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='View dataset samples')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'both'],
                       help='Which dataset to view')
    parser.add_argument('--n_samples', type=int, default=3,
                       help='Number of samples to display')
    parser.add_argument('--tokenized', action='store_true',
                       help='Show tokenized samples')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                       help='Model for tokenizer (if --tokenized)')
    args = parser.parse_args()

    # View raw samples
    if args.dataset in ['wikitext2', 'both']:
        view_wikitext2_samples(args.n_samples)

    if args.dataset in ['c4', 'both']:
        view_c4_samples(args.n_samples)

    # View tokenized samples
    if args.tokenized:
        if args.dataset == 'both':
            view_tokenized_samples('wikitext2', args.model, args.n_samples)
            view_tokenized_samples('c4', args.model, args.n_samples)
        else:
            view_tokenized_samples(args.dataset, args.model, args.n_samples)

if __name__ == '__main__':
    main()
