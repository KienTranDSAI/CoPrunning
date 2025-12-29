import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Get project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
WANDA_LIB_PATH = ROOT_DIR / "wanda"
sys.path.append(str(WANDA_LIB_PATH))

from lib.data import get_loaders

class PerplexityEvaluator:
    """Evaluate perplexity of language models on various datasets"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def calculate_perplexity(self, dataset_name="wikitext2", seqlen=2048):
        """
        Calculate perplexity on a given dataset.

        Args:
            dataset_name: Dataset to use ('wikitext2' or 'c4')
            seqlen: Sequence length for evaluation

        Returns:
            Dictionary with perplexity and related metrics
        """
        print(f"\nEvaluating perplexity on {dataset_name}...")
        print(f"Sequence length: {seqlen}")

        # Load test data
        _, testloader = get_loaders(
            dataset_name,
            nsamples=0,  # We don't need training samples
            seed=0,
            seqlen=seqlen,
            tokenizer=self.tokenizer
        )

        # Calculate perplexity
        if dataset_name == "wikitext2":
            ppl = self._eval_ppl_wikitext(testloader, seqlen)
        else:  # c4
            ppl = self._eval_ppl_c4(testloader, seqlen)

        return {
            'perplexity': ppl,
            'dataset': dataset_name,
            'seqlen': seqlen
        }

    def _eval_ppl_wikitext(self, testenc, seqlen=2048):
        """Evaluate perplexity on WikiText dataset"""
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen

        print(f"Number of samples: {nsamples}")

        nlls = []
        with torch.no_grad():
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(self.device)

                outputs = self.model(batch, labels=batch)
                neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)

                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{nsamples} samples processed", end='\r')

        print()  # New line after progress
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    def _eval_ppl_c4(self, testenc, seqlen=2048):
        """Evaluate perplexity on C4 dataset"""
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen

        print(f"Number of samples: {nsamples}")

        nlls = []
        with torch.no_grad():
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(self.device)

                outputs = self.model(batch, labels=batch)
                neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)

                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{nsamples} samples processed", end='\r')

        print()  # New line after progress
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

def save_results(results, output_file):
    """Save perplexity results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()

    with open(output_path, 'w') as f:
        json.dump(results, indent=2, fp=f)

    print(f"\nResults saved to: {output_path}")

def print_results(results):
    """Pretty print results"""
    print("\n" + "=" * 80)
    print("PERPLEXITY EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {results['model_name']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Sequence Length: {results['seqlen']}")
    print(f"Perplexity: {results['perplexity']:.4f}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Calculate perplexity for language models')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name or path (HuggingFace model or local path)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4'],
                        help='Dataset to evaluate on (default: wikitext2)')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for evaluation (default: 2048)')
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                        help='Directory to cache model weights')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (default: results/perplexity_<model>_<dataset>.json)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set output file if not specified
    if args.output is None:
        model_name_safe = args.model.replace('/', '_').replace('.', '_')
        args.output = f'results/perplexity_{model_name_safe}_{args.dataset}.json'

    print("=" * 80)
    print("PERPLEXITY EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Sequence Length: {args.seqlen}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    print(f"Model loaded on device: {device}")

    # Create evaluator
    evaluator = PerplexityEvaluator(model, tokenizer, device)

    # Calculate perplexity
    result = evaluator.calculate_perplexity(args.dataset, args.seqlen)

    # Add model info to results
    result['model_name'] = args.model
    result['model_path'] = args.model
    result['device'] = str(device)

    # Print and save results
    print_results(result)
    save_results(result, args.output)

if __name__ == "__main__":
    main()
