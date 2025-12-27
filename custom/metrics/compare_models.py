import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime
from tabulate import tabulate

# Get project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
WANDA_LIB_PATH = ROOT_DIR / "wanda"
sys.path.append(str(WANDA_LIB_PATH))

from lib.data import get_loaders
from lib.prune import check_sparsity

def calculate_perplexity(model, tokenizer, device, dataset_name="wikitext2", seqlen=2048):
    """Calculate perplexity for a model"""
    model.eval()

    # Load test data
    _, testloader = get_loaders(
        dataset_name,
        nsamples=0,
        seed=0,
        seqlen=seqlen,
        tokenizer=tokenizer
    )

    testenc = testloader.input_ids
    nsamples = testenc.numel() // seqlen

    nlls = []
    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            outputs = model(batch, labels=batch)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)

            if (i + 1) % 10 == 0:
                print(f"  Sample {i+1}/{nsamples}", end='\r')

    print()
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

def count_parameters(model):
    """Count total and non-zero parameters"""
    total_params = sum(p.numel() for p in model.parameters())

    # Count non-zero parameters
    non_zero_params = 0
    for p in model.parameters():
        non_zero_params += (p != 0).sum().item()

    return total_params, non_zero_params

def evaluate_model(model_path, tokenizer_path, dataset, seqlen, device, cache_dir):
    """Evaluate a single model and return metrics"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*80}")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    # Get model metrics
    total_params, non_zero_params = count_parameters(model)
    sparsity = 1.0 - (non_zero_params / total_params)
    model_size = get_model_size(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters: {non_zero_params:,}")
    print(f"Sparsity: {sparsity*100:.2f}%")
    print(f"Model size: {model_size:.2f} MB")

    # Calculate perplexity
    print(f"\nCalculating perplexity on {dataset}...")
    ppl = calculate_perplexity(model, tokenizer, device, dataset, seqlen)
    print(f"Perplexity: {ppl:.4f}")

    return {
        'model_path': model_path,
        'total_params': total_params,
        'non_zero_params': non_zero_params,
        'sparsity': sparsity,
        'model_size_mb': model_size,
        'perplexity': ppl,
        'dataset': dataset,
        'seqlen': seqlen
    }

def main():
    parser = argparse.ArgumentParser(description='Compare perplexity across multiple models')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of model paths to compare')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='Display names for models (optional)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4'],
                        help='Dataset to evaluate on (default: wikitext2)')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for evaluation (default: 2048)')
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                        help='Directory to cache model weights')
    parser.add_argument('--output', type=str, default='results/model_comparison.json',
                        help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set display names
    if args.names is None:
        args.names = [f"Model {i+1}" for i in range(len(args.models))]
    elif len(args.names) != len(args.models):
        print("Error: Number of names must match number of models")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Evaluate all models
    results = []
    for model_path, name in zip(args.models, args.names):
        # Use the same tokenizer as the model (or first model for pruned versions)
        tokenizer_path = args.models[0] if os.path.exists(os.path.join(model_path, 'config.json')) else model_path

        result = evaluate_model(model_path, tokenizer_path, args.dataset, args.seqlen, device, args.cache_dir)
        result['name'] = name
        results.append(result)

        # Clear GPU memory
        torch.cuda.empty_cache()

    # Create comparison table
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")

    table_data = []
    headers = ["Model", "Total Params", "Non-Zero Params", "Sparsity (%)", "Size (MB)", "Perplexity"]

    for result in results:
        table_data.append([
            result['name'],
            f"{result['total_params']:,}",
            f"{result['non_zero_params']:,}",
            f"{result['sparsity']*100:.2f}",
            f"{result['model_size_mb']:.2f}",
            f"{result['perplexity']:.4f}"
        ])

    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Calculate improvements relative to first model (baseline)
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("IMPROVEMENTS RELATIVE TO BASELINE (First Model)")
        print(f"{'='*80}")

        baseline = results[0]
        improvement_data = []
        improvement_headers = ["Model", "Sparsity Gain (%)", "Size Reduction (%)", "PPL Change", "PPL % Change"]

        for i, result in enumerate(results[1:], 1):
            sparsity_gain = (result['sparsity'] - baseline['sparsity']) * 100
            size_reduction = ((baseline['model_size_mb'] - result['model_size_mb']) / baseline['model_size_mb']) * 100
            ppl_change = result['perplexity'] - baseline['perplexity']
            ppl_pct_change = ((result['perplexity'] - baseline['perplexity']) / baseline['perplexity']) * 100

            improvement_data.append([
                result['name'],
                f"{sparsity_gain:+.2f}",
                f"{size_reduction:+.2f}",
                f"{ppl_change:+.4f}",
                f"{ppl_pct_change:+.2f}"
            ])

        print(tabulate(improvement_data, headers=improvement_headers, tablefmt='grid'))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'seqlen': args.seqlen,
        'models': results
    }

    with open(output_path, 'w') as f:
        json.dump(comparison_results, indent=2, fp=f)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
