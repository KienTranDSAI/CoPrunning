"""
Simplified standalone Wanda pruner for LLaMA models.

Main CLI entry point for running pruning experiments.

Reference: wanda/main.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add the custom/pruner directory to Python path
PRUNER_DIR = Path(__file__).parent.resolve()
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))

import torch
from transformers import AutoTokenizer

from utils.model_utils import load_model, check_sparsity
from utils.evaluator import PerplexityEvaluator
from pruning.wanda import WandaPruner
from pruning.sparsity_patterns import UnstructuredSparsity, NMSparsity
from recovery.weight_redistribution import (
    WeightRedistributor,
    InverseWandaStrategy
)


def main():
    parser = argparse.ArgumentParser(
        description="Simplified standalone Wanda pruner for LLaMA models"
    )

    # Model configuration
    parser.add_argument(
        '--model', type=str, default='meta-llama/Llama-3.2-1B',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--cache_dir', type=str, default='../../llm_weights',
        help='Directory to cache model weights'
    )
    parser.add_argument(
        '--seqlen', type=int, default=2048,
        help='Sequence length for calibration/evaluation'
    )

    # Pruning configuration
    parser.add_argument(
        '--sparsity_ratio', type=float, default=0.5,
        help='Target sparsity ratio (0.0-1.0)'
    )
    parser.add_argument(
        '--sparsity_type', type=str, default='unstructured',
        choices=['unstructured', '2:4', '4:8'],
        help='Sparsity pattern type'
    )

    # Calibration
    parser.add_argument(
        '--dataset', type=str, default='wikitext2',
        choices=['wikitext2', 'c4'],
        help='Calibration dataset (wikitext2=fast, c4=accurate)'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility'
    )

    # Weight Redistribution
    parser.add_argument(
        '--use_recovery', action='store_true',
        help='Apply weight redistribution after pruning'
    )
    parser.add_argument(
        '--inverse_wanda_update_fraction', type=float, default=0.3,
        help='Fraction of survivors to update (0.0-1.0, default: 0.3 = 30%%)'
    )
    parser.add_argument(
        '--inverse_wanda_max_relative_update', type=float, default=2.0,
        help='Max update relative to weight magnitude (default: 2.0)'
    )

    # Output
    parser.add_argument(
        '--save_model', type=str, default=None,
        help='Directory to save pruned model (optional)'
    )
    parser.add_argument(
        '--save_log', type=str, default=None,
        help='File to save pruning results log (optional)'
    )

    args = parser.parse_args()

    # Print configuration
    print("="*80)
    print("Simplified Standalone Wanda Pruner")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Sparsity: {args.sparsity_ratio:.1%} ({args.sparsity_type})")
    print(f"Calibration: {args.dataset} ({args.nsamples} samples)")
    print(f"Sequence length: {args.seqlen}")
    print("="*80)

    # Load model and tokenizer
    print("\n[1/6] Loading model and tokenizer...")
    model = load_model(args.model, args.cache_dir, args.seqlen)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Select sparsity pattern
    print(f"\n[2/6] Initializing {args.sparsity_type} sparsity pattern...")
    if args.sparsity_type == 'unstructured':
        sparsity_pattern = UnstructuredSparsity()
    elif args.sparsity_type == '2:4':
        # For 2:4, we must have sparsity_ratio = 0.5
        if abs(args.sparsity_ratio - 0.5) > 0.01:
            print(f"Warning: 2:4 sparsity requires sparsity_ratio=0.5, "
                  f"but got {args.sparsity_ratio}. Setting to 0.5.")
            args.sparsity_ratio = 0.5
        sparsity_pattern = NMSparsity(n=2, m=4)
    elif args.sparsity_type == '4:8':
        # For 4:8, we must have sparsity_ratio = 0.5
        if abs(args.sparsity_ratio - 0.5) > 0.01:
            print(f"Warning: 4:8 sparsity requires sparsity_ratio=0.5, "
                  f"but got {args.sparsity_ratio}. Setting to 0.5.")
            args.sparsity_ratio = 0.5
        sparsity_pattern = NMSparsity(n=4, m=8)

    # Evaluate original model perplexity
    print(f"\n[3/6] Evaluating original model perplexity...")
    evaluator = PerplexityEvaluator(model, tokenizer)
    original_ppl = evaluator.evaluate(dataset="wikitext2", device=device)
    print(f"Original model perplexity: {original_ppl:.2f}")

    # Create redistributor if enabled
    redistributor = None
    if args.use_recovery:
        print(f"\n[4/7] Initializing weight redistributor...")
        print(f"  Strategy: Inverse Wanda")
        print(f"  Update fraction: {args.inverse_wanda_update_fraction:.1%}")
        print(f"  Max relative update: {args.inverse_wanda_max_relative_update:.1f}x")

        strategy = InverseWandaStrategy(
            update_fraction=args.inverse_wanda_update_fraction,
            max_relative_update=args.inverse_wanda_max_relative_update
        )
        redistributor = WeightRedistributor(strategy)

    # Create pruner
    step_num = 5 if args.use_recovery else 4
    print(f"\n[{step_num}/{'7' if args.use_recovery else '6'}] Running Wanda pruning...")
    pruner = WandaPruner(model, tokenizer, device, redistributor)

    # Run pruning
    pruner.prune(
        sparsity_ratio=args.sparsity_ratio,
        sparsity_pattern=sparsity_pattern,
        nsamples=args.nsamples,
        dataset=args.dataset,
        seed=args.seed
    )

    # Check actual sparsity achieved
    step_num = 6 if args.use_recovery else 5
    print(f"\n[{step_num}/{'7' if args.use_recovery else '6'}] Checking sparsity...")
    actual_sparsity = check_sparsity(model)
    print(f"\nTarget sparsity: {args.sparsity_ratio:.4f}")
    print(f"Actual sparsity: {actual_sparsity:.4f}")

    # Evaluate pruned model perplexity
    step_num = 7 if args.use_recovery else 6
    print(f"\n[{step_num}/{'7' if args.use_recovery else '6'}] Evaluating pruned model perplexity...")
    ppl = evaluator.evaluate(dataset="wikitext2", device=device)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Sparsity type: {args.sparsity_type}")
    print(f"Target sparsity: {args.sparsity_ratio:.4f}")
    print(f"Actual sparsity: {actual_sparsity:.4f}")
    print(f"Original WikiText2 Perplexity: {original_ppl:.2f}")
    print(f"Pruned WikiText2 Perplexity: {ppl:.2f}")
    print(f"Perplexity Increase: {ppl - original_ppl:.2f} ({((ppl - original_ppl) / original_ppl * 100):.2f}%)")
    print("="*80)

    # Save results log
    if args.save_log:
        print(f"\nSaving results to {args.save_log}")
        os.makedirs(os.path.dirname(args.save_log), exist_ok=True)
        with open(args.save_log, 'w') as f:
            f.write("Wanda Pruning Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Sparsity type: {args.sparsity_type}\n")
            f.write(f"Target sparsity: {args.sparsity_ratio:.4f}\n")
            f.write(f"Actual sparsity: {actual_sparsity:.4f}\n")
            f.write(f"Calibration dataset: {args.dataset}\n")
            f.write(f"Calibration samples: {args.nsamples}\n")
            f.write(f"Sequence length: {args.seqlen}\n")
            f.write(f"\nPerplexity Results:\n")
            f.write(f"Original WikiText2 Perplexity: {original_ppl:.2f}\n")
            f.write(f"Pruned WikiText2 Perplexity: {ppl:.2f}\n")
            f.write(f"Perplexity Increase: {ppl - original_ppl:.2f} ({((ppl - original_ppl) / original_ppl * 100):.2f}%)\n")

    # Save pruned model
    if args.save_model:
        print(f"\nSaving pruned model to {args.save_model}")
        os.makedirs(args.save_model, exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("Model saved successfully!")

    print("\nDone!")


if __name__ == "__main__":
    main()
