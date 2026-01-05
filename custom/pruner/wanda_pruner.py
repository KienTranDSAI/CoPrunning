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
        '--model', type=str, default='openbmb/MiniCPM-2B-sft-bf16',
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
    print(f"\n[3/{'7' if args.use_recovery else '6'}] Evaluating original model perplexity...")
    evaluator = PerplexityEvaluator(model, tokenizer)
    original_ppl = evaluator.evaluate(dataset="wikitext2", device=device)
    print(f"Original model perplexity: {original_ppl:.2f}")

    # Initialize perplexity variables
    pruned_ppl = None
    recovered_ppl = None
    recovery_stats = None

    if args.use_recovery:
        # ================================================================
        # STAGE 1: Prune WITHOUT recovery to measure pruning-only impact
        # ================================================================
        print(f"\n[4/7] Running Wanda pruning WITHOUT recovery...")
        pruner = WandaPruner(model, tokenizer, device, redistributor=None)
        pruner.prune(
            sparsity_ratio=args.sparsity_ratio,
            sparsity_pattern=sparsity_pattern,
            nsamples=args.nsamples,
            dataset=args.dataset,
            seed=args.seed
        )

        # Check sparsity after pruning
        print(f"\n[5/7] Checking sparsity after pruning...")
        actual_sparsity = check_sparsity(model)
        print(f"Actual sparsity: {actual_sparsity:.4f}")

        # Evaluate pruned-only model
        print(f"\n[6/7] Evaluating PRUNED model (before recovery)...")
        pruned_ppl = evaluator.evaluate(dataset="wikitext2", device=device)
        print(f"Pruned model perplexity: {pruned_ppl:.2f}")
        print(f"Perplexity increase from pruning: {pruned_ppl - original_ppl:.2f}")

        # ================================================================
        # STAGE 2: Apply recovery to the SAME pruned model (no re-pruning!)
        # ================================================================
        print(f"\n[7/7] Applying weight redistribution to pruned model...")
        print(f"  Strategy: Inverse Wanda")
        print(f"  Update fraction: {args.inverse_wanda_update_fraction:.1%}")
        print(f"  Max relative update: {args.inverse_wanda_max_relative_update:.1f}x")

        strategy = InverseWandaStrategy(
            update_fraction=args.inverse_wanda_update_fraction,
            max_relative_update=args.inverse_wanda_max_relative_update
        )
        redistributor = WeightRedistributor(strategy)

        # Apply recovery using stored pruning state (no re-pruning!)
        recovery_stats = pruner.apply_recovery(redistributor)

        # Evaluate recovered model
        print(f"\nEvaluating RECOVERED model (after recovery)...")
        recovered_ppl = evaluator.evaluate(dataset="wikitext2", device=device)
        print(f"Recovered model perplexity: {recovered_ppl:.2f}")
        print(f"Recovery improvement: {pruned_ppl - recovered_ppl:.2f}")

        ppl = recovered_ppl  # Final perplexity

    else:
        # Standard pruning without recovery
        print(f"\n[4/6] Running Wanda pruning...")
        pruner = WandaPruner(model, tokenizer, device, redistributor=None)
        pruner.prune(
            sparsity_ratio=args.sparsity_ratio,
            sparsity_pattern=sparsity_pattern,
            nsamples=args.nsamples,
            dataset=args.dataset,
            seed=args.seed
        )

        print(f"\n[5/6] Checking sparsity...")
        actual_sparsity = check_sparsity(model)
        print(f"Actual sparsity: {actual_sparsity:.4f}")

        print(f"\n[6/6] Evaluating pruned model...")
        ppl = evaluator.evaluate(dataset="wikitext2", device=device)
        pruned_ppl = ppl

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Sparsity type: {args.sparsity_type}")
    print(f"Target sparsity: {args.sparsity_ratio:.4f}")
    print(f"Actual sparsity: {actual_sparsity:.4f}")
    print()
    print("Perplexity Results:")
    print(f"  [1] Original:        {original_ppl:.2f}")

    if args.use_recovery:
        print(f"  [2] After Pruning:   {pruned_ppl:.2f}  (+{pruned_ppl - original_ppl:.2f}, +{((pruned_ppl - original_ppl) / original_ppl * 100):.2f}%)")
        print(f"  [3] After Recovery:  {recovered_ppl:.2f}  (+{recovered_ppl - original_ppl:.2f}, +{((recovered_ppl - original_ppl) / original_ppl * 100):.2f}%)")
        print()
        print(f"Recovery Impact:")
        print(f"  Perplexity recovered: {pruned_ppl - recovered_ppl:.2f} points")
        print(f"  ({((pruned_ppl - recovered_ppl) / (pruned_ppl - original_ppl) * 100):.1f}% of pruning degradation)")

        if recovery_stats:
            before = recovery_stats.get('before_recovery', {})
            after = recovery_stats.get('after_recovery', {})

            print()
            print(f"Error Statistics (Before Recovery):")
            print(f"  Mean activation error: {before.get('mean_relative_error', 0):.6f}")
            print(f"  Max activation error:  {before.get('max_relative_error', 0):.6f}")
            print(f"  Sum of errors:         {before.get('sum_of_errors', 0):.6f}")
            print(f"  Layers processed:      {before.get('num_layers', 0)}")

            print()
            print(f"Error Statistics (After Recovery):")
            print(f"  Mean activation error: {after.get('mean_relative_error', 0):.6f}")
            print(f"  Max activation error:  {after.get('max_relative_error', 0):.6f}")
            print(f"  Sum of errors:         {after.get('sum_of_errors_after', 0):.6f}")
            print(f"  Sum of updated error:  {after.get('sum_of_updated_error', 0):.6f}")
            print(f"  Weights updated:       {after.get('total_weights_updated', 0):,}")
            print(f"  Layers processed:      {after.get('num_layers', 0)}")
            print(f"  Mean max update:       {after.get('mean_max_update', 0):.6f}")

            # Show improvement
            if before.get('mean_relative_error') and after.get('mean_relative_error'):
                improvement = before['mean_relative_error'] - after['mean_relative_error']
                improvement_pct = (improvement / before['mean_relative_error']) * 100
                print()
                print(f"Recovery Improvement:")
                print(f"  Mean error reduced by: {improvement:.6f} ({improvement_pct:.2f}%)")

                # Show sum of errors improvement
                if before.get('sum_of_errors') and after.get('sum_of_errors_after'):
                    sum_improvement = before['sum_of_errors'] - after['sum_of_errors_after']
                    sum_improvement_pct = (sum_improvement / before['sum_of_errors']) * 100
                    print(f"  Sum of errors reduced by: {sum_improvement:.6f} ({sum_improvement_pct:.2f}%)")
    else:
        print(f"  [2] After Pruning:   {ppl:.2f}  (+{ppl - original_ppl:.2f}, +{((ppl - original_ppl) / original_ppl * 100):.2f}%)")

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
            f.write(f"  [1] Original:       {original_ppl:.2f}\n")

            if args.use_recovery:
                f.write(f"  [2] After Pruning:  {pruned_ppl:.2f}  (+{pruned_ppl - original_ppl:.2f})\n")
                f.write(f"  [3] After Recovery: {recovered_ppl:.2f}  (+{recovered_ppl - original_ppl:.2f})\n")
                f.write(f"\nRecovery Impact:\n")
                f.write(f"  Perplexity recovered: {pruned_ppl - recovered_ppl:.2f} points\n")
                f.write(f"  ({((pruned_ppl - recovered_ppl) / (pruned_ppl - original_ppl) * 100):.1f}% of pruning degradation)\n")

                if recovery_stats:
                    before = recovery_stats.get('before_recovery', {})
                    after = recovery_stats.get('after_recovery', {})

                    f.write(f"\nError Statistics (Before Recovery):\n")
                    f.write(f"  Mean activation error: {before.get('mean_relative_error', 0):.6f}\n")
                    f.write(f"  Max activation error:  {before.get('max_relative_error', 0):.6f}\n")
                    f.write(f"  Sum of errors:         {before.get('sum_of_errors', 0):.6f}\n")
                    f.write(f"  Layers processed:      {before.get('num_layers', 0)}\n")

                    f.write(f"\nError Statistics (After Recovery):\n")
                    f.write(f"  Mean activation error: {after.get('mean_relative_error', 0):.6f}\n")
                    f.write(f"  Max activation error:  {after.get('max_relative_error', 0):.6f}\n")
                    f.write(f"  Sum of errors:         {after.get('sum_of_errors_after', 0):.6f}\n")
                    f.write(f"  Sum of updated error:  {after.get('sum_of_updated_error', 0):.6f}\n")
                    f.write(f"  Weights updated:       {after.get('total_weights_updated', 0):,}\n")
                    f.write(f"  Layers processed:      {after.get('num_layers', 0)}\n")
                    f.write(f"  Mean max update:       {after.get('mean_max_update', 0):.6f}\n")

                    if before.get('mean_relative_error') and after.get('mean_relative_error'):
                        improvement = before['mean_relative_error'] - after['mean_relative_error']
                        improvement_pct = (improvement / before['mean_relative_error']) * 100
                        f.write(f"\nRecovery Improvement:\n")
                        f.write(f"  Mean error reduced by: {improvement:.6f} ({improvement_pct:.2f}%)\n")

                        # Show sum of errors improvement
                        if before.get('sum_of_errors') and after.get('sum_of_errors_after'):
                            sum_improvement = before['sum_of_errors'] - after['sum_of_errors_after']
                            sum_improvement_pct = (sum_improvement / before['sum_of_errors']) * 100
                            f.write(f"  Sum of errors reduced by: {sum_improvement:.6f} ({sum_improvement_pct:.2f}%)\n")
            else:
                f.write(f"  [2] After Pruning:  {ppl:.2f}  (+{ppl - original_ppl:.2f})\n")

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
