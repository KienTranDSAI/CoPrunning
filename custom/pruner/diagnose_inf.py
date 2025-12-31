"""
Diagnostic script to understand why inf values appear in activation difference analysis.

This script helps identify the root cause of inf values in relative difference calculations.
"""

import torch
import sys
from pathlib import Path

# Add the custom/pruner directory to Python path
PRUNER_DIR = Path(__file__).parent.resolve()
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))


def diagnose_inf_causes():
    """
    Simulate the relative difference calculation and identify where inf values come from.
    """
    print("\n" + "="*80)
    print("DIAGNOSING INF VALUES IN RELATIVE DIFFERENCE CALCULATION")
    print("="*80)

    # Test with different scenarios
    scenarios = [
        ("Small denominator, normal numerator",
         torch.tensor([0.01, 0.001, 1e-7, 1e-8]),
         torch.tensor([2.5, 2.5, 2.5, 2.5])),

        ("Zero denominator with clamp",
         torch.tensor([0.0, 0.0, 0.0]),
         torch.tensor([0.1, 1.0, 2.5])),

        ("Normal values",
         torch.tensor([1.0, 2.0, 3.0]),
         torch.tensor([0.1, 0.2, 0.3])),
    ]

    for scenario_name, denominator, numerator in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  Denominator (|pre_activations|): {denominator.tolist()}")
        print(f"  Numerator (|diff|): {numerator.tolist()}")

        # Test with float32
        denom_f32 = denominator.float()
        num_f32 = numerator.float()
        clamped_f32 = torch.clamp(denom_f32, min=1e-6)
        rel_diff_f32 = num_f32 / clamped_f32

        print(f"\n  Float32:")
        print(f"    Clamped denominator: {clamped_f32.tolist()}")
        print(f"    Relative diff: {rel_diff_f32.tolist()}")
        print(f"    Has inf: {torch.isinf(rel_diff_f32).any().item()}")
        print(f"    Num inf: {torch.isinf(rel_diff_f32).sum().item()}")

        # Test with float16
        denom_f16 = denominator.half()
        num_f16 = numerator.half()
        clamped_f16 = torch.clamp(denom_f16, min=1e-6)
        rel_diff_f16 = num_f16 / clamped_f16

        print(f"\n  Float16:")
        print(f"    Clamped denominator: {clamped_f16.tolist()}")
        print(f"    Relative diff: {rel_diff_f16.tolist()}")
        print(f"    Has inf: {torch.isinf(rel_diff_f16).any().item()}")
        print(f"    Num inf: {torch.isinf(rel_diff_f16).sum().item()}")

    # Show float16 limits
    print("\n" + "="*80)
    print("FLOAT16 LIMITATIONS")
    print("="*80)
    print(f"Float16 max value: {torch.finfo(torch.float16).max}")
    print(f"Float16 min positive value: {torch.finfo(torch.float16).tiny}")
    print()

    # Example calculation that causes overflow
    print("Example: Dividing by very small values in float16")
    numerator = torch.tensor([2.594], dtype=torch.float16)  # Max diff from your output
    small_values = torch.tensor([1e-6, 1e-5, 1e-4, 0.01], dtype=torch.float16)

    print(f"Numerator: {numerator.item()}")
    print(f"Small denominators: {small_values.tolist()}")
    print()

    for i, denom in enumerate(small_values):
        result = numerator / denom
        print(f"  {numerator.item():.6f} / {denom.item():.6e} = {result.item()}")
        if torch.isinf(result):
            print(f"    ^^^ OVERFLOW TO INF!")

    # Demonstrate the actual issue
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    print()
    print("When computing relative difference with float16 tensors:")
    print("  rel_diff = |pre - post| / max(|pre|, 1e-6)")
    print()
    print("If |pre| is very close to zero (say 1e-7), it gets clamped to 1e-6.")
    print("Then if |pre - post| is, say, 2.5:")
    print("  rel_diff = 2.5 / 1e-6 = 2,500,000")
    print()
    print(f"But float16 max is only {torch.finfo(torch.float16).max:.0f}!")
    print("So the result overflows to inf.")
    print()
    print("The 4,362 inf elements are locations where:")
    print("  1. pre_activation[i] is very close to zero (< 1e-6)")
    print("  2. The absolute difference |pre - post| is non-trivial")
    print("  3. The division result exceeds float16 max (~65,504)")
    print()

    # Show solutions
    print("="*80)
    print("POTENTIAL SOLUTIONS")
    print("="*80)
    print()
    print("1. Use float32 for relative difference calculation")
    print("   - Convert tensors to float32 before division")
    print("   - More accurate, no overflow for reasonable values")
    print()
    print("2. Use a larger epsilon value (e.g., 1e-3)")
    print("   - Reduces the magnitude of relative differences")
    print("   - Less sensitive to near-zero values")
    print()
    print("3. Cap the relative difference at a maximum value")
    print("   - e.g., torch.clamp(rel_diff, max=1000.0)")
    print("   - Prevents inf but loses information about magnitude")
    print()
    print("4. Filter out elements where denominator is too small")
    print("   - Only compute rel_diff where |pre| > threshold")
    print("   - More interpretable statistics")
    print()


if __name__ == "__main__":
    diagnose_inf_causes()
