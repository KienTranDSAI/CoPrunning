import torch
from transformers import AutoModelForCausalLM
import argparse

def find_linear_layers(model, prefix=''):
    """
    Find all Linear layers in the model that can be pruned.

    Args:
        model: PyTorch model
        prefix: Current path prefix

    Returns:
        List of layer names
    """
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name:  # Skip empty names
                linear_layers.append(name)

    return linear_layers

def main():
    parser = argparse.ArgumentParser(description='List all prunable layers in the model')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Model name or path')
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                        help='Directory to cache model weights')

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print("=" * 80)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    print(f"Model loaded: {type(model).__name__}")
    print("=" * 80)

    # Find all linear layers
    linear_layers = find_linear_layers(model)

    print(f"\nFound {len(linear_layers)} Linear layers that can be pruned:\n")

    # Group by layer type
    from collections import defaultdict
    grouped = defaultdict(list)

    for layer_name in linear_layers:
        if 'self_attn' in layer_name:
            if 'q_proj' in layer_name:
                grouped['Attention Q Projection'].append(layer_name)
            elif 'k_proj' in layer_name:
                grouped['Attention K Projection'].append(layer_name)
            elif 'v_proj' in layer_name:
                grouped['Attention V Projection'].append(layer_name)
            elif 'o_proj' in layer_name:
                grouped['Attention Output Projection'].append(layer_name)
        elif 'mlp' in layer_name:
            if 'gate_proj' in layer_name:
                grouped['MLP Gate Projection'].append(layer_name)
            elif 'up_proj' in layer_name:
                grouped['MLP Up Projection'].append(layer_name)
            elif 'down_proj' in layer_name:
                grouped['MLP Down Projection'].append(layer_name)
        elif 'lm_head' in layer_name:
            grouped['Language Model Head'].append(layer_name)
        else:
            grouped['Other'].append(layer_name)

    # Print grouped layers
    for group_name, layers in sorted(grouped.items()):
        print(f"\n{group_name} ({len(layers)} layers):")
        print("-" * 80)
        for i, layer in enumerate(layers[:5]):  # Show first 5 examples
            print(f"  {layer}")
        if len(layers) > 5:
            print(f"  ... and {len(layers) - 5} more")

    # Print example usage commands
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE:")
    print("=" * 80)

    example_layers = [
        linear_layers[i] for i in [0, len(linear_layers)//4, len(linear_layers)//2]
        if i < len(linear_layers)
    ]

    for i, layer in enumerate(example_layers[:3], 1):
        print(f"\nExample {i}: Analyze {layer}")
        print(f"python visualize_layer_metrics.py --layer {layer}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
