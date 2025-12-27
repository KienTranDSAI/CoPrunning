import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict

def get_all_layer_types(module, prefix='', layer_info=None):
    """
    Recursively traverse the model and collect all layer types with their counts and example paths.

    Args:
        module: PyTorch module to analyze
        prefix: Current path prefix in the model hierarchy
        layer_info: Dictionary to store layer type information

    Returns:
        Dictionary with layer type as key and list of paths as value
    """
    if layer_info is None:
        layer_info = defaultdict(list)

    # Get the type of current module
    module_type = type(module).__name__

    # Add current module to the dictionary
    if prefix:
        layer_info[module_type].append(prefix)

    # Recursively process children
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        get_all_layer_types(child, child_prefix, layer_info)

    return layer_info

def main():
    print("Loading meta-llama/Llama-3.2-1B model...")
    print("=" * 80)

    # Load model with minimal memory usage
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print(f"Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print("=" * 80)

    # Get all layer types
    layer_info = get_all_layer_types(model)

    # Print summary of unique layer types
    print("\nUNIQUE LAYER TYPES IN MODEL:")
    print("=" * 80)

    # Sort by layer type name
    for layer_type in sorted(layer_info.keys()):
        count = len(layer_info[layer_type])
        print(f"\n{layer_type}: {count} instances")

        # Show first few examples
        examples = layer_info[layer_type][:3]
        for example in examples:
            print(f"  - {example}")

        if count > 3:
            print(f"  ... and {count - 3} more")

    print("\n" + "=" * 80)
    print(f"Total unique layer types: {len(layer_info)}")
    print("=" * 80)

    # Print detailed breakdown of specific important layers
    print("\nDETAILED BREAKDOWN OF KEY LAYERS:")
    print("=" * 80)

    important_types = ['Linear', 'Embedding', 'RMSNorm', 'LayerNorm', 'Dropout']
    for layer_type in important_types:
        if layer_type in layer_info:
            print(f"\n{layer_type} layers ({len(layer_info[layer_type])} total):")
            for path in layer_info[layer_type]:
                print(f"  {path}")

if __name__ == "__main__":
    main()
