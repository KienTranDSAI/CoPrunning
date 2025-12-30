"""
Model loading and utility functions.

Standalone reimplementation of core utilities from wanda/lib/prune.py and wanda/main.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def load_model(model_name, cache_dir="llm_weights", seqlen=2048):
    """
    Load a LLaMA model for pruning.

    Reference: wanda/main.py:16-26

    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-1B")
        cache_dir: Directory to cache model weights
        seqlen: Sequence length for calibration/evaluation

    Returns:
        Loaded model with custom seqlen attribute
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )

    # Set sequence length (capped by model's max position embeddings)
    model.seqlen = min(seqlen, model.config.max_position_embeddings)
    print(f"Model loaded. Sequence length: {model.seqlen}")

    return model


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find all layers of specified types in a module.

    Reference: wanda/lib/prune.py:11-30

    Args:
        module: PyTorch module to search
        layers: List of layer types to find (default: [nn.Linear])
        name: Current module name (used for recursion)

    Returns:
        Dictionary mapping layer names to layer objects
    """
    if type(module) in layers:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        full_name = name + '.' + name1 if name != '' else name1
        res.update(find_layers(child, layers=layers, name=full_name))

    return res


def check_sparsity(model):
    """
    Compute actual sparsity achieved in the model.

    Reference: wanda/lib/prune.py:32-56

    Args:
        model: Pruned model

    Returns:
        Overall sparsity ratio (fraction of zero weights)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device):
    """
    Extract calibration inputs by intercepting the first layer's input.

    Uses the "Catcher" pattern to capture inputs without full forward passes.

    Reference: wanda/lib/prune.py:58-95

    Args:
        model: The model to calibrate
        dataloader: Calibration data loader
        device: Device for computation

    Returns:
        Tuple of (inps, outs, attention_mask, position_ids)
        - inps: Input activations (nsamples, seqlen, hidden_size)
        - outs: Pre-allocated output buffer
        - attention_mask: Attention mask from forward pass
        - position_ids: Position IDs from forward pass
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Handle multi-GPU case: get device of embedding layer
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size),
                       dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # Catcher: intercept first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError  # Stop forward pass immediately

    # Replace first layer temporarily
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass  # Expected: Catcher raises ValueError to stop forward pass
    layers[0] = layers[0].module  # Restore original layer

    # Pre-allocate output buffer
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # If position_ids is None, create it (needed for newer transformers versions)
    if position_ids is None:
        position_ids = torch.arange(
            0, model.seqlen, dtype=torch.long, device=device
        ).unsqueeze(0).expand(1, -1)

    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids
