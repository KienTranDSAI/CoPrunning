import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import os

# Add wanda lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../wanda'))
from lib.data import get_loaders

class ActivationCapture:
    """Helper class to capture activations during forward pass"""
    def __init__(self):
        self.input_activations = None
        self.output_activations = None

    def capture_hook(self, module, input, output):
        """Hook function to capture input and output activations"""
        self.input_activations = input[0].detach()
        self.output_activations = output.detach()

def get_layer_by_name(model, layer_name):
    """Get a specific layer from the model by its full name"""
    parts = layer_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module

def compute_activation_norm(activations):
    """
    Compute the L2 norm of activations per output dimension.

    Args:
        activations: Tensor of shape [batch, seq_len, hidden_dim]

    Returns:
        Tensor of shape [hidden_dim] containing the norm for each output dimension
    """
    # Reshape to [batch * seq_len, hidden_dim]
    act_reshaped = activations.reshape(-1, activations.shape[-1])
    # Compute L2 norm for each output dimension
    act_norm = torch.norm(act_reshaped, p=2, dim=0)
    return act_norm

def visualize_layer_metrics(model, tokenizer, layer_name, device, nsamples=128):
    """
    Visualize weight magnitude, activation norm, and Wanda metric for a specific layer.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        layer_name: Full name of the layer to analyze (e.g., 'model.layers.0.self_attn.q_proj')
        device: Device to run on
        nsamples: Number of calibration samples
    """
    print(f"Analyzing layer: {layer_name}")
    print("=" * 80)

    # Get the target layer
    try:
        target_layer = get_layer_by_name(model, layer_name)
        print(f"Layer type: {type(target_layer).__name__}")
        print(f"Layer shape: {target_layer.weight.shape}")
    except AttributeError as e:
        print(f"Error: Could not find layer '{layer_name}'")
        print(f"Error details: {e}")
        return

    # Check if layer has weights
    if not hasattr(target_layer, 'weight'):
        print(f"Error: Layer '{layer_name}' does not have weights")
        return

    # Get weight matrix
    W = target_layer.weight.data
    print(f"Weight matrix shape: {W.shape}")

    # Compute weight magnitude
    W_mag = torch.abs(W).cpu().numpy()

    # Capture activations
    print("\nLoading calibration data...")
    dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)

    print("Capturing activations...")
    activation_capture = ActivationCapture()
    hook = target_layer.register_forward_hook(activation_capture.capture_hook)

    # Run forward passes to collect activations
    activation_norms = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= nsamples:
                break
            inputs = batch[0].to(device)
            _ = model(inputs)

            if activation_capture.input_activations is not None:
                # Compute activation norm for this batch
                act_norm = compute_activation_norm(activation_capture.input_activations)
                activation_norms.append(act_norm)

    hook.remove()

    # Average activation norms across all batches
    if len(activation_norms) > 0:
        avg_act_norm = torch.stack(activation_norms).mean(dim=0).cpu().numpy()
        print(f"Activation norm shape: {avg_act_norm.shape}")

        # Compute Wanda metric: |W| * sqrt(||activation||)
        # Broadcast activation norm to match weight shape
        act_norm_broadcast = np.sqrt(avg_act_norm).reshape(1, -1)
        wanda_metric = W_mag * act_norm_broadcast

        # Create visualizations
        create_visualizations(W_mag, avg_act_norm, wanda_metric, layer_name)
    else:
        print("Error: No activations were captured")

def create_visualizations(W_mag, act_norm, wanda_metric, layer_name):
    """Create comprehensive visualizations of the metrics"""

    fig = plt.figure(figsize=(20, 12))

    # Sanitize layer name for title
    layer_title = layer_name.replace('_', ' ').title()
    fig.suptitle(f'Layer Analysis: {layer_name}', fontsize=16, fontweight='bold')

    # 1. Weight Magnitude Heatmap
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(W_mag, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title('Weight Magnitude Heatmap', fontweight='bold')
    ax1.set_xlabel('Input Dimension')
    ax1.set_ylabel('Output Dimension')
    plt.colorbar(im1, ax=ax1, label='Magnitude')

    # 2. Weight Magnitude Histogram
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(W_mag.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Weight Magnitude Distribution', fontweight='bold')
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Weight Magnitude Statistics
    ax3 = plt.subplot(3, 3, 3)
    ax3.axis('off')
    stats_text = f"""
    Weight Magnitude Statistics:

    Mean:    {W_mag.mean():.6f}
    Std:     {W_mag.std():.6f}
    Min:     {W_mag.min():.6f}
    Max:     {W_mag.max():.6f}
    Median:  {np.median(W_mag):.6f}

    Shape: {W_mag.shape}
    Total params: {W_mag.size}
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Activation Norm Bar Plot (sample of dimensions)
    ax4 = plt.subplot(3, 3, 4)
    sample_size = min(50, len(act_norm))
    indices = np.linspace(0, len(act_norm)-1, sample_size, dtype=int)
    ax4.bar(range(sample_size), act_norm[indices], alpha=0.7, color='green', edgecolor='black')
    ax4.set_title(f'Activation Norm (sampled {sample_size} dims)', fontweight='bold')
    ax4.set_xlabel('Dimension Index (sampled)')
    ax4.set_ylabel('L2 Norm')
    ax4.grid(True, alpha=0.3)

    # 5. Activation Norm Histogram
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(act_norm, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax5.set_title('Activation Norm Distribution', fontweight='bold')
    ax5.set_xlabel('L2 Norm')
    ax5.set_ylabel('Frequency')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    # 6. Activation Norm Statistics
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    Activation Norm Statistics:

    Mean:    {act_norm.mean():.6f}
    Std:     {act_norm.std():.6f}
    Min:     {act_norm.min():.6f}
    Max:     {act_norm.max():.6f}
    Median:  {np.median(act_norm):.6f}

    Shape: {act_norm.shape}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 7. Wanda Metric Heatmap
    ax7 = plt.subplot(3, 3, 7)
    im7 = ax7.imshow(wanda_metric, aspect='auto', cmap='plasma', interpolation='nearest')
    ax7.set_title('Wanda Metric Heatmap\n(|Weight| × √||Activation||)', fontweight='bold')
    ax7.set_xlabel('Input Dimension')
    ax7.set_ylabel('Output Dimension')
    plt.colorbar(im7, ax=ax7, label='Wanda Metric')

    # 8. Wanda Metric Histogram
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(wanda_metric.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
    ax8.set_title('Wanda Metric Distribution', fontweight='bold')
    ax8.set_xlabel('Metric Value')
    ax8.set_ylabel('Frequency')
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)

    # 9. Wanda Metric Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    stats_text = f"""
    Wanda Metric Statistics:

    Mean:    {wanda_metric.mean():.6f}
    Std:     {wanda_metric.std():.6f}
    Min:     {wanda_metric.min():.6f}
    Max:     {wanda_metric.max():.6f}
    Median:  {np.median(wanda_metric):.6f}

    Shape: {wanda_metric.shape}
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()

    # Save figure
    save_name = layer_name.replace('.', '_')
    output_file = f'layer_analysis_{save_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    # Show plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize layer metrics for pruning analysis')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Model name or path')
    parser.add_argument('--layer', type=str, required=True,
                        help='Layer name to analyze (e.g., model.layers.0.self_attn.q_proj)')
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                        help='Directory to cache model weights')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples')

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print("=" * 80)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Visualize the specified layer
    visualize_layer_metrics(model, tokenizer, args.layer, device, args.nsamples)

if __name__ == "__main__":
    main()
