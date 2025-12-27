import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import os
from pathlib import Path

# Get project root directory (CoPrunning/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "assets" / "layer_analysis"

# Add wanda lib to path
WANDA_LIB_PATH = ROOT_DIR / "wanda"
sys.path.append(str(WANDA_LIB_PATH))
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

def visualize_layer_metrics(model, tokenizer, layer_name, device, nsamples=128, dataset="wikitext2", seqlen=2048, output_dir=None):
    """
    Visualize weight magnitude, activation norm, and Wanda metric for a specific layer.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        layer_name: Full name of the layer to analyze (e.g., 'model.layers.0.self_attn.q_proj')
        device: Device to run on
        nsamples: Number of calibration samples
        dataset: Dataset to use for calibration ('wikitext2' or 'c4')
        seqlen: Sequence length for calibration (default: 2048)
        output_dir: Directory to save visualizations (default: ROOT_DIR/assets/layer_analysis)
    """
    if output_dir is None:
        output_dir = ASSETS_DIR
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
    print(f"\nLoading calibration data from {dataset}...")
    print(f"Using sequence length: {seqlen}")
    print("(This may take a few moments on first run - dataset will be cached)")
    dataloader, _ = get_loaders(dataset, nsamples=nsamples, seed=0, seqlen=seqlen, tokenizer=tokenizer)
    print(f"Loaded {len(dataloader)} samples")

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

            print(f"Processing sample {i+1}/{nsamples}...", end='\r')

            inputs = batch[0].to(device)
            _ = model(inputs)

            if activation_capture.input_activations is not None:
                # Compute activation norm for this batch and move to CPU immediately
                act_norm = compute_activation_norm(activation_capture.input_activations)
                activation_norms.append(act_norm.cpu())

                # Clear activation cache to free GPU memory
                activation_capture.input_activations = None
                activation_capture.output_activations = None

            # Delete inputs and clear GPU cache
            del inputs
            torch.cuda.empty_cache()

    print()  # New line after progress
    hook.remove()

    # Average activation norms across all batches
    if len(activation_norms) > 0:
        avg_act_norm = torch.stack(activation_norms).mean(dim=0).cpu().numpy()
        print(f"Activation norm shape: {avg_act_norm.shape}")

        # Compute Wanda metric: |W| * sqrt(||activation||)
        # Broadcast activation norm to match weight shape
        act_norm_broadcast = np.sqrt(avg_act_norm).reshape(1, -1)

        # Use float64 for better numerical stability
        W_mag_64 = W_mag.astype(np.float64)
        act_norm_broadcast_64 = act_norm_broadcast.astype(np.float64)
        wanda_metric = W_mag_64 * act_norm_broadcast_64

        print(f"Wanda metric range: [{wanda_metric.min():.2e}, {wanda_metric.max():.2e}]")

        # Create visualizations
        create_visualizations(W_mag, avg_act_norm, wanda_metric, layer_name, output_dir)
    else:
        print("Error: No activations were captured")

def safe_stats(data):
    """Compute statistics safely, handling inf/nan values"""
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

    return {
        'mean': np.mean(finite_data),
        'std': np.std(finite_data),
        'min': np.min(finite_data),
        'max': np.max(finite_data),
        'median': np.median(finite_data)
    }

def create_visualizations(W_mag, act_norm, wanda_metric, layer_name, output_dir):
    """Create comprehensive visualizations of the metrics"""

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for inf/nan values and replace them for visualization
    num_invalid = np.sum(~np.isfinite(wanda_metric))
    if num_invalid > 0:
        print(f"Warning: Found {num_invalid} inf/nan values in Wanda metric ({100*num_invalid/wanda_metric.size:.2f}%)")
        wanda_metric = np.nan_to_num(wanda_metric, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)

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
    w_stats = safe_stats(W_mag.flatten())
    stats_text = f"""
    Weight Magnitude Statistics:

    Mean:    {w_stats['mean']:.6f}
    Std:     {w_stats['std']:.6f}
    Min:     {w_stats['min']:.6f}
    Max:     {w_stats['max']:.6f}
    Median:  {w_stats['median']:.6f}

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
    a_stats = safe_stats(act_norm.flatten())
    stats_text = f"""
    Activation Norm Statistics:

    Mean:    {a_stats['mean']:.6f}
    Std:     {a_stats['std']:.6f}
    Min:     {a_stats['min']:.6f}
    Max:     {a_stats['max']:.6f}
    Median:  {a_stats['median']:.6f}

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
    wm_stats = safe_stats(wanda_metric.flatten())
    stats_text = f"""
    Wanda Metric Statistics:

    Mean:    {wm_stats['mean']:.6e}
    Std:     {wm_stats['std']:.6e}
    Min:     {wm_stats['min']:.6e}
    Max:     {wm_stats['max']:.6e}
    Median:  {wm_stats['median']:.6e}

    Shape: {wanda_metric.shape}
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()

    # Save figure
    save_name = layer_name.replace('.', '_')
    output_file = output_dir / f'layer_analysis_{save_name}.png'
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
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'c4'],
                        help='Dataset for calibration (default: wikitext2 - faster)')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for calibration (default: 2048)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help=f'Output directory for visualizations (default: {ASSETS_DIR})')

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
    visualize_layer_metrics(model, tokenizer, args.layer, device, args.nsamples, args.dataset, args.seqlen, args.output_dir)

if __name__ == "__main__":
    main()
