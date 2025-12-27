"""
Memory monitoring utilities for tracking GPU memory usage during pruning
"""

import torch
import psutil
import os
from datetime import datetime

class MemoryTracker:
    """Track and log GPU and CPU memory usage"""

    def __init__(self, enabled=True, log_file=None):
        """
        Initialize memory tracker

        Args:
            enabled: Whether to enable tracking
            log_file: Optional file path to save memory logs
        """
        self.enabled = enabled
        self.log_file = log_file
        self.checkpoints = []

        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("Timestamp,Checkpoint,GPU_Allocated_GB,GPU_Reserved_GB,GPU_Peak_GB,CPU_Used_GB,CPU_Percent\n")

    def checkpoint(self, name):
        """
        Record memory usage at a checkpoint

        Args:
            name: Name of the checkpoint
        """
        if not self.enabled:
            return

        stats = self.get_memory_stats()
        stats['checkpoint'] = name
        stats['timestamp'] = datetime.now().isoformat()

        self.checkpoints.append(stats)

        # Print to console
        self._print_stats(name, stats)

        # Write to log file
        if self.log_file:
            self._write_to_log(stats)

    def get_memory_stats(self):
        """Get current memory statistics"""
        stats = {}

        # GPU memory
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            stats['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats['gpu_free_gb'] = stats['gpu_total_gb'] - stats['gpu_allocated_gb']
        else:
            stats['gpu_allocated_gb'] = 0
            stats['gpu_reserved_gb'] = 0
            stats['gpu_peak_gb'] = 0
            stats['gpu_total_gb'] = 0
            stats['gpu_free_gb'] = 0

        # CPU memory
        process = psutil.Process(os.getpid())
        stats['cpu_used_gb'] = process.memory_info().rss / (1024**3)
        stats['cpu_percent'] = process.memory_percent()

        return stats

    def _print_stats(self, name, stats):
        """Print memory statistics to console"""
        print(f"\n{'='*80}")
        print(f"Memory Checkpoint: {name}")
        print(f"{'='*80}")
        print(f"GPU Memory:")
        print(f"  Allocated: {stats['gpu_allocated_gb']:.2f} GB")
        print(f"  Reserved:  {stats['gpu_reserved_gb']:.2f} GB")
        print(f"  Peak:      {stats['gpu_peak_gb']:.2f} GB")
        print(f"  Free:      {stats['gpu_free_gb']:.2f} GB")
        print(f"  Total:     {stats['gpu_total_gb']:.2f} GB")
        print(f"CPU Memory:")
        print(f"  Used:      {stats['cpu_used_gb']:.2f} GB ({stats['cpu_percent']:.1f}%)")
        print(f"{'='*80}\n")

    def _write_to_log(self, stats):
        """Write statistics to log file"""
        with open(self.log_file, 'a') as f:
            f.write(f"{stats['timestamp']},{stats['checkpoint']},")
            f.write(f"{stats['gpu_allocated_gb']:.3f},")
            f.write(f"{stats['gpu_reserved_gb']:.3f},")
            f.write(f"{stats['gpu_peak_gb']:.3f},")
            f.write(f"{stats['cpu_used_gb']:.3f},")
            f.write(f"{stats['cpu_percent']:.2f}\n")

    def print_summary(self):
        """Print summary of all checkpoints"""
        if not self.enabled or not self.checkpoints:
            return

        print(f"\n{'='*80}")
        print("MEMORY USAGE SUMMARY")
        print(f"{'='*80}")
        print(f"{'Checkpoint':<30} {'GPU Alloc':<12} {'GPU Peak':<12} {'CPU Used':<12}")
        print(f"{'-'*80}")

        for cp in self.checkpoints:
            print(f"{cp['checkpoint']:<30} "
                  f"{cp['gpu_allocated_gb']:>10.2f} GB "
                  f"{cp['gpu_peak_gb']:>10.2f} GB "
                  f"{cp['cpu_used_gb']:>10.2f} GB")

        print(f"{'='*80}\n")

        # Print peak memory
        if self.checkpoints:
            peak_gpu = max(cp['gpu_peak_gb'] for cp in self.checkpoints)
            peak_cpu = max(cp['cpu_used_gb'] for cp in self.checkpoints)
            print(f"Peak GPU Memory: {peak_gpu:.2f} GB")
            print(f"Peak CPU Memory: {peak_cpu:.2f} GB")
            print(f"{'='*80}\n")

    def get_memory_delta(self, checkpoint1, checkpoint2):
        """
        Calculate memory difference between two checkpoints

        Args:
            checkpoint1: Name of first checkpoint
            checkpoint2: Name of second checkpoint

        Returns:
            Dictionary with memory deltas
        """
        cp1 = next((cp for cp in self.checkpoints if cp['checkpoint'] == checkpoint1), None)
        cp2 = next((cp for cp in self.checkpoints if cp['checkpoint'] == checkpoint2), None)

        if not cp1 or not cp2:
            return None

        return {
            'gpu_allocated_delta': cp2['gpu_allocated_gb'] - cp1['gpu_allocated_gb'],
            'gpu_peak_delta': cp2['gpu_peak_gb'] - cp1['gpu_peak_gb'],
            'cpu_delta': cp2['cpu_used_gb'] - cp1['cpu_used_gb']
        }


def print_gpu_memory_summary():
    """Quick function to print current GPU memory status"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"\n{'='*80}")
    print("GPU MEMORY STATUS")
    print(f"{'='*80}")

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - allocated

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Peak:      {peak:.2f} GB")
    print(f"Free:      {free:.2f} GB")
    print(f"Total:     {total:.2f} GB")
    print(f"Usage:     {(allocated/total)*100:.1f}%")
    print(f"{'='*80}\n")


def estimate_activation_memory(batch_size, seq_len, hidden_dim, num_layers):
    """
    Estimate activation memory requirements

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers

    Returns:
        Estimated memory in GB
    """
    # Attention activations (Q, K, V, O)
    attn_memory = batch_size * seq_len * hidden_dim * 4 * 2  # fp16

    # MLP activations (gate, up, down)
    ffn_dim = hidden_dim * 4
    mlp_memory = batch_size * seq_len * ffn_dim * 3 * 2  # fp16

    # Total per layer
    per_layer = (attn_memory + mlp_memory) / (1024**3)

    # Total for all layers
    total = per_layer * num_layers

    return total
