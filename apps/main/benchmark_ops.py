import torch
import time
import sys
import json
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import statistics
from torch import nn
import torch.nn.functional as F
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))

from lingua.product_key.memory import HashingMemory, ProductKeyArgs
from lingua.moe import MoEArgs, SparseMoeBlock
from lingua.transformer import FeedForward, Attention


@dataclass
class ModelArgs:
    dim: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 16
    rope_theta: float = 500000.0
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    productkey_args: ProductKeyArgs = ProductKeyArgs()
    moe_args: MoEArgs = MoEArgs(top_k=1)


def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)


def benchmark_layer(
    layer: nn.Module,
    input_size: tuple,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    freq_cis: Optional[torch.Tensor] = None,
):
    """Benchmark a single layer's inference latency."""
    x = torch.randn(input_size, dtype=torch.bfloat16).to(device)
    layer = layer.to(dtype=torch.bfloat16)

    # Warmup runs
    if isinstance(layer, Attention):
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = layer(x, freq_cis, mask="causal")
    else:
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = layer(x)

    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Actual timing runs
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()

        if isinstance(layer, Attention):
            with torch.no_grad():
                _ = layer(x, freq_cis, mask="causal")
        else:
            with torch.no_grad():
                _ = layer(x)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start_time) * 1000)  # Convert to ms

    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "std": statistics.stdev(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


def benchmark_sdpa(input_size, n_heads, num_warmup=10, num_runs=100, device="cuda"):
    """Benchmark PyTorch's scaled_dot_product_attention."""
    batch_size, seq_length, dim = input_size
    query = torch.randn((batch_size, seq_length, n_heads, dim // n_heads), dtype=torch.bfloat16).to(device)
    key = query.clone()
    value = query.clone()

    for _ in range(num_warmup):
        with torch.no_grad():
            F.scaled_dot_product_attention(query, key, value, is_causal=True)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        with torch.no_grad():
            F.scaled_dot_product_attention(query, key, value, is_causal=True)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start_time) * 1000)

    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "std": statistics.stdev(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


def run_benchmarks(batch_size: int = 8, seq_length: int = 512, device: str = "cuda", output_file: Optional[str] = None):
    """Run benchmarks for all three layer types."""
    args = ModelArgs()
    input_size = (batch_size, seq_length, args.dim)

    freq_cis = torch.randn((seq_length, args.dim // args.n_heads // 2, 2, 2), dtype=torch.bfloat16).to(device)

    attention = Attention(
        dim=args.dim,
        head_dim=args.dim // args.n_heads,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        rope_theta=args.rope_theta,
    ).to(device)

    # Initialize layers
    hashing_memory = HashingMemory(
        input_dim=args.dim,
        output_dim=args.dim,
        mem_n_keys=args.productkey_args.mem_n_keys,
        mem_heads=args.productkey_args.mem_heads,
        mem_knn=args.productkey_args.mem_knn,
        mem_share_values=args.productkey_args.mem_share_values,
        mem_k_dim=args.productkey_args.mem_k_dim,
        mem_v_dim=args.productkey_args.mem_v_dim,
        swilu_projection=args.productkey_args.swilu_projection,
        value_fixed_lr=args.productkey_args.value_fixed_lr,
        mem_gated=args.productkey_args.mem_gated,
        peer_variant=args.productkey_args.peer_variant,
    ).to(device)

    sparse_moe = SparseMoeBlock(
        hidden_dim=args.dim,
        ffn_dim=args.moe_args.ffn_dim,
        num_experts=args.moe_args.num_experts,
        top_k=args.moe_args.top_k,
    ).to(device)

    feed_forward = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
    ).to(device)

    # Run benchmarks
    results = {
        "HashingMemory": {
            **benchmark_layer(hashing_memory, input_size, device=device),
            "params": count_parameters(hashing_memory),
        },
        "SparseMoeBlock": {
            **benchmark_layer(sparse_moe, input_size, device=device),
            "params": count_parameters(sparse_moe),
        },
        "FeedForward": {
            **benchmark_layer(feed_forward, input_size, device=device),
            "params": count_parameters(feed_forward),
        },
        "Attention (w/RoPE,Proj)": {
            **benchmark_layer(attention, input_size, freq_cis=freq_cis, device=device),
            "params": count_parameters(attention),
        },
        "SDPA": {
            **benchmark_sdpa(input_size, args.n_heads, device=device),
            "params": "N/A",
        },
    }

    # Print results
    print(f"Benchmark Results (Unit: ms): {batch_size=}, {seq_length=}")
    print("-" * 100)
    print(f"{'Layer Type':<25} {'Params':>12} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 100)

    for layer_name, metrics in results.items():
        params = metrics.get("params", "N/A")
        print(
            f"{layer_name:<25} {params:>12} {metrics['mean']:>10.3f} {metrics['median']:>10.3f} "
            f"{metrics['std']:>10.3f} {metrics['min']:>10.3f} {metrics['max']:>10.3f}"
        )
    print("-" * 100)

    results.update({"batch_size": batch_size, "seq_length": seq_length})
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as f:
            f.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer Benchmarking Tool")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--seq-length", type=int, default=4096, help="Sequence length for inference")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run benchmarks on")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=100, help="Number of timing runs")
    parser.add_argument("--output-file", type=str, default=None, help="File to save benchmark results")

    args = parser.parse_args()
    run_benchmarks(args.batch_size, args.seq_length, args.device, args.output_file)


# srun -J train -N 1 --gpus-per-node 8 --exclusive python apps/main/benchmark_ops.py
