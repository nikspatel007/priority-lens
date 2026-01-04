#!/usr/bin/env python3
"""Benchmark JSON vs SurrealDB data loading performance.

Compares:
1. Initial load time (JSON file parse vs SurrealDB query)
2. Random access time (dict lookup vs DB query)
3. Batch iteration time (DataLoader throughput)
4. Memory usage

Usage:
    # Ensure SurrealDB is running with imported data:
    surreal start file:data/enron.db --user root --pass root

    # Run benchmark:
    python -m db.benchmark --data-dir data --batch-size 64
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def benchmark_json_loading(data_dir: Path, batch_size: int = 32) -> dict:
    """Benchmark JSON-based dataset loading."""
    from src.dataset import create_dataloaders

    gc.collect()
    mem_before = get_memory_mb()

    # Time initial load
    start = time.perf_counter()
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        precompute=True,
    )
    load_time = time.perf_counter() - start

    mem_after = get_memory_mb()

    # Time batch iteration
    start = time.perf_counter()
    batch_count = 0
    for features, labels in train_loader:
        batch_count += 1
        if batch_count >= 100:
            break
    iter_time = time.perf_counter() - start

    # Time random access
    dataset = train_loader.dataset
    indices = torch.randperm(len(dataset))[:1000].tolist()

    start = time.perf_counter()
    for idx in indices:
        _ = dataset[idx]
    access_time = time.perf_counter() - start

    return {
        'load_time_s': load_time,
        'iter_time_s': iter_time,
        'iter_batches': batch_count,
        'access_time_s': access_time,
        'access_count': len(indices),
        'memory_mb': mem_after - mem_before,
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
    }


def benchmark_surreal_loading(
    database: str = 'enron',
    batch_size: int = 32,
    url: str = 'ws://localhost:8000/rpc',
) -> dict:
    """Benchmark SurrealDB-based dataset loading."""
    from db.dataset import create_surreal_dataloaders

    gc.collect()
    mem_before = get_memory_mb()

    # Time initial load (includes DB query)
    start = time.perf_counter()
    train_loader, val_loader, test_loader = create_surreal_dataloaders(
        database=database,
        batch_size=batch_size,
        url=url,
    )
    load_time = time.perf_counter() - start

    mem_after = get_memory_mb()

    # Time batch iteration
    start = time.perf_counter()
    batch_count = 0
    for features, labels in train_loader:
        batch_count += 1
        if batch_count >= 100:
            break
    iter_time = time.perf_counter() - start

    # Time random access
    dataset = train_loader.dataset
    if len(dataset) > 0:
        indices = torch.randperm(len(dataset))[:min(1000, len(dataset))].tolist()

        start = time.perf_counter()
        for idx in indices:
            _ = dataset[idx]
        access_time = time.perf_counter() - start
        access_count = len(indices)
    else:
        access_time = 0
        access_count = 0

    return {
        'load_time_s': load_time,
        'iter_time_s': iter_time,
        'iter_batches': batch_count,
        'access_time_s': access_time,
        'access_count': access_count,
        'memory_mb': mem_after - mem_before,
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
    }


def format_results(name: str, results: dict) -> str:
    """Format benchmark results for display."""
    lines = [
        f"\n{'=' * 60}",
        f"{name} Benchmark Results",
        f"{'=' * 60}",
        f"",
        f"Dataset sizes:",
        f"  Train: {results['train_size']:,}",
        f"  Val:   {results['val_size']:,}",
        f"  Test:  {results['test_size']:,}",
        f"",
        f"Loading:",
        f"  Initial load time: {results['load_time_s']:.3f}s",
        f"  Memory usage:      {results['memory_mb']:.1f} MB",
        f"",
        f"Iteration (100 batches):",
        f"  Time:       {results['iter_time_s']:.3f}s",
        f"  Throughput: {results['iter_batches'] / results['iter_time_s']:.1f} batches/s",
        f"",
    ]

    if results['access_count'] > 0:
        lines.extend([
            f"Random access ({results['access_count']} samples):",
            f"  Time:       {results['access_time_s']:.3f}s",
            f"  Throughput: {results['access_count'] / results['access_time_s']:.0f} samples/s",
        ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark JSON vs SurrealDB data loading'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Directory containing JSON data files'
    )
    parser.add_argument(
        '--database',
        default='enron',
        help='SurrealDB database name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for DataLoader'
    )
    parser.add_argument(
        '--url',
        default='ws://localhost:8000/rpc',
        help='SurrealDB connection URL'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only benchmark JSON loading'
    )
    parser.add_argument(
        '--surreal-only',
        action='store_true',
        help='Only benchmark SurrealDB loading'
    )

    args = parser.parse_args()

    json_results = None
    surreal_results = None

    # Benchmark JSON
    if not args.surreal_only:
        train_json = args.data_dir / 'train.json'
        if train_json.exists():
            print("Benchmarking JSON loading...")
            json_results = benchmark_json_loading(args.data_dir, args.batch_size)
            print(format_results("JSON", json_results))
        else:
            print(f"Warning: {train_json} not found, skipping JSON benchmark")

    # Benchmark SurrealDB
    if not args.json_only:
        print("\nBenchmarking SurrealDB loading...")
        try:
            surreal_results = benchmark_surreal_loading(
                args.database,
                args.batch_size,
                args.url,
            )
            print(format_results("SurrealDB", surreal_results))
        except Exception as e:
            print(f"SurrealDB benchmark failed: {e}")
            print("Make sure SurrealDB is running and data is imported.")

    # Comparison
    if json_results and surreal_results:
        print(f"\n{'=' * 60}")
        print("Comparison (JSON / SurrealDB)")
        print(f"{'=' * 60}")
        print()

        load_ratio = json_results['load_time_s'] / max(surreal_results['load_time_s'], 0.001)
        print(f"Initial load:     {load_ratio:.2f}x " +
              ("(JSON faster)" if load_ratio < 1 else "(SurrealDB faster)"))

        if surreal_results['iter_time_s'] > 0:
            iter_ratio = json_results['iter_time_s'] / surreal_results['iter_time_s']
            print(f"Batch iteration:  {iter_ratio:.2f}x " +
                  ("(JSON faster)" if iter_ratio < 1 else "(SurrealDB faster)"))

        if surreal_results['access_time_s'] > 0 and json_results['access_time_s'] > 0:
            access_ratio = json_results['access_time_s'] / surreal_results['access_time_s']
            print(f"Random access:    {access_ratio:.2f}x " +
                  ("(JSON faster)" if access_ratio < 1 else "(SurrealDB faster)"))

        mem_ratio = json_results['memory_mb'] / max(surreal_results['memory_mb'], 0.1)
        print(f"Memory usage:     {mem_ratio:.2f}x " +
              ("(JSON uses less)" if mem_ratio < 1 else "(SurrealDB uses less)"))

        print()
        print("Note: SurrealDB advantages (graph queries, caching, streaming)")
        print("      are not captured in this basic benchmark.")


if __name__ == '__main__':
    main()
