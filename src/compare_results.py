#!/usr/bin/env python3
"""Compare evaluation results across training stages.

Generates comparison charts and summary tables for tracking model improvement.

Usage:
    python src/compare_results.py eval_results/
    python src/compare_results.py eval_results/baseline.json eval_results/stage_1.json eval_results/stage_2.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(path: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def find_result_files(directory: Path) -> list[Path]:
    """Find all JSON result files in a directory."""
    files = sorted(directory.glob('*.json'))
    # Sort by name, putting baseline first
    files.sort(key=lambda p: (0 if 'baseline' in p.name else 1, p.name))
    return files


def extract_stage_name(path: Path) -> str:
    """Extract a readable stage name from file path."""
    name = path.stem
    if name == 'baseline':
        return 'Baseline'
    elif name.startswith('stage_'):
        return f'Stage {name.split("_")[1]}'
    return name.replace('_', ' ').title()


def print_comparison_table(results: list[tuple[str, dict]]) -> None:
    """Print a comparison table of key metrics."""
    print()
    print("=" * 80)
    print("TRAINING STAGE COMPARISON")
    print("=" * 80)
    print()

    # Header
    headers = ["Metric"] + [name for name, _ in results]
    col_width = max(15, max(len(h) for h in headers) + 2)

    print(f"{'Metric':<25}", end="")
    for name, _ in results:
        print(f"{name:>{col_width}}", end="")
    print()
    print("-" * (25 + col_width * len(results)))

    # Key metrics
    metrics = [
        ("Action Accuracy", "action_accuracy", lambda x: f"{x * 100:.2f}%"),
        ("Priority MAE", "priority_mae", lambda x: f"{x:.4f}"),
        ("Priority Correlation", "priority_correlation", lambda x: f"{x:.4f}"),
        ("Reply F1", ("action_f1", "reply_now"), lambda x: f"{x:.4f}"),
        ("Forward F1", ("action_f1", "forward"), lambda x: f"{x:.4f}"),
        ("Archive F1", ("action_f1", "archive"), lambda x: f"{x:.4f}"),
        ("Delete F1", ("action_f1", "delete"), lambda x: f"{x:.4f}"),
        ("Macro F1 (avg)", None, None),  # Computed
    ]

    for metric_name, metric_key, formatter in metrics:
        print(f"{metric_name:<25}", end="")

        for name, data in results:
            if metric_key is None:
                # Compute macro F1
                f1_scores = data.get("action_f1", {})
                non_zero = [v for v in f1_scores.values() if v > 0]
                value = sum(non_zero) / len(non_zero) if non_zero else 0
                print(f"{value:>{col_width}.4f}", end="")
            elif isinstance(metric_key, tuple):
                # Nested key
                value = data.get(metric_key[0], {}).get(metric_key[1], 0)
                print(f"{formatter(value):>{col_width}}", end="")
            else:
                value = data.get(metric_key, 0)
                print(f"{formatter(value):>{col_width}}", end="")
        print()

    print("=" * 80)
    print()

    # Show improvement from baseline
    if len(results) > 1:
        baseline_name, baseline_data = results[0]
        baseline_acc = baseline_data.get("action_accuracy", 0)

        print("IMPROVEMENT FROM BASELINE:")
        print("-" * 40)

        for name, data in results[1:]:
            acc = data.get("action_accuracy", 0)
            improvement = acc - baseline_acc
            relative = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
            print(f"  {name}: {improvement * 100:+.2f}% accuracy ({relative:+.1f}% relative)")

        print()


def generate_accuracy_chart(
    results: list[tuple[str, dict]],
    output_path: Path,
) -> None:
    """Generate accuracy improvement chart."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping chart generation")
        return

    stages = [name for name, _ in results]
    accuracies = [data.get("action_accuracy", 0) * 100 for _, data in results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy line chart
    ax1.plot(stages, accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Training Stage', fontsize=12)
    ax1.set_ylabel('Action Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Across Training Stages', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Add value labels
    for i, acc in enumerate(accuracies):
        ax1.annotate(f'{acc:.1f}%', (i, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10)

    # F1 score comparison bar chart
    actions = ['reply_now', 'forward', 'archive', 'delete']
    x = range(len(stages))
    width = 0.2

    for i, action in enumerate(actions):
        f1_scores = [data.get("action_f1", {}).get(action, 0) for _, data in results]
        offset = (i - len(actions) / 2 + 0.5) * width
        bars = ax2.bar([xi + offset for xi in x], f1_scores, width, label=action)

    ax2.set_xlabel('Training Stage', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Per-Action F1 Scores', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, rotation=45, ha='right')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Chart saved to {output_path}")


def generate_text_chart(results: list[tuple[str, dict]]) -> str:
    """Generate ASCII chart for terminal output."""
    lines = []
    lines.append("")
    lines.append("ACCURACY IMPROVEMENT CHART")
    lines.append("-" * 60)

    max_width = 40
    stages = [name for name, _ in results]
    accuracies = [data.get("action_accuracy", 0) * 100 for _, data in results]

    max_acc = max(accuracies) if accuracies else 100

    for stage, acc in zip(stages, accuracies):
        bar_len = int(acc / max_acc * max_width) if max_acc > 0 else 0
        bar = "#" * bar_len
        lines.append(f"{stage:>15} |{bar:<{max_width}} {acc:.1f}%")

    lines.append("-" * 60)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation results')
    parser.add_argument(
        'paths',
        nargs='+',
        type=Path,
        help='Result files or directory containing results',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('eval_results/comparison_chart.png'),
        help='Output path for comparison chart (default: eval_results/comparison_chart.png)',
    )
    parser.add_argument(
        '--no-chart',
        action='store_true',
        help='Skip chart generation',
    )

    args = parser.parse_args()

    # Collect result files
    result_files = []
    for path in args.paths:
        if path.is_dir():
            result_files.extend(find_result_files(path))
        elif path.is_file():
            result_files.append(path)
        else:
            print(f"Warning: {path} not found, skipping")

    if not result_files:
        print("Error: No result files found")
        sys.exit(1)

    # Load results
    results = []
    for path in result_files:
        try:
            data = load_results(path)
            stage_name = extract_stage_name(path)
            results.append((stage_name, data))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    if not results:
        print("Error: No valid results loaded")
        sys.exit(1)

    print(f"Loaded {len(results)} evaluation results")

    # Print comparison table
    print_comparison_table(results)

    # Print ASCII chart
    print(generate_text_chart(results))

    # Generate matplotlib chart
    if not args.no_chart:
        generate_accuracy_chart(results, args.output)


if __name__ == '__main__':
    main()
