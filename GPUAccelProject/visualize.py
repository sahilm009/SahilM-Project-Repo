"""
Visualization module for GPU benchmark results
Generates plots comparing CPU vs GPU performance
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_latest_results(results_dir="results"):
    """Load the most recent benchmark results file"""
    pattern = os.path.join(results_dir, "benchmark_log_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No results found in {results_dir}/")
        return None
    
    # Get most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def plot_cpu_vs_gpu(history, output_dir="results"):
    """Create CPU vs GPU comparison plot"""
    sizes = []
    cpu_times = []
    gpu_times = []
    speedups = []
    
    for cycle in history:
        res = cycle["results"]
        if res.get("gpu_time_ms"):
            sizes.append(res["size"])
            cpu_times.append(res["cpu_time_ms"])
            gpu_times.append(res["gpu_time_ms"])
            speedups.append(res["speedup"])
    
    if not sizes:
        print("No valid GPU results to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: CPU vs GPU Time
    x = np.arange(len(sizes))
    width = 0.35
    
    ax1.bar(x - width/2, cpu_times, width, label='CPU', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, gpu_times, width, label='GPU', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Matrix Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('CPU vs GPU Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Speedup
    ax2.plot(range(len(speedups)), speedups, marker='o', linewidth=2, 
             markersize=8, color='#2ecc71')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1x (No speedup)')
    
    ax2.set_xlabel('Benchmark Run', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (x times faster)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Speedup Over CPU', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(speedups)))
    ax2.set_xticklabels([f'Run {i+1}' for i in range(len(speedups))])
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Add speedup values on points
    for i, speedup in enumerate(speedups):
        ax2.annotate(f'{speedup:.1f}x', 
                    (i, speedup), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center',
                    fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_plot_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {filename}")
    
    plt.show()


def generate_markdown_report(history, output_dir="results"):
    """Generate a markdown report of results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"report_{timestamp}.md")
    
    lines = []
    lines.append("# GPU-Accelerated Multi-Agent Runtime Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Cycles:** {len(history)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary table
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Cycle | Matrix Size | CPU Time (ms) | GPU Time (ms) | Speedup |")
    lines.append("|-------|-------------|---------------|---------------|---------|")
    
    for i, cycle in enumerate(history, 1):
        res = cycle["results"]
        size = res["size"]
        cpu = res["cpu_time_ms"]
        gpu = res.get("gpu_time_ms", "N/A")
        speedup = res.get("speedup", "N/A")
        
        if isinstance(gpu, (int, float)):
            gpu_str = f"{gpu:.2f}"
        else:
            gpu_str = "N/A"
        
        if isinstance(speedup, (int, float)):
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        lines.append(f"| {i} | {size}x{size} | {cpu:.2f} | {gpu_str} | {speedup_str} |")
    
    lines.append("")
    
    # Best result
    valid_results = [c["results"] for c in history if c["results"].get("speedup")]
    if valid_results:
        best = max(valid_results, key=lambda x: x["speedup"])
        lines.append("## Best Performance")
        lines.append("")
        lines.append(f"- **Matrix Size:** {best['size']}x{best['size']}")
        lines.append(f"- **Speedup:** {best['speedup']:.2f}x")
        lines.append(f"- **GPU Backend:** {best.get('gpu_backend', 'Unknown')}")
        lines.append("")
    
    # Detailed results
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    
    for i, cycle in enumerate(history, 1):
        lines.append(f"### Cycle {i}")
        lines.append("")
        lines.append(f"**Task Planning:**")
        lines.append(f"- {cycle['task'].get('reasoning', 'N/A')}")
        lines.append("")
        lines.append("**Evaluation:**")
        lines.append("```")
        lines.append(cycle['report'])
        lines.append("```")
        lines.append("")
    
    # Write report
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"📝 Report saved to: {filename}")


def main():
    """Main visualization function"""
    print("\n" + "="*60)
    print("GPU BENCHMARK VISUALIZATION")
    print("="*60 + "\n")
    
    # Load results
    history = load_latest_results()
    
    if history is None:
        print("\n⚠️  No results found. Run runtime.py first to generate data.")
        return
    
    print(f"Loaded {len(history)} benchmark cycles\n")
    
    # Generate visualizations
    plot_cpu_vs_gpu(history)
    generate_markdown_report(history)
    
    print("\n✅ Visualization complete!")
    print("")


if __name__ == "__main__":
    main()