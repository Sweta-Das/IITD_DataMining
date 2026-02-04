#!/usr/bin/env python3
import sys
import json
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def plot_timing_results(results_file, output_plot):
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    support_levels = results['support_levels']
    gspan_times = results['gspan']
    fsg_times = results['fsg']
    gaston_times = results['gaston']
    

    plt.figure(figsize=(10, 6))
    

    plt.plot(support_levels, gspan_times, marker='o', linewidth=2, markersize=8, label='gSpan')
    plt.plot(support_levels, fsg_times, marker='s', linewidth=2, markersize=8, label='FSG')
    plt.plot(support_levels, gaston_times, marker='^', linewidth=2, markersize=8, label='Gaston')
    
    plt.xlabel('Minimum Support (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Frequent Subgraph Mining: Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    

    plt.xticks(support_levels)

    plt.tight_layout()
    
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")
    
    print("\n" + "="*60)
    print("Timing Summary:")
    print("="*60)
    print(f"{'Support':<10} {'gSpan (s)':<15} {'FSG (s)':<15} {'Gaston (s)':<15}")
    print("-" * 60)
    for i, support in enumerate(support_levels):
        print(f"{support}%{'':<7} {gspan_times[i]:<15.2f} {fsg_times[i]:<15.2f} {gaston_times[i]:<15.2f}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py <results_json> <output_plot>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_plot = sys.argv[2]
    
    plot_timing_results(results_file, output_plot)
