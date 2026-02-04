#!/usr/bin/env python3
"""
Run gSpan, FSG, and Gaston algorithms at different support levels and time them
"""
import sys
import os
import subprocess
import time
import json

def count_graphs(dataset_path):
    """Count total number of graphs in the dataset"""
    count = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip().startswith('t #'):
                count += 1
    return count

def run_gspan(executable, dataset, support_percent, output_file, total_graphs):
    """Run gSpan and return execution time"""
    # gSpan uses decimal format: -s 0.5 for 50%
    support_decimal = support_percent / 100.0
    
    cmd = [executable, '-f', dataset, '-s', str(support_decimal), '-o']
    
    print(f"Running gSpan with support {support_decimal} ({support_percent}%)...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        
        # gSpan creates output as dataset.fp
        fp_file = f"{dataset}.fp"
        if os.path.exists(fp_file):
            os.rename(fp_file, output_file)
        else:
            # Create empty output if no patterns found
            with open(output_file, 'w') as f:
                f.write(result.stdout)
        
        execution_time = end_time - start_time
        print(f"gSpan completed in {execution_time:.2f} seconds")
        return execution_time
    except subprocess.TimeoutExpired:
        print(f"gSpan timed out after 3600 seconds")
        # Clean up .fp file if it exists
        fp_file = f"{dataset}.fp"
        if os.path.exists(fp_file):
            os.rename(fp_file, output_file)
        return 3600
    except Exception as e:
        print(f"Error running gSpan: {e}")
        return -1

def run_fsg(executable, dataset, support_percent, output_file, total_graphs):
    """Run FSG and return execution time"""
    # FSG uses percentage format: -s50 for 50% (no space between -s and value)
    support_arg = f"-s{int(support_percent)}"
    
    cmd = [executable, support_arg, dataset]
    
    print(f"Running FSG with support {support_percent}%...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        
        # FSG may create a .fp file like gSpan
        fp_file = f"{dataset}.fp"
        if os.path.exists(fp_file):
            os.rename(fp_file, output_file)
        else:
            # Save FSG output from stdout
            with open(output_file, 'w') as f:
                f.write(result.stdout)
        
        execution_time = end_time - start_time
        print(f"FSG completed in {execution_time:.2f} seconds")
        return execution_time
    except subprocess.TimeoutExpired:
        print(f"FSG timed out after 3600 seconds")
        # Try to save .fp file if it exists
        fp_file = f"{dataset}.fp"
        if os.path.exists(fp_file):
            os.rename(fp_file, output_file)
        return 3600
    except Exception as e:
        print(f"Error running FSG: {e}")
        return -1

def run_gaston(executable, dataset, support_percent, output_file, total_graphs):
    """Run Gaston and return execution time"""
    # Gaston uses absolute support
    support = max(1, int(total_graphs * support_percent / 100))
    
    cmd = [executable, str(support), dataset, output_file]
    
    print(f"Running Gaston with support {support} ({support_percent}%)...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Gaston completed in {execution_time:.2f} seconds")
        return execution_time
    except subprocess.TimeoutExpired:
        print(f"Gaston timed out after 3600 seconds")
        return 3600
    except Exception as e:
        print(f"Error running Gaston: {e}")
        return -1

def main():
    if len(sys.argv) != 7:
        print("Usage: python run_algorithms.py <gspan_exe> <fsg_exe> <gaston_exe> <gspan_dataset> <fsg_dataset> <output_dir>")
        sys.exit(1)
    
    gspan_exe = sys.argv[1]
    fsg_exe = sys.argv[2]
    gaston_exe = sys.argv[3]
    gspan_dataset = sys.argv[4]
    fsg_dataset = sys.argv[5]
    output_dir = sys.argv[6]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Support levels to test
    # Full set for final run: [5, 10, 25, 50, 95]
    # For quick testing, use: [25, 50, 95]
    support_levels = [5, 10, 25, 50, 95]
    
    # Count total graphs
    print("Counting total graphs...")
    total_graphs = count_graphs(gspan_dataset)
    print(f"Total graphs: {total_graphs}")
    
    # Store timing results
    results = {
        'support_levels': support_levels,
        'gspan': [],
        'fsg': [],
        'gaston': []
    }
    
    # Run algorithms at different support levels
    for support in support_levels:
        print(f"\n{'='*60}")
        print(f"Testing with support = {support}%")
        print(f"{'='*60}\n")
        
        # Run gSpan
        gspan_output = os.path.join(output_dir, f"gspan{support}")
        gspan_time = run_gspan(gspan_exe, gspan_dataset, support, gspan_output, total_graphs)
        results['gspan'].append(gspan_time)
        print()
        
        # Run FSG
        fsg_output = os.path.join(output_dir, f"fsg{support}")
        fsg_time = run_fsg(fsg_exe, fsg_dataset, support, fsg_output, total_graphs)
        results['fsg'].append(fsg_time)
        print()
        
        # Run Gaston
        gaston_output = os.path.join(output_dir, f"gaston{support}")
        gaston_time = run_gaston(gaston_exe, gspan_dataset, support, gaston_output, total_graphs)
        results['gaston'].append(gaston_time)
        print()
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'timing_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("Summary of execution times (seconds):")
    print(f"{'Support':<10} {'gSpan':<15} {'FSG':<15} {'Gaston':<15}")
    print("-" * 55)
    for i, support in enumerate(support_levels):
        print(f"{support}%{'':<7} {results['gspan'][i]:<15.2f} {results['fsg'][i]:<15.2f} {results['gaston'][i]:<15.2f}")

if __name__ == "__main__":
    main()
