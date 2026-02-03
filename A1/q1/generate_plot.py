import sys
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_plot(csv_filename, output_filename):

    if not os.path.exists(csv_filename):
        print(f"ERROR: '{csv_filename}' not found.")
        return

    data = defaultdict(dict)

    # Read CSV manually
    with open(csv_filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            support = int(row['SupportThreshold'])
            algo = row['Algorithm']
            runtime = float(row['RunTime(s)'])
            data[algo][support] = runtime

    supports = sorted({s for algo in data for s in data[algo]})

    plt.figure(figsize=(10, 7))

    for algo in data:
        runtimes = [data[algo][s] for s in supports]
        plt.plot(supports, runtimes, marker='o', label=algo)

    plt.title('Performance Comparison: Apriori vs. FP-Growth')
    plt.xlabel('Minimum Support Threshold (%)')
    plt.ylabel('Runtime (seconds)')
    plt.yscale('log')
    plt.xticks(supports)
    plt.legend()

    plt.savefig(output_filename, dpi=300)
    print(f"SUCCESS: Plot saved as '{output_filename}'")


if __name__ == '__main__':
    generate_plot(sys.argv[1], sys.argv[2])
