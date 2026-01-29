import sys
import pandas as pd
import matplotlib.pyplot as plt

def generate_plot(csv_pth, img_pth):
    """Reads runtime CSV data and generates a comparison plot."""
    try:
        df = pd.read_csv(csv_pth)
    except FileNotFoundError:
        print(f"Error: {csv_pth} not found.")
        return
    
    pivot_df = df.pivot(index='SupportThreshold', columns='Algorithm', values='Runtime(s)')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        pivot_df.index,
        pivot_df['Apriori'],
        marker='o',
        label='Apriori',
        linestyle='-',
        color='blue'
    )

    ax.plot(
        pivot_df.index,
        pivot_df['FP-Growth'],
        marker='s',
        label='FP-Growth',
        linestyle='--',
        color='orange'
    )

    ax.set_title("Runtime Comparison: Apriori vs. FP-Growth", fontsize=14)
    ax.set_xlabel("Support Threshold (%)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_xticks(df['SupportThreshold'].unique())
    ax.set_yscale('log') # Best to see differences in runtime
    ax.legend()
    ax.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig(img_pth)

    print(f"Plot saved to {img_pth}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_plot.py <input_csv_path> <output_image_path>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    output_image_path = sys.argv[2]

    generate_plot(input_csv_path, output_image_path)