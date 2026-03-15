import matplotlib.pyplot as plt

# Data
simulations = [50, 200, 500, 1000]

dataset1 = [77.07, 78.58, 78.02, 76.26]
dataset2 = [88.31, 92.25, 92.17, 92.12]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10,4))

# Dataset 1 plot
axes[0].plot(simulations, dataset1, marker='o')
axes[0].set_title("Dataset 1 (h = -1)")
axes[0].set_xlabel("Number of Simulations")
axes[0].set_ylabel("Reduction (%)")
axes[0].grid(True)

# Dataset 2 plot
axes[1].plot(simulations, dataset2, marker='o')
axes[1].set_title("Dataset 2 (h = 3)")
axes[1].set_xlabel("Number of Simulations")
axes[1].set_ylabel("Reduction (%)")
axes[1].grid(True)

plt.tight_layout()

# Save figure
plt.savefig("aib253027_plots.png", dpi=300)

plt.show()