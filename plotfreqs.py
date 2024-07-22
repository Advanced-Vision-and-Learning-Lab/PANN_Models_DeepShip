import matplotlib.pyplot as plt
import numpy as np

# Data
frequencies = [2, 4, 8, 16, 32, 64]
models = ['CNN14_8K', 'CNN14_16K', 'CNN14_32K']
accuracy_means = {
    'CNN14_8K': [74.2, 72.8, 70.4, 66.9, 67.3, 63.1],
    'CNN14_16K': [74.2, 73.9, 72.0, 69.5, 67.3, 65.9],
    'CNN14_32K': [73.6, 75.1, 73.0, 71.5, 70.3, 69.6],
}
accuracy_stds = {
    'CNN14_8K': [0.5, 0.7, 0.6, 0.6, 0.4, 1.3],
    'CNN14_16K': [1.3, 0.3, 0.6, 0.4, 0.6, 1.1],
    'CNN14_32K': [0.7, 0.3, 0.6, 0.6, 0.2, 1.1],
}

# Plotting
fig, ax = plt.subplots(figsize=(6, 4))

colors = {'CNN14_8K': '#17becf', 'CNN14_16K': '#2ca02c', 'CNN14_32K': '#e377c2'}
markers = {'CNN14_8K': 'o', 'CNN14_16K': 'd', 'CNN14_32K': '^'}

for model in models:
    ax.errorbar(frequencies, accuracy_means[model], yerr=accuracy_stds[model], fmt=markers[model], 
                label=model, color=colors[model], alpha=0.8, capsize=7, markersize=6, elinewidth=3, capthick=3, linestyle='None')

ax.set_xlabel('Data Sampling Rate (kHz)', fontsize=10)
ax.set_ylabel('Test Accuracy (%)', fontsize=10)
ax.set_xticks(frequencies)
ax.set_ylim(61, 76)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', linewidth=0.7)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


plt.savefig('features/freq_accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.close()
