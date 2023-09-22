import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

INPUT_FEATURES = ['pn', 'pe', 'pd',
                  'u', 'v', 'w',
                  'e0', 'e1', 'e2', 'e3',
                  'p', 'q', 'r',
                  'delta_e', 'delta_a', 'delta_r', 'delta_t']

data_path = '/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/test/dataset_orbit_1.csv'
save_path = '/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/test/'

# Read CSV data using Pandas
df = pd.read_csv(data_path)

# Create a color palette for the lines
colors = plt.cm.viridis(np.linspace(0, 1, len(INPUT_FEATURES)))

# Assuming a constant sampling rate of 0.02 seconds per sample
sampling_rate = 0.02  # seconds
num_samples = len(df)
time = np.arange(0, num_samples * sampling_rate, sampling_rate)

# Define the number of rows and columns for subplots
num_rows = 5  # Number of rows
num_cols = 4  # Number of columns

# Calculate the figure size based on the number of rows and columns
fig_width = 16  # Adjust the width of the figure as needed
fig_height = 16  # Adjust the height of the figure as needed

# Plot features in rows and columns
fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
plt.style.use('seaborn-darkgrid')

for i, (feature, color) in enumerate(zip(INPUT_FEATURES, colors)):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].plot(time, df[feature], label=feature, color=color, linewidth=2)
    axs[row, col].set_title(feature)
    axs[row, col].set_xlabel('Time (s)', fontsize=10)
    axs[row, col].set_ylabel('Value', fontsize=10)
    axs[row, col].legend(loc='upper right', fontsize=8)

# Remove empty subplots if there are more features than can fit
for i in range(len(INPUT_FEATURES), num_rows * num_cols):
    fig.delaxes(axs.flatten()[i])

# Adjust spacing and labels
fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.suptitle('Feature Plots', fontsize=16)

plt.savefig(save_path + 'features.png', dpi=300)  # Increase DPI for higher resolution
plt.show()
