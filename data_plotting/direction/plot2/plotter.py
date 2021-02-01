import matplotlib.pyplot as plt
import numpy as np
from toolbox import load_file
from constants import plots_dir
import sys
import argparse
from termcolor import colored

# Parse arguments
parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")

args = parser.parse_args()
i_file = args.i_file
i_event = args.i_event

print(colored(f"Plotting antenna signals for event{i_event} in file{i_file}...", "yellow"))

# Loading data
data = load_file(i_file)
print(data)
print(f"Data shape: {data.shape}")

event_data = data[i_event]

# Plotting
fig, axs = plt.subplots(5)
fig.suptitle(f'Plot of 4 LDPA & 1 dipole of SouthPole data for event {i_event} in file {i_file}')

for i in range(5):
    axs[i].plot(event_data[i])
    axs[i].set_xlim([0, 511])
    if i != 4:
        axs[i].set_title(f'LDPA {i+1}')

axs[4].set_title('Dipole')

for ax in axs.flat:
    ax.set(xlabel='time (ns)', ylabel=f'signal (uV)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.set_size_inches(12, 10)

plt.savefig(f"{plots_dir}/plot_file{i_file}_event{i_event}.png")

print(colored(f"Done plotting antenna signals for event{i_event} in file{i_file}!", "green", attrs=["bold"]))
print("")