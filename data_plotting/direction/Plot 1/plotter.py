import matplotlib.pyplot as plt
import numpy as np
from Plot_4LPDA_1dipole_SouthPole import load_file
import sys

# Call this function as python plotter.py i_file i_event
i_file = int(sys.argv[1])
i_event = int(sys.argv[2])

# Loading data
data = load_file(i_file)
print(f"Data shape: {data.shape}")

event_data = data[i_event]

# Plotting
fig, axs = plt.subplots(5)
fig.suptitle('Plot of 4 LDPA & 1 dipole')

for i in range(5):
    axs[i].plot(event_data[i])

    # Remove x ticks on all except last plot
    # if i != 4:
    #     axs[i].set_xticks([])
    #     axs[i].set_xticklabels([])

for ax in axs.flat:
    ax.set(xlabel='time (ns)', ylabel='signal (mV)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.set_size_inches(18.5, 10.5)

plt.savefig(f"plots/plot_file{i_file}_event{i_event}.png")