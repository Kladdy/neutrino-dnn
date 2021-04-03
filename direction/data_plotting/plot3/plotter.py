import matplotlib.pyplot as plt
import numpy as np
from toolbox import load_file
from constants import plots_dir
import sys
import argparse
import os
from termcolor import colored
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

# Parse arguments
parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")
parser.add_argument("bandpass", type=str ,help="which bandpass to use (none, 300MHz, or 500MHz)")

args = parser.parse_args()
i_file = args.i_file
i_event = args.i_event
bandpass = args.bandpass

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Make sure bandpass is valid
if bandpass not in ["none", "300MHz", "500MHz"]:
    raise ValueError(f"'{bandpass}' is not a valid argument for bandpass. Valid values are none, 300MHz, or 500MHz.")

print(colored(f"Plotting antenna signals for event{i_event} in file{i_file}...", "yellow"))

# Loading data
data, nu_direction = load_file(i_file, bandpass_filter=bandpass)
print(f"Data shape: {data.shape}")

# Print out norms
print("The following are the norms of the data!")
normed_nu_direction = np.array([np.linalg.norm(v) for v in nu_direction])
print(normed_nu_direction)

event_data = data[i_event]
direction_data = nu_direction[i_event]

# Getting x axis (1 step is 0.5 ns)
x_axis_double = range(int(len(event_data[0])))
x_axis = [float(x)/2 for x in x_axis_double]

# Plotting
fig, axs = plt.subplots(5)
fig.suptitle(f'Plot of 4 LPDA & 1 dipole of SouthPole data for event {i_event} in file {i_file} with bandpass {bandpass}')

for i in range(5):
    axs[i].plot(x_axis, event_data[i])
    axs[i].set_xlim([min(x_axis), max(x_axis)])
    if i != 4:
        axs[i].set_title(f'LPDA {i+1}')

axs[4].set_title('Dipole')

for ax in axs.flat:
    ax.set(xlabel='time (ns)', ylabel=f'signal (μV)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.set_size_inches(12, 10)

plt.savefig(f"{plots_dir}/signal_file{i_file}_event{i_event}_bandpass{bandpass}.png")

fig.clear()
plt.close(fig)

# Plot direction sphere
fig_sphere = plt.figure()
ax_sphere = fig_sphere.gca(projection='3d')
#ax_sphere.set_aspect('equal')

# Draw sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax_sphere.plot_wireframe(x, y, z, color="grey", lw=1)

# Draw a point
ax_sphere.scatter([0], [0], [-1], color="g", s=100)

# Draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

arrow_scale = 2
norm = np.linalg.norm(direction_data)

XX = direction_data[0]/norm*arrow_scale
YY = direction_data[1]/norm*arrow_scale
ZZ = direction_data[2]/norm*arrow_scale

a = Arrow3D([0, XX], [0, YY], [-1, -1 + ZZ], mutation_scale=20,
            lw=3, arrowstyle="-|>", color="g")
ax_sphere.add_artist(a)

# Set viewing angle
ax_sphere.view_init(-25, -45)

#ax_sphere.set(xlabel="x", ylabel="y", zlabel="z")
ax_sphere.set_zticklabels([])
ax_sphere.set_yticklabels([])
ax_sphere.set_xticklabels([])

# Hide grid lines
ax_sphere.grid(False)

# Hide axes ticks
ax_sphere.set_xticks([])
ax_sphere.set_yticks([])
ax_sphere.set_zticks([])

plt.savefig(f"{plots_dir}/direction_file{i_file}_event{i_event}.png")
# ---------------------

print(colored(f"Done plotting antenna signals for event{i_event} in file{i_file}!", "green", attrs=["bold"]))
print("")