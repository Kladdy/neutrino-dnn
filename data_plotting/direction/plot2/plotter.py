import matplotlib.pyplot as plt
import numpy as np
from toolbox import load_file
from constants import plots_dir
import sys
import argparse
import os
from termcolor import colored
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from itertools import product, combinations

# Parse arguments
parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")

args = parser.parse_args()
i_file = args.i_file
i_event = args.i_event

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

print(colored(f"Plotting antenna signals for event{i_event} in file{i_file}...", "yellow"))

# Loading data
data, nu_direction = load_file(i_file)
print(f"Data shape: {data.shape}")

event_data = data[i_event]
direction_data = nu_direction[i_event]

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

plt.savefig(f"{plots_dir}/signal_file{i_file}_event{i_event}.png")

fig.clear()
plt.close(fig)

# Plot direction sphere
fig_sphere = plt.figure()
ax_sphere = fig_sphere.gca(projection='3d')
#ax_sphere.set_aspect('equal')

# Draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
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

a = Arrow3D([0, 1/2], [0, 1/2], [0-1, -1-1/2], mutation_scale=20,
            lw=3, arrowstyle="-|>", color="g")
ax_sphere.add_artist(a)


# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax_sphere.plot([xb], [yb], [zb], 'w')

# Set viewing angle
ax_sphere.view_init(-25, -45)

ax_sphere.set(xlabel="x", ylabel="y", zlabel="z")
ax_sphere.set_zticklabels([])
ax_sphere.set_yticklabels([])
ax_sphere.set_xticklabels([])

plt.savefig(f"{plots_dir}/direction_file{i_file}_event{i_event}.png")
# ---------------------

print(colored(f"Done plotting antenna signals for event{i_event} in file{i_file}!", "green", attrs=["bold"]))
print("")