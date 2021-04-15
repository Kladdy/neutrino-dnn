# Imports
import argparse
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint
# -------

# Constants
plots_dir = "plots"

# Parse arguments
parser = argparse.ArgumentParser(description='Test inference speed on dl1 machine')
parser.add_argument("run_id", type=str, help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int, help="the id of file to do inference on")
parser.add_argument("n_threads", type=int, help="the amount of threads used when training")

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
n_threads = args.n_threads

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting plotting of inference test for Pi 3B...", "yellow")

with open(f'{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_test.npy', 'rb') as f:
    batch_sizes = np.load(f)
    times_mean = np.load(f)
    times_std = np.load(f)

fig, ax = plt.subplots(1,1)

ax.set_xscale("log")
ax.set_yscale("log")

ax.errorbar(batch_sizes, times_mean, fmt="o", color="mediumorchid", yerr=times_std)

ax.set(title=f'Time per inference over events per inference on Pi 3B with {n_threads} threads')
ax.set(xlabel="Events per inference")
ax.set(ylabel="Time per inference (s)")
plt.tight_layout()
# plt.xlabel("Batch size")
# plt.ylabel("Time")
plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_plot.png")

cprint("Inference plot for Pi 3B done!", "green")




