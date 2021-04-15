# Imports
import argparse
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from toolbox import load_file
from termcolor import cprint
from tensorflow.keras.models import load_model
# -------

# Constants
models_dir = "/mnt/md0/sstjaernholm/neutrino-dnn/final_models"
plots_dir = "plots"

# Parse arguments
parser = argparse.ArgumentParser(description='Test inference speed on dl1 machine')
parser.add_argument("run_id", type=str, help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_files", type=str, help="the ids of files to do inference on")

args = parser.parse_args()
run_id = args.run_id
i_files = args.i_files

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting plotting of inference test for dl1...", "yellow")

with open(f'{plots_dir}/model_{run_name}_file_{i_files}_inference_test.npy', 'rb') as f:
    batch_sizes = np.load(f)
    times_mean = np.load(f)
    times_std = np.load(f)

fig, ax = plt.subplots(1,1)

ax.set_xscale("log")
ax.set_yscale("log")

ax.plot(batch_sizes, times_mean, "*", color="mediumorchid")

ax.set(title='Amount of time for one inference as a function of events per inference')
ax.set(xlabel=r"Events per inference $N_{events, inf}$")
ax.set(ylabel=r"Time per inference t_{inf} (s)")
plt.tight_layout()
# plt.xlabel("Batch size")
# plt.ylabel("Time")
plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_files}_inference_plot.eps", format="eps")

cprint("Inference test for dl1 done!", "green")




