# Imports
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse
from constants import run_version,  plots_dir
# -------

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Values
path = f"/mnt/md0/sstjaernholm/neutrino-dnn/direction/{run_version}/saved_models"
# ------

# Parse arguments
parser = argparse.ArgumentParser(description='Plot loss function')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

print(f"{bcolors.OKGREEN}Plotting loss for {run_name}...{bcolors.ENDC}")

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Reading data
df = pandas.read_csv(f'{path}/{filename}')

epoch = df['epoch']
loss = df['loss']
val_loss = df['val_loss']

# Find for which epoch the lowest val_loss is achieved
epoch_min = np.argmin(val_loss)

# Plotting
fig, axs = plt.plot(epoch, loss, "", epoch, val_loss)
plt.title(f'Model loss for {run_name}, min val_loss at epoch {epoch_min}')

plt.xlabel('epoch')
plt.legend(["loss", "val_loss"])
#fig.set_size_inches(12, 10)

plt.savefig(os.path.join(plots_dir, f"plot_loss_{run_name}.png"))
print(f"Saved loss plot for {run_name}!")