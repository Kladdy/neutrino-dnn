# GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------

import os
import numpy as np
import pickle
from tensorflow import keras
import time
#from toolbox import load_file
from constants import datapath, n_files, n_files_val, dataset, dataset_name, dataset_em
import datasets
import argparse
from matplotlib import pyplot as plt
from termcolor import colored
from generate_noise_realizations import load_one_file_properties, realize_noise

# Parse arguments
parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")
parser.add_argument("n_noise_iterations", type=int ,help="amount of noise relizations")

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
i_event = args.i_event
n_noise_iterations = args.n_noise_iterations

# Save the run name and filename
run_name = f"run{run_id}"

print(colored(f"Starting skymap evaluating for {run_name}, file {i_file}, event {i_event}...", "yellow"))

# Load the model
print("Loading model...")
model = keras.models.load_model(f'../models/model.{run_name}.h5')

# Load data and make noise realizations
print("Loading data and making noise realizations...")
data, nu_direction, nu_energy = load_one_file_properties(i_file, i_event)
noise_realized_data, noise_realized_direction = realize_noise(data, nu_direction, n_noise_iterations)

# Make predictions
print("Make predictions...")
nu_direction_prediction_noisy = model.predict(noise_realized_data)

# Print out norms
print("The following are the norms of prediction!")
normed_nu_direction = np.array([np.linalg.norm(v) for v in nu_direction_prediction_noisy])
print(normed_nu_direction)

# Save predicted angles
with open(f'plots/model.{run_name}.h5_predicted_file_{i_file}_{i_event}_{n_noise_iterations}.pkl', "bw") as fout:
    pickle.dump([nu_direction_prediction_noisy, nu_direction, nu_energy], fout, protocol=4)

print(colored(f"Done evaluating skymap for {run_name}!", "green", attrs=["bold"]))
print("")


