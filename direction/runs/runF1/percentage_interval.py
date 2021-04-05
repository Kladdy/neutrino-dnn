# GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------

# Imports
import matplotlib.pyplot as plt
import numpy as np
from constants import plots_dir, saved_model_dir, dataset
from toolbox import get_pred_angle_diff_data, calculate_percentage_interval
import sys
import argparse
import os
import time
import pickle
from NuRadioReco.utilities import units
from scipy import stats
from termcolor import colored
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from radiotools import plthelpers as php
from tensorflow import keras
from radiotools import helper as hp
import functools
# -------

# Loading data and label files and also other properties
def load_file(i_file, norm=1e-6):
    t0 = time.time()
    print(f"loading file {i_file}", flush=True)

    # Load 500 MHz filter
    filt = np.load("bandpass_filters/500MHz_filter.npy")

    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]
    
    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
    print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith_data = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth_data = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction_data = hp.spherical_to_cartesian(nu_zenith_data, nu_azimuth_data)

    nu_energy_data = np.array(labels_tmp.item()["nu_energy"])
    nu_flavor_data = np.array(labels_tmp.item()["nu_flavor"])
    shower_energy_data = np.array(labels_tmp.item()["shower_energy"])

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    data /= norm

    nu_zenith_data = nu_zenith_data[idx]
    nu_azimuth_data = nu_azimuth_data[idx]
    nu_direction_data = nu_direction_data[idx]
    nu_energy_data = nu_energy_data[idx]
    nu_flavor_data = nu_flavor_data[idx]
    shower_energy_data = shower_energy_data[idx]

    return data, nu_direction_data, nu_zenith_data, nu_azimuth_data, nu_energy_data, nu_flavor_data, shower_energy_data

# Parse arguments
parser = argparse.ArgumentParser(description='Plot resolution as a function of different parameters')
parser.add_argument("run_id", type=str ,help="the id of the run to be analyzed, eg '3.2' for run3.2")
parser.add_argument("percentage_intervals", type=str, help="the percentage intervals, comma-separated, (20,50,68,80)")

args = parser.parse_args()
run_id = args.run_id
percentage_intervals = args.percentage_intervals

# Save the run name
run_name = f"run{run_id}"

# Parse percentage intervals
percentage_intervals = percentage_intervals.split(',')

print(colored(f"Plotting percentage plots for {run_name}...", "yellow"))

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Load the model
model = keras.models.load_model(f'{saved_model_dir}/model.{run_name}.h5')

# Load test file data and make predictions
    # Load first file
data, nu_direction, nu_zenith, nu_azimuth, nu_energy, nu_flavor, shower_energy = load_file(dataset.test_file_ids[0])

    # Then load rest of files
if len(dataset.test_file_ids) > 1:
    for test_file_id in dataset.test_file_ids:
        if test_file_id != dataset.test_file_ids[0]:
            data_tmp, nu_direction_tmp, nu_zenith_tmp, nu_azimuth_tmp, nu_energy_tmp, nu_flavor_tmp, shower_energy_tmp = load_file(test_file_id)

            data = np.concatenate((data, data_tmp))
            nu_direction = np.concatenate((nu_direction, nu_direction_tmp))
            nu_zenith = np.concatenate((nu_zenith, nu_zenith_tmp))
            nu_azimuth = np.concatenate((nu_azimuth, nu_azimuth_tmp))
            nu_energy = np.concatenate((nu_energy, nu_energy_tmp))
            nu_flavor = np.concatenate((nu_flavor, nu_flavor_tmp))
            shower_energy = np.concatenate((shower_energy, shower_energy_tmp))


# Get angle difference data
angle_difference_data = get_pred_angle_diff_data(run_name)

# --------- Energy plotting ---------
# Create figure
fig_energy = plt.figure()

# Calculate binned statistics
ax = fig_energy.add_subplot(1, 1, 1)
nu_energy_bins = np.logspace(np.log10(1e17),np.log10(1e19), 30)
nu_energy_bins_with_one_extra = np.append(np.logspace(np.log10(1e17),np.log10(1e19), 30), [1e20])


for i in len(percentage_intervals):
    percentage = float(f"0.{percentage_intervals[i]}")
    partial_func = functools.partial(calculate_percentage_interval, percentage=percentage)
    binned_resolution_nu_energy = stats.binned_statistic(nu_energy, angle_difference_data, bins = nu_energy_bins_with_one_extra, statistic=partial_func)[0]
    print(len(binned_resolution_nu_energy))
    print(type(binned_resolution_nu_energy))


ax.plot(nu_energy_bins, binned_resolution_nu_energy, "o")
# ax.set_ylim(0, 0.4)
ax.set_xlabel("true nu energy (eV)")
ax.set_ylabel("angular resolution (°)")
ax.set_xscale('log')

# ax = fig_energy.add_subplot(1, 2, 2)
# ax.plot(nu_energy, angle_difference_data, 'o')
# ax.set_xscale('log')

plt.title(f"Mean resolution as a function of nu_energy for {run_name}")
fig_energy.tight_layout()
fig_energy.savefig(f"{plots_dir}/mean_resolution_nu_energy_{run_name}.png")
# ___________________________________

print(colored(f"Done plotting percentage plots for {run_name}!", "green", attrs=["bold"]))
print("")