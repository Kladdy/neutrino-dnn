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
from constants import plots_dir, saved_model_dir, datapath, data_filename, label_filename, test_file_ids, run_version
from toolbox import get_pred_angle_diff_data
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
# -------

# Loading data and label files and also other properties
def load_file(i_file, norm=1e-6):
    t0 = time.time()
    print(f"loading file {i_file}", flush=True)

    # Load 300 MHz filter
    filt = np.load("bandpass_filters/300MHz_filter.npy")

#     t0 = time.time()
#     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]

    labels_tmp = np.load(os.path.join(datapath, f"{label_filename}{i_file:04d}.npy"), allow_pickle=True)
    print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction = hp.spherical_to_cartesian(nu_zenith, nu_azimuth)

    nu_energy = np.array(labels_tmp.item()["nu_energy"])
    nu_flavor = np.array(labels_tmp.item()["nu_flavor"])
    shower_energy = np.array(labels_tmp.item()["shower_energy"])

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    data /= norm

    nu_zenith = nu_zenith[idx]
    nu_azimuth = nu_azimuth[idx]
    nu_direction = nu_direction[idx]
    nu_energy = nu_energy[idx]
    nu_flavor = nu_flavor[idx]
    shower_energy = shower_energy[idx]

    return data, nu_direction, nu_zenith, nu_azimuth, nu_energy, nu_flavor, shower_energy

# Parse arguments
parser = argparse.ArgumentParser(description='Plot resolution as a function of different parameters')
parser.add_argument("run_id", type=str ,help="the id of the run to be analyzed, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name
run_name = f"run{run_id}"

print(colored(f"Plotting resolution as function of neutrino signal properties for {run_name}...", "yellow"))

# Make sure run_name is compatible with run_version
this_run_version = run_name.split(".")[0]
this_run_id = run_name.split(".")[1]
assert this_run_version == run_version, f"run_version ({run_version}) does not match the run version for this run ({this_run_version})"

# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# Load the model
model = keras.models.load_model(f'{saved_model_dir}/model.{run_name}.h5')

# Load test file data and make predictions
    # Load first file
data, nu_direction, nu_zenith, nu_azimuth, nu_energy, nu_flavor, shower_energy = load_file(test_file_ids[0])

    # Then load rest of files
if len(test_file_ids) > 1:
    for test_file_id in test_file_ids:
        if test_file_id != test_file_ids[0]:
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
binned_resolution_nu_energy = stats.binned_statistic(nu_energy, angle_difference_data, bins = nu_energy_bins_with_one_extra)[0]

ax.plot(nu_energy_bins, binned_resolution_nu_energy, "o")
# ax.set_ylim(0, 0.4)
ax.set_xlabel("nu energy (eV)")
ax.set_ylabel("angular resolution (°)")
ax.set_xscale('log')


# ax = fig_energy.add_subplot(1, 2, 2)
# ax.plot(nu_energy, angle_difference_data, 'o')
# ax.set_xscale('log')

plt.title(f"Mean resolution as a function of nu_energy for {run_name}")
fig_energy.tight_layout()
fig_energy.savefig(f"{plots_dir}/mean_resolution_nu_energy_{run_name}.png")
# ___________________________________

# --------- Azimuth plotting ---------
# Create figure
fig_azimuth = plt.figure()

# Calculate binned statistics
ax = fig_azimuth.add_subplot(1, 1, 1)
nu_azimuth_bins = np.linspace(0,2*np.pi, 30)
nu_azimuth_bins_with_one_extra = np.append(np.linspace(0,2*np.pi, 30), 2*np.pi+1)
binned_resolution_nu_azimuth = stats.binned_statistic(nu_azimuth, angle_difference_data, bins = nu_azimuth_bins_with_one_extra)[0]

ax.plot(nu_azimuth_bins / units.deg, binned_resolution_nu_azimuth, "o")
# ax.set_ylim(0, 0.4)
ax.set_xlabel("azimuth (°)")
ax.set_ylabel("angular resolution (°)")


plt.title(f"Mean resolution as a function of nu_azimuth for {run_name}")
fig_azimuth.tight_layout()
fig_azimuth.savefig(f"{plots_dir}/mean_resolution_nu_azimuth_{run_name}.png")
# ___________________________________

# --------- Zenith plotting ---------
# Create figure
fig_zenith = plt.figure()

# Calculate binned statistics
ax = fig_zenith.add_subplot(1, 1, 1)
nu_zenith_bins = np.linspace(0,np.pi, 30)
nu_zenith_bins_with_one_extra = np.append(np.linspace(0,np.pi, 30), np.pi+1)
binned_resolution_nu_zenith = stats.binned_statistic(nu_zenith, angle_difference_data, bins = nu_zenith_bins_with_one_extra)[0]

ax.plot(nu_zenith_bins / units.deg, binned_resolution_nu_zenith, "o")
# ax.set_ylim(0, 0.4)
ax.set_xlabel("zenith (°)")
ax.set_ylabel("angular resolution (°)")

plt.title(f"Mean resolution as a function of nu_zenith for {run_name}")
fig_zenith.tight_layout()
fig_zenith.savefig(f"{plots_dir}/mean_resolution_nu_zenith_{run_name}.png")
# ___________________________________

# --------- SNR plotting ---------
max_LPDA = np.max(np.max(np.abs(data[:, :, 0:4]), axis=1), axis=1)

# Create figure
fig_SNR = plt.figure()

# Calculate binned statistics
ax = fig_SNR.add_subplot(1, 1, 1)

SNR_means = np.arange(0.5, 20.5, 2)
SNR_bins = np.append(np.arange(0, 20, 2), [10000])

binned_resolution_SNR_mean = stats.binned_statistic(max_LPDA[:, 0] / 10., angle_difference_data, bins=SNR_bins)[0]

ax.plot(SNR_means, binned_resolution_SNR_mean, "o")
# ax.set_ylim(0, 0.4)
ax.set_xlabel("SNR")
ax.set_ylabel("angular resolution (°)")

plt.title(f"Mean resolution as a function of SNR for {run_name}")
fig_SNR.tight_layout()
fig_SNR.savefig(f"{plots_dir}/mean_resolution_SNR_{run_name}.png")
# ___________________________________

print(colored(f"Plotting angular resolution depending on properties for {run_name}!", "green", attrs=["bold"]))
print("")