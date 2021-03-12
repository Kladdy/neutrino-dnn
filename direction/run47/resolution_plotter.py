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
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
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


fig = plt.figure()
# Calculate binned statistics
ax = fig.add_subplot(1, 2, 1)
nu_energy_bins = np.logspace(np.log10(1e17),np.log10(1e19), 30)
binned_nu_energy = stats.binned_statistic(nu_energy, angle_difference_data, bins = nu_energy_bins)
print(binned_nu_energy)
print(binned_nu_energy[0])


ax = fig.add_subplot(1, 2, 2)

# We can set the number of bins with the `bins` kwarg
ax.plot(nu_energy, angle_difference_data, 'o')
ax.set_xscale('log')

plt.title(f"Resolution as a function of nu_energy for {run_name}")
fig.savefig(f"{plots_dir}/resolution_nu_energy_{run_name}.png")

print(colored(f"Plotting angular resolution depending on properties for {run_name}!", "green", attrs=["bold"]))
print("")