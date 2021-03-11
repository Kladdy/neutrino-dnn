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
from constants import plots_dir, saved_model_dir, datapath, data_filename, label_filename, test_file_ids
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
#     t0 = time.time()
#     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
    labels_tmp = np.load(os.path.join(datapath, f"{label_filename}{i_file:04d}.npy"), allow_pickle=True)
#     print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction = hp.spherical_to_cartesian(nu_zenith, nu_azimuth)

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    nu_direction = nu_direction[idx]
    data /= norm

    print(labels_tmp.item())

    return data, nu_direction


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
data, nu_direction = load_file(test_file_ids[0])

    # Then load rest of files
if len(test_file_ids) > 1:
    for test_file_id in test_file_ids:
        if test_file_id != test_file_ids[0]:
            data_tmp, nu_direction_tmp = load_file(test_file_id)

            data = np.concatenate((data, data_tmp))
            nu_direction = np.concatenate((nu_direction, nu_direction_tmp))

nu_direction_predict = model.predict(data)

angle_difference_data = np.array([hp.get_angle(nu_direction_predict[i], nu_direction[i]) for i in range(len(nu_direction))]) / units.deg



print(colored(f"Plotting angular resolution depending on properties for {run_name}!", "green", attrs=["bold"]))
print("")