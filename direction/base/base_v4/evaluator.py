# GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------

# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from tensorflow import keras
import os
import time
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
import argparse
from termcolor import colored
from toolbox import load_file
from constants import datapath, data_filename, label_filename, test_file_id
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate angular resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

print(colored(f"Evaluating angular resolution for {run_name}...", "yellow"))

# Load the model
model = keras.models.load_model(f'saved_models/model.{run_name}.h5')

# Load test file data
data, nu_direction = load_file(test_file_id)
nu_direction_predict = model.predict(data)

# Save predicted angles
with open(f'saved_models/model.{run_name}.h5_predicted.pkl', "bw") as fout:
    pickle.dump([nu_direction_predict, nu_direction], fout, protocol=4)

print(colored(f"Done evaluating angular resolution for {run_name}!", "green", attrs=["bold"]))
print("")