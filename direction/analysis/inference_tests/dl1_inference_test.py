# GPU allocation
from gpuutils import GpuUtils # pylint: disable=import-error
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------

# Imports
import argparse
import os
import time
import numpy as np
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
parser.add_argument("i_file", type=int, help="the id file to do inference on")

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting inference test for dl1...", "yellow")

# Load model
model = load_model(f'{models_dir}/model.{run_name}.h5')

data, nu_direction = load_file(i_file)

# Create list of amount of events to do inference on each prediction
amount_of_events_per_pred = np.logspace(np.log10(10**0), np.log(90000), 30, dtype=int)
print(amount_of_events_per_pred)
times = []

# Make pedictions and time it
for i in range(len(amount_of_events_per_pred)):
    t0 = time.time()

    nu_direction_predict = model.predict(data[1:amount_of_events_per_pred[i],:,:,:])

    t = time.time() - t0
    times.append(t)

print(amount_of_events_per_pred)
print(times)

cprint("Inference test for dl1 done!", "green")

