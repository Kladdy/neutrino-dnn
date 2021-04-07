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

# Split file ids
test_file_ids = i_files.split(sep=",")
test_file_ids = [int(s) for s in test_file_ids]

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting inference test for dl1...", "yellow")

# Load model
model = load_model(f'{models_dir}/model.{run_name}.h5')

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


# Create list of amount of events to do inference on each prediction
amount_of_events_per_pred = np.logspace(np.log10(10**2), np.log10(290000), 20, dtype=int)
times = []

# Make pedictions and time it
for i in range(len(amount_of_events_per_pred)):
    print(f"On step {i}/{len(amount_of_events_per_pred)}...")
    data_tmp = data[0:amount_of_events_per_pred[i],:,:,:]

    t0 = time.time()

    nu_direction_predict = model.predict(data_tmp)

    t = time.time() - t0
    times.append(t)

print(amount_of_events_per_pred)
print(times)

plt.semilogx(amount_of_events_per_pred, times)
plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_file}_inference_test.png")

cprint("Inference test for dl1 done!", "green")

