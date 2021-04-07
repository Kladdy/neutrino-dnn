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
parser.add_argument("i_file", type=int, help="the id of file to do inference on")

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

# Load test file data and make predictions
    # Load first file
data, nu_direction = load_file(i_file)

# Amount of times to do 1-inferences:
times_mean = []
times_std = []

batch_sizes = np.logspace(np.log10(10), np.log10(10000), dtype=int)

for batch_size in batch_sizes:
    times = []

    N = min(100, int(np.floor(99000/batch_size)))

    # Make pedictions and time it
    for i in range(N):
        print(f"On step {i}/{N}...")
        data_tmp = data[(i)*batch_size+1:(i+1)*batch_size, :, :, :]
        #data_tmp = data_tmp[np.newaxis, :, :, :]
        print(data_tmp.shape)

        t0 = time.time()

        nu_direction_predict = model.predict(data_tmp)

        t = time.time() - t0
        if i != 0:
            times.append(t)

    print(times)

    mean = np.mean(times)
    std = np.std(times)

    times_mean.append(mean)
    times_std.append(std)

print(times_mean)
print(times_std)

cprint("Inference test for dl1 done!", "green")

