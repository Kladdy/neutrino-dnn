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

# Load test file data
    # Load first file
data, nu_direction = load_file(test_file_ids[0])

    # Then load rest of files
if len(test_file_ids) > 1:
    for test_file_id in test_file_ids:
        if test_file_id != test_file_ids[0]:
            data_tmp, nu_direction_tmp = load_file(test_file_id)

            data = np.concatenate((data, data_tmp))
            nu_direction = np.concatenate((nu_direction, nu_direction_tmp))

# Amount of times to do 1-inferences:
times_mean = []
times_std = []

batch_sizes = np.logspace(np.log10(100), np.log10(1000), num=5, dtype=int)

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

fig, ax = plt.subplots(1,1)

ax.set_xscale("log")
ax.set_yscale("log")

ax.errorbar(batch_sizes, times_mean, fmt="o", yerr=times_std)

ax.set(title='Time per inference over events per inference')
ax.set(xlabel="Events per inference")
ax.set(ylabel="Time per inference (s)")
# plt.xlabel("Batch size")
# plt.ylabel("Time")
plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_files}_inference_test.png")

with open(f'{plots_dir}/model_{run_name}_file_{i_files}_inference_test.npy', 'wb') as f:
    np.save(f, batch_sizes)
    np.save(f, times_mean)
    np.save(f, times_std)

cprint("Inference test for dl1 done!", "green")

