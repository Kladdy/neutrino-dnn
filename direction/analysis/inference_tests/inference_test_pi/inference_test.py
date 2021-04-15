# Imports
import argparse
import os
import time
import numpy as np
from constants import models_path
from matplotlib import pyplot as plt
from toolbox import load_file
from termcolor import cprint
from tflite_runtime.interpreter import Interpreter 
# -------

# Constants
plots_dir = "plots"

# Parse arguments
parser = argparse.ArgumentParser(description='Test inference speed on dl1 machine')
parser.add_argument("run_id", type=str, help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int, help="the id of file to do inference on")
parser.add_argument("n_events_to_load", type=int, help="amount of events to load")
parser.add_argument("n_threads", type=int, help="amount of threads to use")

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
n_events_to_load = args.n_events_to_load
n_threads = args.n_threads

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting inference test for Pi 3B...", "yellow")

# Load model
#model = load_model(f'{models_path}/model.{run_name}.h5')
path_to_model = f'{models_path}/model.{run_name}.tflite'
interpreter = Interpreter(path_to_model, num_threads=n_threads)
interpreter.allocate_tensors()

# Get input tensor.
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
print("Model loaded, input_shape:",input_shape)
print("Input details:", input_details)

# Load test file data
    # Load first file
data = load_file(i_file, n_events_to_load)

# Amount of times to do 1-inferences:
times_mean = []
times_std = []

batch_sizes = np.logspace(np.log10(5), np.log10(n_events_to_load/10.0), num=20, dtype=int)
# Remove duplicates
batch_sizes = list(set(batch_sizes))

print("Batch sizes: ", batch_sizes)
print("n_events_to_load/10.0", n_events_to_load/10.0)
print("data.shape", data.shape)

for batch_size in batch_sizes:
    times = []

    N = min(30, int(np.floor((n_events_to_load-1)/batch_size)))
    print("This time, N = ", N)
    # Make pedictions and time it
    for i in range(N):
        print(f"On step {i}/{N}...")
        data_tmp = data[(i)*batch_size+1:(i+1)*batch_size, :, :, :]
        #data_tmp = data_tmp[np.newaxis, :, :, :]
        print("Shape of data_tmp:", data_tmp.shape)

        data_tmp = np.array(data_tmp, dtype=np.float32)

        if i == 0:
            interpreter.resize_tensor_input(0, [data_tmp.shape[0], 5, 512, 1])
            interpreter.allocate_tensors()

        t0 = time.time()

        interpreter.set_tensor(input_details[0]['index'], data_tmp)

        interpreter.invoke()

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
plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_test.png")

with open(f'{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_test.npy', 'wb') as f:
    np.save(f, batch_sizes)
    np.save(f, times_mean)
    np.save(f, times_std)

cprint("Inference test for Pi 3B done!", "green")
