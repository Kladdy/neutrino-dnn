# Imports
import argparse
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint
from scipy.optimize import curve_fit
# -------

def func(x, k, m):
    return k * x + m

# Constants
plots_dir = "plots"

# Parse arguments
parser = argparse.ArgumentParser(description='Test inference speed on dl1 machine')
parser.add_argument("run_id", type=str, help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int, help="the id of file to do inference on")

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file

n_threads_list = [1,3]
threads_colors = ["dodgerblue", "forestgreen", "maroon"]
threads_colors_fit = ["mediumorchid", "darkorange", "royalblue"]

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting plotting of inference test for Pi 3B...", "yellow")

fig, ax = plt.subplots(1,1)

x_logspace = np.logspace(np.log10(4), np.log10(150))
#x_logspace = np.array([4.0, 150.0])

for i in range(len(n_threads_list)):
    n_threads = n_threads_list[i]

    if n_threads == 1:
        thread_string = "thread"
    else:
        thread_string = "threads"

    with open(f'{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_test.npy', 'rb') as f:
        batch_sizes = np.load(f)
        times_mean = np.load(f)
        times_std = np.load(f)

    ax.set_xscale("log")
    ax.set_yscale("log")

    legend_string = f"data, {str(n_threads)} {thread_string}"
    print(legend_string)

    #ax.plot(batch_sizes, times_mean, "*", color=threads_colors[i], label=f"data, {n_threads} {thread_string}")
    ax.plot(batch_sizes, times_mean, "*", color=threads_colors[i], label=legend_string)

    # Make fit
    popt, pcov = curve_fit(func, batch_sizes, times_mean)

    ax.plot(x_logspace, func(x_logspace, *popt), color=threads_colors_fit[i],
        label=r'fit: $k=$%5.3f, $m=$%5.3f' % tuple(popt))

    
    # plt.xlabel("Batch size")
    # plt.ylabel("Time")





ax.set(title='Amount of time for one inference as a function of events\nper inference when inferencing on a Raspberry Pi 3B')
ax.set(xlabel=r"Events per inference $N_{events, inf}$")
ax.set(ylabel=r"Time per inference $t_{inf}$ (s)")
plt.xlim(4, 120)
plt.ylim(0.15, 10)
plt.tight_layout()


plt.legend(loc="upper left")

plt.savefig(f"{plots_dir}/FINAL_model_{run_name}_file_{i_file}_inference_plot_raspberryPi.eps", format="eps")
#plt.savefig(f"{plots_dir}/FINAL_model_{run_name}_file_{i_file}_inference_plot_raspberryPi.png")



#plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_plot.png")

cprint("Inference plot for Pi 3B done!", "green")




