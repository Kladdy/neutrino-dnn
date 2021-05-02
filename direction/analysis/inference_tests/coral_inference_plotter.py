# Imports
import argparse
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from radiotools import plthelpers as php
from termcolor import cprint
from scipy.optimize import curve_fit
# -------

# Constants
plots_dir = "plots"

# Parse arguments
parser = argparse.ArgumentParser(description='Test inference speed on dl1 machine')
parser.add_argument("run_id", type=str, help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int, help="the id of file to do inference on")
parser.add_argument("n_events_to_load", type=int, help="amount of events to load")
parser.add_argument("--eps", dest="eps", action="store_true", help="if save as eps or not")
parser.add_argument('--fit', dest='fit', action='store_true', help="whether or not to do Rayleigh fit")
parser.set_defaults(eps=False)
parser.set_defaults(fit=False)

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
n_events_to_load = args.n_events_to_load
eps = args.eps
fit = args.fit

n_threads_list = [1,3]
threads_colors = ["dodgerblue", "forestgreen", "darkorange"]
threads_colors_fit = ["mediumorchid", "maroon", "royalblue"]

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting plotting of inference test for Coral Dev board...", "yellow")

fig, ax = plt.subplots(1,1)

x_logspace = np.logspace(np.log10(4), np.log10(150))
#x_logspace = np.array([4.0, 150.0])

with open(f'{plots_dir}/model_{run_name}_file_{i_file}_iterations_{n_events_to_load}_inference_test.npy', 'rb') as f:
    times = np.load(f)
    mean = np.load(f)
    std = np.load(f)

print(times)

bins = np.linspace(0.9, 1.1, 10)
fig, ax = php.get_histogram(times*1000, bins=bins,
                            xlabel=r"Time per inference (ms)", stats=False,
                            ylabel="Count", kwargs={'color':"lightsteelblue", 'ec':"k"})

# New fit method:
if fit:
    def gauss_fit(x, A, mu, sigma):
        return A * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2)

    ax = plt.gca()

    p = ax.patches

    xdata = [patch.get_x() for patch in p]
    ydata = [patch.get_height() for patch in p]

    #print(xdata)
    #print(ydata)

    # line = ax.lines[0]
    # xdata = line.get_xdata()
    # ydata = line.get_ydata()

    popt, pcov = curve_fit(gauss_fit, xdata, ydata, p0=[700, 5])

    #print("testing plotting...")
    #plt.plot(xdata, ydata, label="test")

    x_fit = np.linspace(0.8*min(xdata), 1.1*max(xdata), 200)

    popt_abs = np.abs(popt)
    sigma_value = popt_abs[1]

    plt.plot(x_fit, gauss_fit(x_fit, *popt), '-', color="darkorange",
         label=fr'fit: $\sigma={sigma_value:5.2f}$')



plt.title('Amount of time for one inference as a function of events per \ninference when inferencing on an Edge TPU Dev Board')

plt.tight_layout()

fig.legend(loc="upper left")

print("Do save as eps?", eps)

if eps:
    plt.savefig(f"{plots_dir}/FINAL_model_{run_name}_file_{i_file}_iterations_{n_events_to_load}_inference_plot_Coral.eps", format="eps")
else:
    plt.savefig(f"{plots_dir}/FINAL_model_{run_name}_file_{i_file}_iterations_{n_events_to_load}_inference_plot_Coral.png")

cprint("Inference plot for Coral Dev board!", "green")




